#!/usr/bin/env python3
"""
Generalized Floorplan Metadata, Geometry Validation & Core-Order Mapping (Stage C Step 7).

Dynamically inspects any 2D HotSpot floorplan (.flp, _tim.flp, .lcf) and its calibrated
Stage A resistance matrix (thermal_matrix_*.csv) without hardcoding:
  - active core count N
  - passive FILL block count M
  - core names or numeric ordering
  - Big vs. Little core IDs

Fails loudly on any geometry error, overlap, boundary violation, or core-order mismatch.
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

KELVIN_OFFSET: float = 273.15
AMBIENT_TEMP_C: float = 45.0
DEFAULT_DT_SECONDS: float = 0.01

BIG_CORE_POWER_RANGE_W: Tuple[float, float] = (10.0, 20.0)
LITTLE_CORE_POWER_RANGE_W: Tuple[float, float] = (5.0, 10.0)


@dataclass(frozen=True)
class FloorplanBlock:
    name: str
    width_m: float
    height_m: float
    x_m: float
    y_m: float
    is_fill: bool
    core_type: str  # "Big", "Little", or "FILL"
    power_range_w: Tuple[float, float]

    @property
    def width_mm(self) -> float:
        return self.width_m * 1e3

    @property
    def height_mm(self) -> float:
        return self.height_m * 1e3

    @property
    def area_mm2(self) -> float:
        return (self.width_m * 1e3) * (self.height_m * 1e3)


@dataclass(frozen=True)
class FloorplanSpec:
    full_name: str
    fp_dir: Path
    flp_path: Path
    tim_path: Path
    lcf_path: Path
    r_matrix_path: Path
    blocks: Tuple[FloorplanBlock, ...]
    active_cores: Tuple[FloorplanBlock, ...]
    passive_blocks: Tuple[FloorplanBlock, ...]
    all_block_names: Tuple[str, ...]
    active_core_names: Tuple[str, ...]
    passive_block_names: Tuple[str, ...]
    die_width_m: float
    die_height_m: float
    r_matrix: np.ndarray  # shape (N, N), strictly aligned to active_core_names

    @property
    def num_active_cores(self) -> int:
        return len(self.active_cores)

    @property
    def num_passive_blocks(self) -> int:
        return len(self.passive_blocks)

    @property
    def num_total_blocks(self) -> int:
        return len(self.blocks)


def classify_block_from_geometry(name: str, width_m: float, height_m: float) -> Tuple[bool, str, Tuple[float, float]]:
    """
    Classify block strictly from name (FILL vs active) and physical dimensions (Big vs Little).
    Never infers Big/Little from core index numbers.
    """
    if name.upper().startswith("FILL"):
        return True, "FILL", (0.0, 0.0)
    w_mm = width_m * 1e3
    h_mm = height_m * 1e3
    if math.isclose(w_mm, 4.0, abs_tol=1e-3) and math.isclose(h_mm, 4.0, abs_tol=1e-3):
        return False, "Big", BIG_CORE_POWER_RANGE_W
    if math.isclose(w_mm, 2.0, abs_tol=1e-3) and math.isclose(h_mm, 2.0, abs_tol=1e-3):
        return False, "Little", LITTLE_CORE_POWER_RANGE_W
    raise ValueError(
        f"Unsupported active core dimensions for '{name}': {w_mm:.4f} x {h_mm:.4f} mm. "
        f"Documented Stage A sizes are 4x4 mm (Big) and 2x2 mm (Little)."
    )


def parse_flp_blocks(flp_path: Path) -> List[FloorplanBlock]:
    if not flp_path.is_file():
        raise FileNotFoundError(f"FLP file not found: {flp_path}")
    blocks: List[FloorplanBlock] = []
    seen_names = set()
    for line_num, raw in enumerate(flp_path.read_text(encoding="utf-8").splitlines(), start=1):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid .flp line {line_num} in {flp_path}: '{s}'")
        name = parts[0]
        if name in seen_names:
            raise ValueError(f"Duplicate block name '{name}' in {flp_path}")
        seen_names.add(name)

        w, h, x, y = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        if not (math.isfinite(w) and math.isfinite(h) and math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"Non-finite coordinates in {flp_path} line {line_num}")
        if w <= 0.0 or h <= 0.0:
            raise ValueError(f"Non-positive dimensions for '{name}' in {flp_path}: w={w}, h={h}")
        if x < -1e-12 or y < -1e-12:
            raise ValueError(f"Negative coordinates for '{name}' in {flp_path}: x={x}, y={y}")

        is_fill, c_type, p_range = classify_block_from_geometry(name, w, h)
        blocks.append(
            FloorplanBlock(
                name=name,
                width_m=w,
                height_m=h,
                x_m=x,
                y_m=y,
                is_fill=is_fill,
                core_type=c_type,
                power_range_w=p_range,
            )
        )
    if not blocks:
        raise ValueError(f"No blocks found in {flp_path}")
    return blocks


def validate_blocks_geometry(
    blocks: List[FloorplanBlock],
    tim_path: Optional[Path] = None,
) -> Tuple[float, float]:
    """Validate zero pairwise overlaps and die boundary containment."""
    tol = 1e-9
    die_w = max(b.x_m + b.width_m for b in blocks)
    die_h = max(b.y_m + b.height_m for b in blocks)

    for b in blocks:
        if b.x_m + b.width_m > die_w + tol or b.y_m + b.height_m > die_h + tol:
            raise ValueError(f"Block '{b.name}' exceeds die bounds ({die_w} x {die_h})")

    for i in range(len(blocks)):
        a = blocks[i]
        ax1, ay1, ax2, ay2 = a.x_m, a.y_m, a.x_m + a.width_m, a.y_m + a.height_m
        for j in range(i + 1, len(blocks)):
            b = blocks[j]
            bx1, by1, bx2, by2 = b.x_m, b.y_m, b.x_m + b.width_m, b.y_m + b.height_m
            if (min(ax2, bx2) - max(ax1, bx1) > tol) and (min(ay2, by2) - max(ay1, by1) > tol):
                raise ValueError(f"Overlapping blocks detected: '{a.name}' and '{b.name}'")

    if tim_path is not None:
        if not tim_path.is_file():
            raise FileNotFoundError(f"TIM FLP file not found: {tim_path}")
        tim_lines = [
            l.strip()
            for l in tim_path.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
        if len(tim_lines) != 1:
            raise ValueError(f"Expected 1 block in {tim_path}, found {len(tim_lines)}")
        _, tw_s, th_s, tx_s, ty_s = tim_lines[0].split()
        tw, th, tx, ty = float(tw_s), float(th_s), float(tx_s), float(ty_s)
        if not (math.isclose(tw, die_w, abs_tol=1e-6) and math.isclose(th, die_h, abs_tol=1e-6)):
            raise ValueError(f"TIM dimensions ({tw}x{th}) != die dimensions ({die_w}x{die_h})")
        if not (math.isclose(tx, 0.0, abs_tol=1e-9) and math.isclose(ty, 0.0, abs_tol=1e-9)):
            raise ValueError(f"TIM origin ({tx}, {ty}) != (0, 0)")

    return die_w, die_h


def load_resistance_matrix_for_cores(
    r_csv_path: Path,
    expected_core_names: Tuple[str, ...],
) -> np.ndarray:
    """
    Load N x N Stage A resistance matrix and strictly verify that both row and column
    headers match expected_core_names in exact order.
    """
    if not r_csv_path.is_file():
        raise FileNotFoundError(f"Stage A resistance matrix not found: {r_csv_path}")
    with r_csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.reader(f) if any(cell.strip() for cell in r)]

    n_cores = len(expected_core_names)
    if len(rows) != n_cores + 1:
        raise ValueError(
            f"R matrix row count mismatch in {r_csv_path}: expected {n_cores + 1}, got {len(rows)}"
        )

    header_cores = tuple(c.strip() for c in rows[0][1:])
    if header_cores != expected_core_names:
        raise ValueError(
            f"R matrix column order mismatch in {r_csv_path}:\n"
            f"  Expected: {expected_core_names}\n"
            f"  Got:      {header_cores}"
        )

    mat = np.zeros((n_cores, n_cores), dtype=np.float64)
    row_names: List[str] = []
    for i, r in enumerate(rows[1:]):
        if len(r) != n_cores + 1:
            raise ValueError(f"Row {i+1} length {len(r)} != {n_cores + 1} in {r_csv_path}")
        row_names.append(r[0].strip())
        mat[i, :] = [float(x) for x in r[1:]]

    if tuple(row_names) != expected_core_names:
        raise ValueError(
            f"R matrix row order mismatch in {r_csv_path}:\n"
            f"  Expected: {expected_core_names}\n"
            f"  Got:      {tuple(row_names)}"
        )
    if not np.all(np.isfinite(mat)):
        raise ValueError(f"Non-finite values in R matrix {r_csv_path}")
    return mat


def load_floorplan_spec(
    fp_dir: Path,
    full_name: str,
    r_matrix_path: Optional[Path] = None,
) -> FloorplanSpec:
    """Load and validate a complete floorplan specification and its N x N Stage A R matrix."""
    fp_id = fp_dir.name
    fp_compact = fp_id.replace("_", "")
    flp_path = fp_dir / f"{fp_id}_layer_0.flp"
    tim_path = fp_dir / f"{fp_id}_tim.flp"
    lcf_path = fp_dir / f"{fp_id}.lcf"
    if r_matrix_path is None:
        r_matrix_path = fp_dir / "generated_samples" / f"thermal_matrix_{fp_compact}.csv"

    if not lcf_path.is_file():
        raise FileNotFoundError(f"LCF file not found: {lcf_path}")

    blocks = parse_flp_blocks(flp_path)
    die_w, die_h = validate_blocks_geometry(blocks, tim_path=tim_path)

    active_cores = tuple(b for b in blocks if not b.is_fill)
    passive_blocks = tuple(b for b in blocks if b.is_fill)
    if not active_cores:
        raise ValueError(f"No active cores found in {flp_path}")

    active_core_names = tuple(b.name for b in active_cores)
    passive_block_names = tuple(b.name for b in passive_blocks)
    all_block_names = tuple(b.name for b in blocks)

    r_mat = load_resistance_matrix_for_cores(r_matrix_path, active_core_names)

    return FloorplanSpec(
        full_name=full_name,
        fp_dir=fp_dir,
        flp_path=flp_path,
        tim_path=tim_path,
        lcf_path=lcf_path,
        r_matrix_path=r_matrix_path,
        blocks=tuple(blocks),
        active_cores=active_cores,
        passive_blocks=passive_blocks,
        all_block_names=all_block_names,
        active_core_names=active_core_names,
        passive_block_names=passive_block_names,
        die_width_m=die_w,
        die_height_m=die_h,
        r_matrix=r_mat,
    )


def parse_ptrace_for_spec(
    ptrace_path: Path,
    spec: FloorplanSpec,
    expected_steps: int = 200,
) -> np.ndarray:
    """
    Parse a .ptrace file, verify exact block column ordering against spec.all_block_names,
    verify FILL blocks are 0.0 W, and return active-core power array of shape (T, N)
    strictly aligned to spec.active_core_names.
    """
    lines = [
        line.strip()
        for line in ptrace_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if len(lines) < 2:
        raise ValueError(f"Empty or header-only ptrace: {ptrace_path}")

    headers = tuple(lines[0].split())
    if headers != spec.all_block_names:
        raise ValueError(
            f"Power trace column order mismatch in {ptrace_path}:\n"
            f"  Expected (.flp order): {spec.all_block_names}\n"
            f"  Got:                   {headers}"
        )

    rows = [[float(x) for x in line.split()] for line in lines[1:]]
    if expected_steps is not None and len(rows) != expected_steps:
        raise ValueError(f"Expected {expected_steps} rows in {ptrace_path}, got {len(rows)}")

    idx_map = {name: i for i, name in enumerate(headers)}
    for p_name in spec.passive_block_names:
        p_col = idx_map[p_name]
        for r_i, row in enumerate(rows, start=1):
            if row[p_col] != 0.0:
                raise ValueError(f"Non-zero passive block '{p_name}' power {row[p_col]} at row {r_i}")

    power_w = np.zeros((len(rows), spec.num_active_cores), dtype=np.float64)
    for c_i, c_name in enumerate(spec.active_core_names):
        col = idx_map[c_name]
        for r_i, row in enumerate(rows):
            power_w[r_i, c_i] = row[col]
    return power_w


def parse_ttrace_celsius_for_spec(
    ttrace_path: Path,
    spec: FloorplanSpec,
    expected_steps: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a HotSpot .ttrace file in Kelvin, verify exact column alignment with spec.all_block_names,
    convert active cores to Celsius (T_C = T_K - 273.15), and return (temp_k, temp_c) of shape (T, N).
    """
    lines = [
        line.strip()
        for line in ttrace_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if len(lines) < 2:
        raise ValueError(f"Empty or header-only ttrace: {ttrace_path}")

    headers = tuple(lines[0].split())
    if headers != spec.all_block_names:
        raise ValueError(
            f"Temperature trace column order mismatch in {ttrace_path}:\n"
            f"  Expected (.flp order): {spec.all_block_names}\n"
            f"  Got:                   {headers}"
        )

    rows = [[float(x) for x in line.split()] for line in lines[1:]]
    if expected_steps is not None and len(rows) != expected_steps:
        raise ValueError(f"Expected {expected_steps} rows in {ttrace_path}, got {len(rows)}")

    idx_map = {name: i for i, name in enumerate(headers)}
    temp_k = np.zeros((len(rows), spec.num_active_cores), dtype=np.float64)
    for c_i, c_name in enumerate(spec.active_core_names):
        col = idx_map[c_name]
        for r_i, row in enumerate(rows):
            val = row[col]
            if not math.isfinite(val) or val <= KELVIN_OFFSET:
                raise ValueError(f"Invalid Kelvin temperature {val} for '{c_name}' at row {r_i + 1}")
            temp_k[r_i, c_i] = val

    return temp_k, temp_k - KELVIN_OFFSET
