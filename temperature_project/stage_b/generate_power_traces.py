#!/usr/bin/env python3
"""Stage B Power-Trace Generator for HotSpot Thermal Simulations.

Parses floorplan (.flp) files, dynamically identifies Big cores (4x4 mm) and
Little cores (2x2 mm) by their actual geometry, and generates power traces
(.ptrace) strictly matching the floorplan's block header order.

Never assumes Core_0 and Core_1 are Big cores. Validates 2 Big and 6 Little
cores per floorplan and rejects inconsistent or unknown geometry.
"""
import argparse
from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

# Thermal model geometry constants
BIG_CORE_DIM_M = 0.004          # 4.0 mm
LITTLE_CORE_DIM_M = 0.002       # 2.0 mm
DIM_TOLERANCE_M = 1e-5          # 10 um tolerance for floating-point comparisons

# Reference power levels
BIG_CORE_REF_POWER_W = 20.0     # 20 W per Big core
LITTLE_CORE_REF_POWER_W = 8.0   # 8 W per Little core
FILL_REF_POWER_W = 0.0          # 0 W for passive silicon filler
TOTAL_REF_POWER_W = (2 * BIG_CORE_REF_POWER_W) + (6 * LITTLE_CORE_REF_POWER_W) # 88.0 W

# Standard power ranges for random workload generation (ML dataset)
DEFAULT_BIG_RANGE = (10.0, 20.0)
DEFAULT_LITTLE_RANGE = (5.0, 10.0)

REQUIRED_BIG_CORES = 2
REQUIRED_LITTLE_CORES = 6
REQUIRED_TOTAL_CORES = 8


class CoreType(str, Enum):
    BIG = "Big"
    LITTLE = "Little"
    FILL = "FILL"


class GeometryValidationError(ValueError):
    """Raised when a floorplan violates core count, dimensions, or block naming rules."""
    pass


class PowerAssignmentError(ValueError):
    """Raised when power specification or assignment is invalid."""
    pass


@dataclass(frozen=True)
class BlockInfo:
    name: str
    width: float
    height: float
    left_x: float
    bottom_y: float
    block_type: CoreType

    @property
    def area_m2(self) -> float:
        return self.width * self.height


@dataclass
class FloorplanProfile:
    flp_path: Path
    die_width: float
    die_height: float
    blocks: List[BlockInfo]
    big_cores: List[str]
    little_cores: List[str]
    fill_blocks: List[str]
    header_order: List[str]

    @property
    def num_big(self) -> int:
        return len(self.big_cores)

    @property
    def num_little(self) -> int:
        return len(self.little_cores)

    @property
    def num_fills(self) -> int:
        return len(self.fill_blocks)

    @property
    def total_active_area_m2(self) -> float:
        return (self.num_big * BIG_CORE_DIM_M * BIG_CORE_DIM_M +
                self.num_little * LITTLE_CORE_DIM_M * LITTLE_CORE_DIM_M)


@dataclass
class PowerTrace:
    headers: List[str]
    powers: List[float]
    sample_id: Optional[str] = None

    @property
    def total_power_w(self) -> float:
        return sum(self.powers)

    def power_dict(self) -> Dict[str, float]:
        return dict(zip(self.headers, self.powers))


def read_flp_lines(path: Union[str, Path]) -> List[Tuple[str, float, float, float, float]]:
    """Parse a HotSpot .flp file into (name, width, height, left_x, bottom_y) tuples."""
    flp_path = Path(path)
    if not flp_path.is_file():
        raise FileNotFoundError(f"Floorplan file not found: {flp_path}")

    raw_blocks = []
    content = flp_path.read_text(encoding="utf-8")
    for line_idx, line in enumerate(content.splitlines(), start=1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise GeometryValidationError(
                f"{flp_path}:{line_idx}: expected 5 fields (name, width, height, left-x, bottom-y), got {len(parts)}"
            )
        name = parts[0]
        try:
            w, h, x, y = (float(v) for v in parts[1:])
        except ValueError as exc:
            raise GeometryValidationError(
                f"{flp_path}:{line_idx}: non-numeric coordinates for block '{name}': {exc}"
            ) from exc

        if not all(math.isfinite(v) for v in (w, h, x, y)) or w <= 0 or h <= 0 or x < 0 or y < 0:
            raise GeometryValidationError(
                f"{flp_path}:{line_idx}: block '{name}' has non-positive size or negative position: "
                f"w={w}, h={h}, x={x}, y={y}"
            )
        raw_blocks.append((name, w, h, x, y))

    if not raw_blocks:
        raise GeometryValidationError(f"{flp_path}: floorplan file contains no blocks")
    return raw_blocks


def classify_block(name: str, width: float, height: float, tol: float = DIM_TOLERANCE_M) -> CoreType:
    """Classify a block based on its name prefix and physical dimensions."""
    if name.startswith("Core_"):
        is_big = (math.isclose(width, BIG_CORE_DIM_M, abs_tol=tol) and
                  math.isclose(height, BIG_CORE_DIM_M, abs_tol=tol))
        is_little = (math.isclose(width, LITTLE_CORE_DIM_M, abs_tol=tol) and
                     math.isclose(height, LITTLE_CORE_DIM_M, abs_tol=tol))
        if is_big:
            return CoreType.BIG
        if is_little:
            return CoreType.LITTLE
        raise GeometryValidationError(
            f"Core '{name}' has non-standard dimensions: {width*1e3:.3f}x{height*1e3:.3f} mm. "
            f"Expected Big (4x4 mm) or Little (2x2 mm)."
        )

    if name.startswith("FILL_"):
        return CoreType.FILL

    raise GeometryValidationError(
        f"Unknown block name '{name}'. Expected 'Core_*' or 'FILL_*' prefix."
    )


def analyze_floorplan(path: Union[str, Path], tol: float = DIM_TOLERANCE_M) -> FloorplanProfile:
    """Analyze a floorplan file, validating core counts, dimensions, and naming."""
    flp_path = Path(path).resolve()
    raw_blocks = read_flp_lines(flp_path)

    seen_names = set()
    blocks: List[BlockInfo] = []
    big_cores: List[str] = []
    little_cores: List[str] = []
    fill_blocks: List[str] = []
    header_order: List[str] = []

    max_x = 0.0
    max_y = 0.0

    for name, w, h, x, y in raw_blocks:
        if name in seen_names:
            raise GeometryValidationError(f"{flp_path}: duplicate block name '{name}'")
        seen_names.add(name)

        b_type = classify_block(name, w, h, tol=tol)
        block = BlockInfo(name=name, width=w, height=h, left_x=x, bottom_y=y, block_type=b_type)
        blocks.append(block)
        header_order.append(name)

        if b_type == CoreType.BIG:
            big_cores.append(name)
        elif b_type == CoreType.LITTLE:
            little_cores.append(name)
        elif b_type == CoreType.FILL:
            fill_blocks.append(name)

        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    # Validate core counts
    if len(big_cores) != REQUIRED_BIG_CORES:
        raise GeometryValidationError(
            f"{flp_path.name}: expected exactly {REQUIRED_BIG_CORES} Big cores (4x4 mm), found {len(big_cores)} ({big_cores})"
        )
    if len(little_cores) != REQUIRED_LITTLE_CORES:
        raise GeometryValidationError(
            f"{flp_path.name}: expected exactly {REQUIRED_LITTLE_CORES} Little cores (2x2 mm), found {len(little_cores)} ({little_cores})"
        )
    total_cores = len(big_cores) + len(little_cores)
    if total_cores != REQUIRED_TOTAL_CORES:
        raise GeometryValidationError(
            f"{flp_path.name}: expected {REQUIRED_TOTAL_CORES} active cores, found {total_cores}"
        )

    return FloorplanProfile(
        flp_path=flp_path,
        die_width=max_x,
        die_height=max_y,
        blocks=blocks,
        big_cores=big_cores,
        little_cores=little_cores,
        fill_blocks=fill_blocks,
        header_order=header_order,
    )


def generate_reference_trace(profile_or_path: Union[FloorplanProfile, Path, str]) -> PowerTrace:
    """Generate the standard reference power trace (20 W Big, 8 W Little, 0 W FILL).

    Total power: exactly 88.0 W. Order strictly matches the floorplan blocks.
    """
    profile = (profile_or_path if isinstance(profile_or_path, FloorplanProfile)
               else analyze_floorplan(profile_or_path))

    powers: List[float] = []
    for b in profile.blocks:
        if b.block_type == CoreType.BIG:
            powers.append(BIG_CORE_REF_POWER_W)
        elif b.block_type == CoreType.LITTLE:
            powers.append(LITTLE_CORE_REF_POWER_W)
        elif b.block_type == CoreType.FILL:
            powers.append(FILL_REF_POWER_W)
        else:
            raise PowerAssignmentError(f"Unhandled block type {b.block_type} for '{b.name}'")

    trace = PowerTrace(headers=list(profile.header_order), powers=powers, sample_id="reference")

    # Sanity checks
    if not math.isclose(trace.total_power_w, TOTAL_REF_POWER_W, abs_tol=1e-5):
        raise PowerAssignmentError(
            f"Reference trace total power is {trace.total_power_w:.2f} W, expected {TOTAL_REF_POWER_W:.2f} W"
        )
    return trace


def generate_custom_trace(
    profile_or_path: Union[FloorplanProfile, Path, str],
    core_powers: Dict[str, float],
    fill_power: float = 0.0,
    sample_id: Optional[str] = None
) -> PowerTrace:
    """Generate a power trace from a mapping of core names to power values."""
    profile = (profile_or_path if isinstance(profile_or_path, FloorplanProfile)
               else analyze_floorplan(profile_or_path))

    active_core_names = set(profile.big_cores + profile.little_cores)
    provided_core_names = set(core_powers.keys())

    missing = active_core_names - provided_core_names
    if missing:
        raise PowerAssignmentError(f"Missing power values for cores: {sorted(missing)}")

    extra = provided_core_names - active_core_names
    if extra:
        raise PowerAssignmentError(f"Provided powers for non-active blocks or unknown cores: {sorted(extra)}")

    if not math.isfinite(fill_power) or fill_power != 0.0:
        raise PowerAssignmentError(f"Passive FILL blocks must have 0.0 W, requested: {fill_power} W")

    powers: List[float] = []
    for b in profile.blocks:
        if b.block_type in (CoreType.BIG, CoreType.LITTLE):
            p = float(core_powers[b.name])
            if not math.isfinite(p) or p < 0:
                raise PowerAssignmentError(f"Negative power {p} W specified for core '{b.name}'")
            powers.append(p)
        else:
            powers.append(fill_power)

    return PowerTrace(headers=list(profile.header_order), powers=powers, sample_id=sample_id)


def generate_random_traces(
    profile_or_path: Union[FloorplanProfile, Path, str],
    num_samples: int,
    seed: Optional[int] = None,
    big_range: Tuple[float, float] = DEFAULT_BIG_RANGE,
    little_range: Tuple[float, float] = DEFAULT_LITTLE_RANGE,
    include_reference: bool = True,
    round_decimals: int = 2
) -> List[PowerTrace]:
    """Generate a series of unique power traces for ML dataset training.

    Args:
        profile_or_path: FloorplanProfile or path to .flp file.
        num_samples: Total number of samples to generate (>= 1).
        seed: Random seed for exact reproducibility.
        big_range: (min_w, max_w) for Big cores.
        little_range: (min_w, max_w) for Little cores.
        include_reference: If True, Sample 1 is the 20W/8W reference workload.
        round_decimals: Decimal rounding for power values (default: 2).
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")

    profile = (profile_or_path if isinstance(profile_or_path, FloorplanProfile)
               else analyze_floorplan(profile_or_path))

    b_min, b_max = big_range
    l_min, l_max = little_range
    if not all(math.isfinite(v) for v in big_range) or b_min <= 0 or b_max < b_min:
        raise ValueError(f"Invalid Big core power range: {big_range}")
    if not all(math.isfinite(v) for v in little_range) or l_min <= 0 or l_max < l_min:
        raise ValueError(f"Invalid Little core power range: {little_range}")

    if type(round_decimals) is not int or not 0 <= round_decimals <= 6:
        raise ValueError("round_decimals must be an integer from 0 through 6")
    scale = 10 ** round_decimals
    b_ticks = (math.ceil(b_min * scale), math.floor(b_max * scale))
    l_ticks = (math.ceil(l_min * scale), math.floor(l_max * scale))
    if b_ticks[0] > b_ticks[1] or l_ticks[0] > l_ticks[1]:
        raise ValueError("Power range has no representable value at the requested precision")
    capacity = (b_ticks[1]-b_ticks[0]+1)**2 * (l_ticks[1]-l_ticks[0]+1)**6
    if include_reference and not (b_min <= 20 <= b_max and l_min <= 8 <= l_max):
        capacity += 1
    if num_samples > capacity:
        raise ValueError("Requested more unique configurations than the ranges permit")
    rng = random.Random(seed)
    traces: List[PowerTrace] = []
    seen_active_tuples = set()

    start_idx = 1
    if include_reference:
        ref_trace = generate_reference_trace(profile)
        ref_trace.sample_id = f"sample_{start_idx:03d}"
        active_tuple = tuple(round(ref_trace.power_dict()[c], round_decimals)
                             for c in profile.big_cores + profile.little_cores)
        seen_active_tuples.add(active_tuple)
        traces.append(ref_trace)
        start_idx += 1

    max_attempts = max(num_samples * 100, 5000)
    attempts = 0
    while len(traces) < num_samples:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not generate {num_samples} unique power traces after {attempts} attempts."
            )

        core_powers: Dict[str, float] = {}
        for c in profile.big_cores:
            core_powers[c] = rng.randint(*b_ticks) / scale
        for c in profile.little_cores:
            core_powers[c] = rng.randint(*l_ticks) / scale

        active_tuple = tuple(core_powers[c] for c in profile.big_cores + profile.little_cores)
        if active_tuple in seen_active_tuples:
            continue
        seen_active_tuples.add(active_tuple)

        sample_num = len(traces) + 1
        trace = generate_custom_trace(
            profile,
            core_powers=core_powers,
            fill_power=0.0,
            sample_id=f"sample_{sample_num:03d}"
        )
        traces.append(trace)

    return traces


def format_ptrace(
    trace: PowerTrace,
    delimiter: str = "\t",
    float_format: str = "{:.2f}"
) -> str:
    """Format a PowerTrace into standard HotSpot .ptrace file content."""
    header_line = delimiter.join(trace.headers)
    power_line = delimiter.join(float_format.format(p) for p in trace.powers)
    return f"{header_line}\n{power_line}\n"


def write_ptrace(
    path: Union[str, Path],
    trace: PowerTrace,
    delimiter: str = "\t",
    float_format: str = "{:.2f}"
) -> Path:
    """Write a PowerTrace to a file, creating parent directories if needed."""
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = format_ptrace(trace, delimiter=delimiter, float_format=float_format)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate HotSpot power traces dynamically bound to floorplan core geometry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--flp", type=Path, help="Path to single floorplan (.flp) file")
    p.add_argument("--dir", type=Path, help="Directory containing floorplans (searched recursively for *_layer_0.flp)")
    p.add_argument("--output", "-o", type=Path, help="Output .ptrace path (when --flp is specified)")
    p.add_argument("--output-dir", type=Path, help="Output directory for generated traces")
    p.add_argument("--mode", choices=["reference", "random", "validate"], default="reference",
                   help="Mode: reference (20W/8W), random (ML samples), or validate (check geometry without writing)")
    p.add_argument("--samples", "-n", type=int, default=1, help="Number of samples to generate in random mode")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducible random workloads")
    p.add_argument("--big-range", nargs=2, type=float, default=DEFAULT_BIG_RANGE, metavar=("MIN", "MAX"),
                   help="Big core power range in Watts")
    p.add_argument("--little-range", nargs=2, type=float, default=DEFAULT_LITTLE_RANGE, metavar=("MIN", "MAX"),
                   help="Little core power range in Watts")
    p.add_argument("--delimiter", choices=["tab", "space"], default="tab",
                   help="Delimiter between columns in .ptrace file")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.flp and not args.dir:
        parser.error("Either --flp or --dir must be specified.")

    sep = "\t" if args.delimiter == "tab" else " "

    flp_files: List[Path] = []
    if args.flp:
        flp_files.append(args.flp.resolve())
    if args.dir:
        search_dir = args.dir.resolve()
        if not search_dir.is_dir():
            print(f"Error: directory does not exist: {search_dir}", file=sys.stderr)
            return 1
        found = sorted(search_dir.glob("**/*_layer_0.flp"))
        if not found:
            # Also check flat flp
            found = sorted(search_dir.glob("*.flp"))
        flp_files.extend(found)

    # De-duplicate
    flp_files = sorted(set(flp_files))
    if not flp_files:
        print("No floorplan (.flp) files found to process.", file=sys.stderr)
        return 1

    print(f"Discovered {len(flp_files)} floorplan(s) to process in mode '{args.mode}'.")
    has_errors = False

    for flp_path in flp_files:
        try:
            profile = analyze_floorplan(flp_path)
            print(
                f"\n[OK] {flp_path.name}: {profile.num_big} Big ({', '.join(profile.big_cores)}), "
                f"{profile.num_little} Little ({', '.join(profile.little_cores)}), {profile.num_fills} FILL blocks"
            )

            if args.mode == "validate":
                continue

            if args.mode == "reference":
                trace = generate_reference_trace(profile)
                if args.output and len(flp_files) == 1:
                    out_path = args.output.resolve()
                elif args.output_dir:
                    out_path = args.output_dir.resolve() / f"{flp_path.stem.replace('_layer_0', '')}_power.ptrace"
                else:
                    out_path = flp_path.with_name(f"{flp_path.stem.replace('_layer_0', '')}_power.ptrace")

                write_ptrace(out_path, trace, delimiter=sep)
                print(f"     Wrote reference trace (Total: {trace.total_power_w:.1f} W) -> {out_path}")

            elif args.mode == "random":
                traces = generate_random_traces(
                    profile,
                    num_samples=args.samples,
                    seed=args.seed,
                    big_range=tuple(args.big_range),
                    little_range=tuple(args.little_range),
                    include_reference=True
                )
                target_dir = args.output_dir.resolve() if args.output_dir else flp_path.parent / "generated_samples/ptrace"
                target_dir.mkdir(parents=True, exist_ok=True)
                for t in traces:
                    out_path = target_dir / f"{t.sample_id}.ptrace"
                    write_ptrace(out_path, t, delimiter=sep)
                print(f"     Wrote {len(traces)} power traces to {target_dir}")

        except Exception as exc:
            print(f"[ERROR] {flp_path}: {exc}", file=sys.stderr)
            has_errors = True

    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
