"""
Production Stage B v2 Feature Pipeline (Permutation-Equivariant Physics Features).

Computes the audited 63-feature per-core representation (feature set F4:
self_global + pairwise_coupling + die_spreading + boundary_images) for 2D
steady-state 8-core thermal prediction.

Original Stage B v1 252-feature builder (`stage_b/ml/features.py`) is untouched.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import numpy as np

VERSION = 2
AMBIENT_C = 45.0

PRODUCTION_GROUPS: Tuple[str, ...] = (
    "self_global",
    "pairwise_coupling",
    "die_spreading",
    "boundary_images",
)


def build_per_core_features(
    geometry_mm: np.ndarray,
    powers_w: np.ndarray,
    die_mm: np.ndarray,
    groups: Sequence[str] = PRODUCTION_GROUPS,
) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    """
    Build permutation-equivariant per-core physical features.

    Parameters
    ----------
    geometry_mm : np.ndarray, shape (N, 8, 4) or (8, 4)
        [width_mm, height_mm, x_mm, y_mm] for each of the 8 cores.
    powers_w : np.ndarray, shape (N, 8) or (8,)
        Power in Watts for each of the 8 cores.
    die_mm : np.ndarray, shape (N, 2) or (2,)
        [die_width_mm, die_height_mm].
    groups : sequence of str
        Feature groups to compute (defaults to audited PRODUCTION_GROUPS).

    Returns
    -------
    X : np.ndarray, shape (N * 8, F)
        Row `n * 8 + i` contains the physical feature vector for core `i` of sample `n`.
    definitions : list of dict
        Ordered feature metadata [{'name': ..., 'group': ..., 'unit': ...}, ...].
    """
    g = np.asarray(geometry_mm, dtype=np.float64)
    p = np.asarray(powers_w, dtype=np.float64)
    die = np.asarray(die_mm, dtype=np.float64)
    if g.ndim == 2:
        g = g[None, :, :]
        p = p[None, :]
        die = die[None, :]

    N = g.shape[0]
    w = g[:, :, 0]
    h = g[:, :, 1]
    x = g[:, :, 2]
    y = g[:, :, 3]
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    area = w * h
    is_big = np.isclose(w, 4.0, atol=1e-5).astype(np.float64)
    pdens = p / np.maximum(area, 1e-6)

    die_w = die[:, 0:1]
    die_h = die[:, 1:2]
    die_area = die_w * die_h
    sqrt_area = np.sqrt(np.maximum(die_area, 1e-6))
    die_min = np.minimum(die_w, die_h)
    die_max = np.maximum(die_w, die_h)
    aspect_ratio = die_max / np.maximum(die_min, 1e-6)
    die_perim = 2.0 * (die_w + die_h)
    active_area = np.sum(area, axis=1, keepdims=True)
    whitespace = 1.0 - active_area / np.maximum(die_area, 1e-6)

    p_tot = np.sum(p, axis=1, keepdims=True)
    p_other = p_tot - p
    p_big_tot = np.sum(p * is_big, axis=1, keepdims=True)
    p_little_tot = p_tot - p_big_tot

    dx = cx[:, :, None] - cx[:, None, :]
    dy = cy[:, :, None] - cy[:, None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    gap_x = np.maximum(np.abs(dx) - 0.5 * (w[:, :, None] + w[:, None, :]), 0.0)
    gap_y = np.maximum(np.abs(dy) - 0.5 * (h[:, :, None] + h[:, None, :]), 0.0)
    edge_gap = np.sqrt(gap_x * gap_x + gap_y * gap_y)

    off_diag = 1.0 - np.eye(8, dtype=np.float64)[None, :, :]
    p_src = p[:, None, :] * off_diag
    pdens_src = pdens[:, None, :] * off_diag
    big_src = is_big[:, None, :] * off_diag
    little_src = (1.0 - is_big[:, None, :]) * off_diag

    feats: List[np.ndarray] = []
    defs: List[Dict[str, str]] = []

    def add(name: str, values: np.ndarray, group: str, unit: str) -> None:
        arr = np.broadcast_to(values, (N, 8))
        feats.append(arr.reshape(N * 8))
        defs.append({"name": name, "group": group, "unit": unit})

    if "self_global" in groups:
        grp = "self_global"
        add("self.power_w", p, grp, "W")
        add("self.power_density_w_mm2", pdens, grp, "W/mm2")
        add("self.area_mm2", area, grp, "mm2")
        add("self.is_big", is_big, grp, "binary")
        add("self.power_x_is_big", p * is_big, grp, "W")
        add("self.power_x_is_little", p * (1.0 - is_big), grp, "W")
        add("global.total_power_w", p_tot, grp, "W")
        add("global.other_power_w", p_other, grp, "W")
        add("global.big_power_w", p_big_tot, grp, "W")
        add("global.little_power_w", p_little_tot, grp, "W")
        add("self.power_fraction", p / np.maximum(p_tot, 1e-6), grp, "fraction")

    if "pairwise_coupling" in groups:
        grp = "pairwise_coupling"
        for c in (0.5, 1.0, 2.0):
            add(f"coupling.sum_p_over_d_plus_{c}", np.sum(p_src / (dist + c), axis=2), grp, "W/mm")
        for c2 in (1.0, 4.0):
            add(f"coupling.sum_p_over_d2_plus_{c2}", np.sum(p_src / (dist * dist + c2), axis=2), grp, "W/mm2")
        for lam in (1.5, 3.0, 6.0, 12.0):
            add(f"coupling.sum_p_exp_d_{lam}", np.sum(p_src * np.exp(-dist / lam), axis=2), grp, "W")
        add("coupling.sum_p_log_kernel", np.sum(p_src * np.log1p(10.0 / (dist + 0.5)), axis=2), grp, "W")
        add("coupling.sum_p_over_gap_0.5", np.sum(p_src / (edge_gap + 0.5), axis=2), grp, "W/mm")
        add("coupling.sum_p_exp_gap_2.0", np.sum(p_src * np.exp(-edge_gap / 2.0), axis=2), grp, "W")
        add("coupling.sum_pdens_over_d_1.0", np.sum(pdens_src / (dist + 1.0), axis=2), grp, "W/mm3")
        add("coupling.big_sum_p_over_d_1.0", np.sum((p_src * big_src) / (dist + 1.0), axis=2), grp, "W/mm")
        add("coupling.little_sum_p_over_d_1.0", np.sum((p_src * little_src) / (dist + 1.0), axis=2), grp, "W/mm")
        add("coupling.big_sum_p_exp_3.0", np.sum((p_src * big_src) * np.exp(-dist / 3.0), axis=2), grp, "W")
        add("coupling.little_sum_p_exp_3.0", np.sum((p_src * little_src) * np.exp(-dist / 3.0), axis=2), grp, "W")
        c_exp3 = np.sum(p_src * np.exp(-dist / 3.0), axis=2)
        c_inv1 = np.sum(p_src / (dist + 1.0), axis=2)
        add("coupling.is_big_x_sum_p_exp_3.0", is_big * c_exp3, grp, "W")
        add("coupling.is_big_x_sum_p_over_d_1.0", is_big * c_inv1, grp, "W/mm")
        dist_masked = dist + (1.0 - off_diag) * 1e6
        min_d = np.min(dist_masked, axis=2)
        mean_d = np.sum(dist * off_diag, axis=2) / 7.0
        nearest_idx = np.argmin(dist_masked, axis=2)
        nearest_p = np.take_along_axis(p, nearest_idx, axis=1)
        add("geom.min_neighbor_dist_mm", min_d, grp, "mm")
        add("geom.mean_neighbor_dist_mm", mean_d, grp, "mm")
        add("coupling.nearest_p_over_d", nearest_p / (min_d + 0.5), grp, "W/mm")

    if "die_spreading" in groups:
        grp = "die_spreading"
        add("die.width_mm", die_w, grp, "mm")
        add("die.height_mm", die_h, grp, "mm")
        add("die.min_dim_mm", die_min, grp, "mm")
        add("die.max_dim_mm", die_max, grp, "mm")
        add("die.area_mm2", die_area, grp, "mm2")
        add("die.aspect_ratio", aspect_ratio, grp, "ratio")
        add("die.whitespace_fraction", whitespace, grp, "fraction")
        add("spread.p_tot_over_area", p_tot / np.maximum(die_area, 1.0), grp, "W/mm2")
        add("spread.p_tot_over_sqrt_area", p_tot / sqrt_area, grp, "W/mm")
        add("spread.p_tot_over_perim", p_tot / np.maximum(die_perim, 1.0), grp, "W/mm")
        add("spread.p_tot_over_min_dim", p_tot / np.maximum(die_min, 1.0), grp, "W/mm")
        add("spread.p_tot_x_aspect_ratio", p_tot * aspect_ratio, grp, "W")
        add("spread.p_tot_x_whitespace", p_tot * whitespace, grp, "W")
        add("spread.p_tot_x_log_area", p_tot * np.log(np.maximum(die_area, 1.0)), grp, "W")
        add("spread.self_p_over_sqrt_area", p / sqrt_area, grp, "W/mm")
        add("spread.self_p_over_area", p / np.maximum(die_area, 1.0), grp, "W/mm2")

    if "boundary_images" in groups:
        grp = "boundary_images"
        d_left = x
        d_right = np.maximum(die_w - (x + w), 0.0)
        d_bot = y
        d_top = np.maximum(die_h - (y + h), 0.0)
        c_left = cx
        c_right = np.maximum(die_w - cx, 0.0)
        c_bot = cy
        c_top = np.maximum(die_h - cy, 0.0)

        edge_stack = np.stack([d_left, d_right, d_bot, d_top], axis=2)
        edge_sorted = np.sort(edge_stack, axis=2)
        min_edge = edge_sorted[:, :, 0]
        second_edge = edge_sorted[:, :, 1]

        rel_x = np.abs(cx / np.maximum(die_w, 1e-6) - 0.5) * 2.0
        rel_y = np.abs(cy / np.maximum(die_h, 1e-6) - 0.5) * 2.0
        rel_r = np.sqrt(rel_x * rel_x + rel_y * rel_y)

        add("boundary.min_edge_dist_mm", min_edge, grp, "mm")
        add("boundary.second_edge_dist_mm", second_edge, grp, "mm")
        add("boundary.rel_center_dist", rel_r, grp, "ratio")
        add("boundary.rel_x_abs", rel_x, grp, "ratio")
        add("boundary.rel_y_abs", rel_y, grp, "ratio")

        wall_c = np.stack([c_left, c_right, c_bot, c_top], axis=2)
        self_img_d = 2.0 * wall_c
        self_img_inv = np.sum(1.0 / (self_img_d + 1.0), axis=2)
        self_img_exp3 = np.sum(np.exp(-self_img_d / 3.0), axis=2)
        self_img_exp6 = np.sum(np.exp(-self_img_d / 6.0), axis=2)
        add("boundary.self_p_x_img_inv", p * self_img_inv, grp, "W/mm")
        add("boundary.self_p_x_img_exp3", p * self_img_exp3, grp, "W")
        add("boundary.self_p_x_img_exp6", p * self_img_exp6, grp, "W")
        add("boundary.self_pdens_x_img_exp3", pdens * self_img_exp3, grp, "W/mm2")

        d_img_l = np.sqrt((cx[:, :, None] + cx[:, None, :]) ** 2 + dy * dy)
        d_img_r = np.sqrt((2.0 * die_w[:, :, None] - cx[:, :, None] - cx[:, None, :]) ** 2 + dy * dy)
        d_img_b = np.sqrt(dx * dx + (cy[:, :, None] + cy[:, None, :]) ** 2)
        d_img_t = np.sqrt(dx * dx + (2.0 * die_h[:, :, None] - cy[:, :, None] - cy[:, None, :]) ** 2)

        nbr_img_exp3 = np.sum(
            p_src * (np.exp(-d_img_l / 3.0) + np.exp(-d_img_r / 3.0) + np.exp(-d_img_b / 3.0) + np.exp(-d_img_t / 3.0)),
            axis=2,
        )
        nbr_img_exp6 = np.sum(
            p_src * (np.exp(-d_img_l / 6.0) + np.exp(-d_img_r / 6.0) + np.exp(-d_img_b / 6.0) + np.exp(-d_img_t / 6.0)),
            axis=2,
        )
        nbr_img_inv = np.sum(
            p_src * (1.0 / (d_img_l + 1.0) + 1.0 / (d_img_r + 1.0) + 1.0 / (d_img_b + 1.0) + 1.0 / (d_img_t + 1.0)),
            axis=2,
        )
        add("boundary.nbr_img_exp3", nbr_img_exp3, grp, "W")
        add("boundary.nbr_img_exp6", nbr_img_exp6, grp, "W")
        add("boundary.nbr_img_inv", nbr_img_inv, grp, "W/mm")

        p_rel_r = np.sum(p * rel_r, axis=1, keepdims=True) / np.maximum(p_tot, 1e-6)
        add("boundary.p_tot_x_rel_r", p_tot * rel_r, grp, "W")
        add("boundary.p_tot_x_chip_p_rel_r", p_tot * p_rel_r, grp, "W")

    X = np.column_stack(feats)
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite engineered Stage B v2 features")
    return X, defs
