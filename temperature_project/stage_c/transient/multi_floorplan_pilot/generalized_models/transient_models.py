#!/usr/bin/env python3
"""
Generalized Variable-N One-Speed and Two-Speed Transient Predictors (Stage C Step 7).

Supports arbitrary active core count N derived dynamically from FloorplanSpec:
  - D[k] = R @ P[k]                                         (R in R^{N x N}, P[k] in R^N)
  - One-Speed (N transient params; N^2 + N total):
      T[k+1] = T[k] + (1 - exp(-dt / tau)) * (45 + D[k] - T[k]),  T[0] = 45.0 °C
  - Two-Speed (3N transient params; N^2 + 3N total):
      F[k+1] = F[k] + (1 - exp(-dt / tau_fast)) * (beta * D[k] - F[k])
      S[k+1] = S[k] + (1 - exp(-dt / tau_slow)) * ((1 - beta) * D[k] - S[k])
      T[k+1] = 45.0 + F[k+1] + S[k+1],                            F[0] = S[0] = 0.0 °C

Physical constraints preserved:
  - One-Speed: tau_i > 0
  - Two-Speed: 0 < beta_i < 1 and 0 < tau_fast_i < tau_slow_i
"""

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from floorplan_metadata import AMBIENT_TEMP_C, DEFAULT_DT_SECONDS


def tau_to_alpha(tau_seconds: np.ndarray, dt_seconds: float = DEFAULT_DT_SECONDS) -> np.ndarray:
    """Compute alpha = 1 - exp(-dt / tau) using expm1 for numerical precision."""
    tau = np.asarray(tau_seconds, dtype=np.float64)
    if tau.ndim != 1 or len(tau) == 0:
        raise ValueError(f"tau must be a non-empty 1D array, got shape {tau.shape}")
    if np.any(tau <= 0.0) or not np.all(np.isfinite(tau)):
        raise ValueError(f"All tau values must be finite and > 0, got {tau}")
    return -np.expm1(-dt_seconds / tau)


def alpha_to_tau(alpha: np.ndarray, dt_seconds: float = DEFAULT_DT_SECONDS) -> np.ndarray:
    """Invert alpha in (0, 1) to tau = -dt / ln(1 - alpha)."""
    a = np.asarray(alpha, dtype=np.float64)
    if np.any(a <= 0.0) or np.any(a >= 1.0) or not np.all(np.isfinite(a)):
        raise ValueError(f"All alpha values must lie strictly in (0, 1), got {a}")
    return -dt_seconds / np.log1p(-a)


def validate_two_speed_constraints(
    beta: np.ndarray,
    tau_fast_seconds: np.ndarray,
    tau_slow_seconds: np.ndarray,
    n_cores: int,
) -> None:
    b = np.asarray(beta, dtype=np.float64)
    tf = np.asarray(tau_fast_seconds, dtype=np.float64)
    ts = np.asarray(tau_slow_seconds, dtype=np.float64)
    if b.shape != (n_cores,) or tf.shape != (n_cores,) or ts.shape != (n_cores,):
        raise ValueError(
            f"Parameter shape mismatch for N={n_cores}: "
            f"beta={b.shape}, tau_fast={tf.shape}, tau_slow={ts.shape}"
        )
    if not (np.all(np.isfinite(b)) and np.all(np.isfinite(tf)) and np.all(np.isfinite(ts))):
        raise ValueError("All two-speed parameters must be finite.")
    if np.any(b <= 0.0) or np.any(b >= 1.0):
        raise ValueError(f"Constraint violation: 0 < beta_i < 1 required, got {b}")
    if np.any(tf <= 0.0):
        raise ValueError(f"Constraint violation: tau_fast_i > 0 required, got {tf}")
    if np.any(ts <= tf):
        raise ValueError(f"Constraint violation: tau_fast_i < tau_slow_i required")


def physical_to_unconstrained(
    beta: np.ndarray,
    tau_fast_seconds: np.ndarray,
    tau_slow_seconds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = np.asarray(beta, dtype=np.float64)
    tf = np.asarray(tau_fast_seconds, dtype=np.float64)
    ts = np.asarray(tau_slow_seconds, dtype=np.float64)
    validate_two_speed_constraints(b, tf, ts, len(b))
    return np.log(b / (1.0 - b)), np.log(tf), np.log(ts - tf)


def unconstrained_to_physical(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    beta = 1.0 / (1.0 + np.exp(-np.asarray(u, dtype=np.float64)))
    tau_fast = np.exp(np.asarray(v, dtype=np.float64))
    tau_slow = tau_fast + np.exp(np.asarray(w, dtype=np.float64))
    return beta, tau_fast, tau_slow


@dataclass(frozen=True)
class GeneralizedOneSpeedModel:
    """Variable-N discrete-time one-speed transient temperature predictor."""

    core_names: Tuple[str, ...]
    r_matrix: np.ndarray       # (N, N)
    tau_seconds: np.ndarray    # (N,)
    alpha: np.ndarray          # (N,)
    dt_seconds: float = DEFAULT_DT_SECONDS
    ambient_temp_c: float = AMBIENT_TEMP_C

    @property
    def num_cores(self) -> int:
        return len(self.core_names)

    @property
    def transient_param_count(self) -> int:
        return self.num_cores

    @property
    def total_param_count(self) -> int:
        n = self.num_cores
        return n * n + n

    @property
    def raw_float64_bytes(self) -> int:
        return 8 * self.total_param_count

    @classmethod
    def from_tau(
        cls,
        r_matrix: np.ndarray,
        tau_seconds: np.ndarray,
        core_names: Tuple[str, ...],
        dt_seconds: float = DEFAULT_DT_SECONDS,
        ambient_temp_c: float = AMBIENT_TEMP_C,
    ) -> "GeneralizedOneSpeedModel":
        n = len(core_names)
        if n == 0:
            raise ValueError("core_names cannot be empty.")
        r_arr = np.asarray(r_matrix, dtype=np.float64)
        tau_arr = np.asarray(tau_seconds, dtype=np.float64)
        if r_arr.shape != (n, n):
            raise ValueError(f"Invalid R shape {r_arr.shape} for N={n}")
        if tau_arr.shape != (n,):
            raise ValueError(f"Invalid tau shape {tau_arr.shape} for N={n}")
        alpha_arr = tau_to_alpha(tau_arr, dt_seconds=dt_seconds)
        return cls(
            core_names=tuple(core_names),
            r_matrix=r_arr,
            tau_seconds=tau_arr,
            alpha=alpha_arr,
            dt_seconds=float(dt_seconds),
            ambient_temp_c=float(ambient_temp_c),
        )

    def predict_sequence_c(
        self,
        power_seq_w: np.ndarray,
        init_temp_c: float = AMBIENT_TEMP_C,
    ) -> np.ndarray:
        p = np.asarray(power_seq_w, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != self.num_cores:
            raise ValueError(f"Expected power shape (T, {self.num_cores}), got {p.shape}")
        t_ss_seq = self.ambient_temp_c + (p @ self.r_matrix.T)
        num_steps, num_cores = t_ss_seq.shape
        preds = np.empty((num_steps, num_cores), dtype=np.float64)
        t_curr = np.full(num_cores, float(init_temp_c), dtype=np.float64)
        alpha = self.alpha
        one_minus_alpha = 1.0 - alpha
        for k in range(num_steps):
            t_curr = one_minus_alpha * t_curr + alpha * t_ss_seq[k]
            preds[k] = t_curr
        return preds


@dataclass(frozen=True)
class GeneralizedTwoSpeedModel:
    """Variable-N discrete-time two-speed (dual-RC) transient temperature predictor."""

    core_names: Tuple[str, ...]
    r_matrix: np.ndarray           # (N, N)
    beta: np.ndarray               # (N,)
    tau_fast_seconds: np.ndarray   # (N,)
    tau_slow_seconds: np.ndarray   # (N,)
    alpha_fast: np.ndarray         # (N,)
    alpha_slow: np.ndarray         # (N,)
    dt_seconds: float = DEFAULT_DT_SECONDS
    ambient_temp_c: float = AMBIENT_TEMP_C

    @property
    def num_cores(self) -> int:
        return len(self.core_names)

    @property
    def transient_param_count(self) -> int:
        return 3 * self.num_cores

    @property
    def total_param_count(self) -> int:
        n = self.num_cores
        return n * n + 3 * n

    @property
    def raw_float64_bytes(self) -> int:
        return 8 * self.total_param_count

    @classmethod
    def from_physical_params(
        cls,
        r_matrix: np.ndarray,
        beta: np.ndarray,
        tau_fast_seconds: np.ndarray,
        tau_slow_seconds: np.ndarray,
        core_names: Tuple[str, ...],
        dt_seconds: float = DEFAULT_DT_SECONDS,
        ambient_temp_c: float = AMBIENT_TEMP_C,
    ) -> "GeneralizedTwoSpeedModel":
        n = len(core_names)
        if n == 0:
            raise ValueError("core_names cannot be empty.")
        r_arr = np.asarray(r_matrix, dtype=np.float64)
        if r_arr.shape != (n, n):
            raise ValueError(f"Invalid R shape {r_arr.shape} for N={n}")
        b_arr = np.asarray(beta, dtype=np.float64)
        tf_arr = np.asarray(tau_fast_seconds, dtype=np.float64)
        ts_arr = np.asarray(tau_slow_seconds, dtype=np.float64)
        validate_two_speed_constraints(b_arr, tf_arr, ts_arr, n)
        return cls(
            core_names=tuple(core_names),
            r_matrix=r_arr,
            beta=b_arr,
            tau_fast_seconds=tf_arr,
            tau_slow_seconds=ts_arr,
            alpha_fast=tau_to_alpha(tf_arr, dt_seconds=dt_seconds),
            alpha_slow=tau_to_alpha(ts_arr, dt_seconds=dt_seconds),
            dt_seconds=float(dt_seconds),
            ambient_temp_c=float(ambient_temp_c),
        )

    def predict_sequence_c(
        self,
        power_seq_w: np.ndarray,
        init_temp_c: float = AMBIENT_TEMP_C,
    ) -> np.ndarray:
        if not math.isclose(init_temp_c, self.ambient_temp_c, abs_tol=1e-12):
            raise ValueError("Two-speed model requires T[0] = ambient_temp_c (F[0]=S[0]=0).")
        p = np.asarray(power_seq_w, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != self.num_cores:
            raise ValueError(f"Expected power shape (T, {self.num_cores}), got {p.shape}")

        d_seq = p @ self.r_matrix.T
        num_steps, num_cores = d_seq.shape
        preds = np.empty((num_steps, num_cores), dtype=np.float64)

        f_curr = np.zeros(num_cores, dtype=np.float64)
        s_curr = np.zeros(num_cores, dtype=np.float64)
        af_beta = self.alpha_fast * self.beta
        as_om_beta = self.alpha_slow * (1.0 - self.beta)
        om_af = 1.0 - self.alpha_fast
        om_as = 1.0 - self.alpha_slow
        amb = self.ambient_temp_c

        for k in range(num_steps):
            dk = d_seq[k]
            f_curr = om_af * f_curr + af_beta * dk
            s_curr = om_as * s_curr + as_om_beta * dk
            preds[k] = amb + f_curr + s_curr

        return preds


def compute_error_metrics(
    pred_c: np.ndarray,
    target_c: np.ndarray,
    core_names: Tuple[str, ...],
) -> Dict[str, object]:
    if pred_c.shape != target_c.shape:
        raise ValueError(f"Shape mismatch: {pred_c.shape} vs {target_c.shape}")
    diff = pred_c - target_c
    abs_diff = np.abs(diff)
    sq_diff = diff * diff

    mse = float(np.mean(sq_diff))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(abs_diff))
    max_err = float(np.max(abs_diff))

    per_core_rmse = [float(math.sqrt(float(np.mean(sq_diff[:, i])))) for i in range(len(core_names))]
    per_core_mae = [float(np.mean(abs_diff[:, i])) for i in range(len(core_names))]
    per_core_max = [float(np.max(abs_diff[:, i])) for i in range(len(core_names))]

    return {
        "mse_c2": mse,
        "rmse_c": rmse,
        "mae_c": mae,
        "max_error_c": max_err,
        "per_core_rmse_c": per_core_rmse,
        "per_core_mae_c": per_core_mae,
        "per_core_max_error_c": per_core_max,
    }
