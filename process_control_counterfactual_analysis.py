"""
r2r_counterfactual_diagnostic.py
================================

Counterfactual ("what if there had been no control") reconstruction and
diagnostics for a high-mix run-to-run (R2R) control process with
feedforward (FF) + feedback (FB) control, MIMO process gain, and
delayed / sparse metrology.

WHY THE RECONSTRUCTION IS SIMPLE
--------------------------------
The controller's only actuator is the control knob ``u``. FF and FB both act
*through* the knob. So the realized effect of ALL control action on a measured
lot is captured entirely by how far the knob was moved from its no-control
baseline ``u0``:

    control_effect = M @ (u_used - u0)          # M = process gain (dynamics)
    y_nocontrol    = y_observed - control_effect
    d_realized     = y_observed - M @ u_used    # disturbance backed out of y

Consequences (these resolve the usual worries about "we don't know m and g"):
  * You do NOT need the FF gain g, the FF function f(), or the tool state to
    reconstruct the counterfactual. They were only used to *decide* u, and you
    observe u_used directly. Here they feed only secondary diagnostics
    (model-consistency and FF-effectiveness).
  * The process sensitivity-to-disturbance never appears: each measured lot's
    disturbance AND its noise are already realized inside y_observed, so the
    reconstruction keeps that lot's true incoming condition for free.
  * Reconstruction is PER-LOT (not cumulative) as long as the disturbance is
    exogenous to the knob (the knob does not physically alter tool drift). The
    controller's tool_state integrates, but that is an *estimate*, not the
    physical process, and we bypass it by observing u_used.

The one thing that must be accurate is the process gain matrix M (it is what we
subtract). Its uncertainty is propagated by Monte Carlo (``m_rel_sigma``).

Each function is independent and reusable; ``run_diagnostic`` orchestrates them.
Run this file directly to execute a self-test on synthetic data.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------------
# Configuration: map your column names + supply the extra info the diagnostic
# needs (baseline knob, targets/specs, gain uncertainty). See module README at
# the bottom of run_diagnostic for what is *required* vs *optional*.
# ----------------------------------------------------------------------------
@dataclass
class DiagnosticConfig:
    # --- grouping / keys -----------------------------------------------------
    group_cols: Sequence[str] = ("process_id", "tool_id")  # control context
    material_col: str = "material_id"

    # --- array-bearing columns (cells hold lists / np.ndarrays) --------------
    M_col: str = "process_dynamics_m"        # (n_out, n_knob) process gain
    knob_used_col: str = "used_knob"          # (n_knob,) actually-used setting
    knob_initial_col: str = "initial_knob"    # (n_knob,) recipe nominal
    knob_recommended_col: str = "recommended_knob"
    y_col: str = "post_meas"                  # (n_out,) ; None/NaN when unmeasured
    ff_col: str = "ff_disturbance"            # FF disturbance estimate
    g_col: str = "ff_gain_g"                  # scalar / (n_out,) / (n_out,n_ff)
    tool_state_col: str = "last_tool_state"   # (n_out,) FB state estimate
    y_pred_col: str = "y_pred"                # (n_out,) predicted post-meas

    # --- ordering for the disturbance sequence -------------------------------
    order_time_col: str = "process_complete_dt"

    # --- no-control baseline knob u0 -----------------------------------------
    baseline: str = "initial"  # "initial" | "recommended" | "zero" | "custom"
    baseline_custom: Optional[np.ndarray] = None

    # --- FF convention -------------------------------------------------------
    ff_already_scaled: bool = False  # True => ff_col already in output units

    # --- performance metric inputs (optional) --------------------------------
    target: Optional[np.ndarray] = None  # (n_out,)
    usl: Optional[np.ndarray] = None     # (n_out,) upper spec limit
    lsl: Optional[np.ndarray] = None     # (n_out,) lower spec limit

    # --- uncertainty propagation (optional) ----------------------------------
    m_rel_sigma: float = 0.0   # systematic 1-sigma relative error on each M elem
    n_mc: int = 500
    random_state: int = 0

    # --- diagnostics knobs ---------------------------------------------------
    min_n_for_dynamics: int = 8  # below this, skip stationarity verdict


# ----------------------------------------------------------------------------
# Small coercion helpers (dataframe cells -> numpy)
# ----------------------------------------------------------------------------
def _as_vec(cell) -> Optional[np.ndarray]:
    if cell is None:
        return None
    if np.isscalar(cell):
        return None if pd.isna(cell) else np.array([float(cell)])
    arr = np.asarray(cell, dtype=float).ravel()
    return arr


def _as_mat(cell) -> np.ndarray:
    arr = np.asarray(cell, dtype=float)
    if arr.ndim == 1:        # treat a flat vector as a row -> (1, k) diag-ish
        arr = arr.reshape(1, -1)
    return arr


def _is_measured(cell) -> bool:
    if cell is None:
        return False
    if np.isscalar(cell):
        return not pd.isna(cell)
    return np.asarray(cell).size > 0


def _stack_vec(series: pd.Series) -> np.ndarray:
    """Stack a column of equal-length vectors into (n, d)."""
    rows = [_as_vec(c) for c in series]
    return np.vstack(rows)


def _stack_mat(series: pd.Series) -> np.ndarray:
    """Stack a column of (n_out, n_knob) matrices into (n, n_out, n_knob)."""
    return np.stack([_as_mat(c) for c in series], axis=0)


def _baseline_knob(df: pd.DataFrame, cfg: DiagnosticConfig, n_knob: int) -> np.ndarray:
    n = len(df)
    if cfg.baseline == "initial":
        return _stack_vec(df[cfg.knob_initial_col])
    if cfg.baseline == "recommended":
        return _stack_vec(df[cfg.knob_recommended_col])
    if cfg.baseline == "zero":
        return np.zeros((n, n_knob))
    if cfg.baseline == "custom":
        if cfg.baseline_custom is None:
            raise ValueError("baseline='custom' requires cfg.baseline_custom")
        return np.tile(np.asarray(cfg.baseline_custom, float), (n, 1))
    raise ValueError(f"unknown baseline {cfg.baseline!r}")


# ----------------------------------------------------------------------------
# STEP 5 — counterfactual reconstruction (the core operation)
# ----------------------------------------------------------------------------
def reconstruct(df: pd.DataFrame, cfg: DiagnosticConfig) -> pd.DataFrame:
    """Reconstruct the no-control output for every *measured* lot.

    Returns a tidy DataFrame (one row per measured lot) with, per output dim j:
        y_obs_j, y_nc_j, d_real_j, ctrl_eff_j
    plus group keys, material id, and the ordering timestamp.
    """
    m = df[df[cfg.y_col].apply(_is_measured)].copy()
    if m.empty:
        raise ValueError("no measured lots (post-meas) found")

    M = _stack_mat(m[cfg.M_col])            # (n, n_out, n_knob)
    u_used = _stack_vec(m[cfg.knob_used_col])    # (n, n_knob)
    y_obs = _stack_vec(m[cfg.y_col])             # (n, n_out)
    n_knob = u_used.shape[1]
    u0 = _baseline_knob(m, cfg, n_knob)          # (n, n_knob)

    # control_effect[i] = M[i] @ (u_used[i] - u0[i]) ; y_nc = y_obs - effect
    delta_u = u_used - u0
    ctrl_eff = np.einsum("ijk,ik->ij", M, delta_u)      # (n, n_out)
    y_nc = y_obs - ctrl_eff
    d_real = y_obs - np.einsum("ijk,ik->ij", M, u_used)  # disturbance back-out

    n_out = y_obs.shape[1]
    out = m[list(cfg.group_cols) + [cfg.material_col, cfg.order_time_col]].reset_index(drop=True)
    for j in range(n_out):
        out[f"y_obs_{j}"] = y_obs[:, j]
        out[f"y_nc_{j}"] = y_nc[:, j]
        out[f"d_real_{j}"] = d_real[:, j]
        out[f"ctrl_eff_{j}"] = ctrl_eff[:, j]
    out.attrs["n_out"] = n_out
    return out.sort_values(list(cfg.group_cols) + [cfg.order_time_col]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# STEP 5b — inspectable trace of the reconstruction arithmetic
# ----------------------------------------------------------------------------
# reconstruct() does the math inside vectorized einsum calls, so you can see the
# resulting y_nc but not *how each number was produced*. The two helpers below
# expand the same arithmetic to the individual-term level:
#
#     ctrl_eff[i,j] = sum_k  M[i,j,k] * (u_used[i,k] - u0[i,k])
#     y_nc[i,j]     = y_obs[i,j] - ctrl_eff[i,j]
#     d_real[i,j]   = y_obs[i,j] - sum_k M[i,j,k] * u_used[i,k]
#
#   * reconstruct_trace() -> tidy long DataFrame, one row per (lot, out j, knob k),
#     carrying every operand and the per-term product, plus the assembled totals.
#     Group/sum it however you like to audit any number; it reconstructs y_nc by
#     construction (see the self-test).
#   * explain_lot() -> a plain-text, fully-substituted derivation for ONE lot,
#     e.g. "ctrl_eff[0] = 1.000*2.345 + 0.200*(-0.013) = 2.342".
# ----------------------------------------------------------------------------
def reconstruct_trace(df: pd.DataFrame, cfg: DiagnosticConfig) -> pd.DataFrame:
    """Term-level expansion of the counterfactual reconstruction.

    Granularity is one row per (measured lot, output dim j, knob k). Per-term
    columns (``M``, ``u_used``, ``u0``, ``delta_u``, ``term``) let you see each
    multiply; the per-(lot, j) totals (``ctrl_eff``, ``y_obs``, ``y_nc``,
    ``d_real``) are repeated across that lot's knob rows so a simple
    groupby-sum of ``term`` equals ``ctrl_eff`` and ``y_obs - ctrl_eff`` equals
    ``y_nc``. Sorted to match :func:`reconstruct`.
    """
    m = df[df[cfg.y_col].apply(_is_measured)].copy()
    if m.empty:
        raise ValueError("no measured lots (post-meas) found")

    M = _stack_mat(m[cfg.M_col])              # (n, n_out, n_knob)
    u_used = _stack_vec(m[cfg.knob_used_col])      # (n, n_knob)
    y_obs = _stack_vec(m[cfg.y_col])               # (n, n_out)
    n, n_out, n_knob = M.shape
    u0 = _baseline_knob(m, cfg, n_knob)            # (n, n_knob)
    delta_u = u_used - u0

    ctrl_eff = np.einsum("ijk,ik->ij", M, delta_u)        # (n, n_out)
    y_nc = y_obs - ctrl_eff
    d_real = y_obs - np.einsum("ijk,ik->ij", M, u_used)

    keys = m[list(cfg.group_cols) + [cfg.material_col, cfg.order_time_col]].reset_index(drop=True)
    rows = []
    for i in range(n):
        key = {c: keys.iloc[i][c] for c in cfg.group_cols}
        key[cfg.material_col] = keys.iloc[i][cfg.material_col]
        key[cfg.order_time_col] = keys.iloc[i][cfg.order_time_col]
        for j in range(n_out):
            for k in range(n_knob):
                rows.append({
                    **key,
                    "output_dim": j,
                    "knob": k,
                    "M": float(M[i, j, k]),
                    "u_used": float(u_used[i, k]),
                    "u0": float(u0[i, k]),
                    "delta_u": float(delta_u[i, k]),
                    "term": float(M[i, j, k] * delta_u[i, k]),  # contribution to ctrl_eff
                    # per-(lot, j) assembled totals (repeated across knob rows):
                    "ctrl_eff": float(ctrl_eff[i, j]),
                    "y_obs": float(y_obs[i, j]),
                    "y_nc": float(y_nc[i, j]),
                    "d_real": float(d_real[i, j]),
                })
    trace = pd.DataFrame(rows)
    trace.attrs["n_out"] = n_out
    trace.attrs["n_knob"] = n_knob
    trace.attrs["baseline"] = cfg.baseline
    return trace.sort_values(
        list(cfg.group_cols) + [cfg.order_time_col, "output_dim", "knob"]
    ).reset_index(drop=True)


def explain_lot(df: pd.DataFrame, cfg: DiagnosticConfig,
                material_id=None, index: Optional[int] = None,
                precision: int = 3) -> str:
    """Plain-text, fully-substituted derivation of y_nc for a single lot.

    Select the lot by ``material_id`` (matched on ``cfg.material_col``) or by
    positional ``index`` into the measured-and-sorted lots (same order as
    :func:`reconstruct`). Returns a human-readable string showing the baseline
    choice, each knob move, and the term-by-term sum that yields ctrl_eff, y_nc
    and d_real for every output dimension.
    """
    trace = reconstruct_trace(df, cfg)
    lots = trace[[cfg.material_col] + list(cfg.group_cols) + [cfg.order_time_col]].drop_duplicates()
    lots = lots.reset_index(drop=True)

    if material_id is not None:
        sel = lots[lots[cfg.material_col] == material_id]
        if sel.empty:
            raise ValueError(f"material_id {material_id!r} not found among measured lots")
        mat = sel.iloc[0][cfg.material_col]
    elif index is not None:
        if not (0 <= index < len(lots)):
            raise IndexError(f"index {index} out of range (0..{len(lots) - 1})")
        mat = lots.iloc[index][cfg.material_col]
    else:
        mat = lots.iloc[0][cfg.material_col]

    sub = trace[trace[cfg.material_col] == mat]
    head = sub.iloc[0]
    p = precision

    def f(x):  # signed, fixed-precision, parenthesize negatives for readability
        s = f"{x:.{p}f}"
        return f"({s})" if x < 0 else s

    keystr = ", ".join(f"{c}={head[c]}" for c in cfg.group_cols)
    lines = [
        f"Lot {mat}  ({keystr})  @ {head[cfg.order_time_col]}",
        f"baseline u0 = {cfg.baseline!r}",
    ]

    # knob moves (same for every output dim -> read off the j == first-dim rows)
    j0 = sub["output_dim"].min()
    knob_rows = sub[sub["output_dim"] == j0].sort_values("knob")
    lines.append("knob moves  Δu = u_used - u0:")
    for _, r in knob_rows.iterrows():
        lines.append(
            f"    knob[{int(r['knob'])}]: {r['u_used']:.{p}f} - {r['u0']:.{p}f} = {f(r['delta_u'])}"
        )

    for j in sorted(sub["output_dim"].unique()):
        jr = sub[sub["output_dim"] == j].sort_values("knob")
        first = jr.iloc[0]
        sym_terms = " + ".join(f"M[{j},{int(r['knob'])}]·Δu[{int(r['knob'])}]"
                               for _, r in jr.iterrows())
        num_terms = " + ".join(f"{r['M']:.{p}f}·{f(r['delta_u'])}"
                               for _, r in jr.iterrows())
        partials = " + ".join(f(r["term"]) for _, r in jr.iterrows())
        lines += [
            f"output dim {j}:",
            f"  ctrl_eff[{j}] = Σ M[{j},k]·Δu[k] = {sym_terms}",
            f"               = {num_terms}",
            f"               = {partials}",
            f"               = {first['ctrl_eff']:.{p}f}",
            f"  y_nc[{j}]   = y_obs[{j}] - ctrl_eff[{j}] = "
            f"{first['y_obs']:.{p}f} - {f(first['ctrl_eff'])} = {first['y_nc']:.{p}f}",
            f"  d_real[{j}] = y_obs[{j}] - Σ M[{j},k]·u_used[k] = {first['d_real']:.{p}f}",
        ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# STEP 4 — disturbance dynamics: stationary vs drifting (unit root)
# ----------------------------------------------------------------------------
def _adf_tstat(x: np.ndarray, max_lag: int = 1):
    """Augmented Dickey-Fuller regression t-stat on the lagged-level coeff.

    Delta x_t = a + rho*x_{t-1} + sum gamma_i Delta x_{t-i} + e.
    Returns (t_stat, rho_hat). More negative t => reject unit root (stationary).
    """
    x = np.asarray(x, float)
    dx = np.diff(x)
    n = len(dx)
    p = int(min(max_lag, max(0, n // 2 - 2)))
    # rows valid for t in [p, n-1) of dx ; build design matrix
    rows = range(p, n)
    Y, X = [], []
    for t in rows:
        cols = [1.0, x[t]]                       # const, level lag
        cols += [dx[t - i] for i in range(1, p + 1)]  # augmenting diffs
        X.append(cols)
        Y.append(dx[t])
    X = np.asarray(X); Y = np.asarray(Y)
    if X.shape[0] <= X.shape[1]:
        return np.nan, np.nan
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    dof = X.shape[0] - X.shape[1]
    s2 = (resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se_rho = np.sqrt(cov[1, 1])
    return beta[1] / se_rho, beta[1]


def _kpss_stat(x: np.ndarray):
    """KPSS level-stationarity LM statistic (Newey-West long-run variance).

    Large statistic => reject stationarity (evidence of unit root / drift).
    5% critical value for level stationarity ~ 0.463.
    """
    x = np.asarray(x, float)
    n = len(x)
    e = x - x.mean()
    S = np.cumsum(e)
    lag = int(np.floor(4 * (n / 100.0) ** 0.25))
    s2 = (e @ e) / n
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        s2 += 2.0 * w * (e[l:] @ e[:-l]) / n
    return (S @ S) / (n ** 2 * s2)


def _variance_ratio(x: np.ndarray, q: int = 2) -> float:
    """VR(q) ~ 1 random walk, <1 mean-reverting, >1 trending/drift."""
    x = np.asarray(x, float)
    d1 = np.diff(x)
    dq = x[q:] - x[:-q]
    v1 = np.var(d1, ddof=1)
    vq = np.var(dq, ddof=1)
    return (vq / (q * v1)) if v1 > 0 else np.nan


def stationarity_test(x: np.ndarray) -> dict:
    """Combine ADF + KPSS + variance-ratio into a single verdict."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    res = {"n": len(x)}
    adf_t, rho = _adf_tstat(x)
    kpss = _kpss_stat(x)
    vr2, vr4 = _variance_ratio(x, 2), _variance_ratio(x, 4)
    adf_crit, kpss_crit = -2.89, 0.463  # 5% (constant)
    adf_reject_ur = (adf_t < adf_crit) if np.isfinite(adf_t) else None
    kpss_reject_stat = (kpss > kpss_crit) if np.isfinite(kpss) else None
    # verdict
    if adf_reject_ur and not kpss_reject_stat:
        verdict = "stationary"
    elif (adf_reject_ur is False) and kpss_reject_stat:
        verdict = "drifting (unit root)"
    else:  # mixed/weak -> lean on variance ratio
        verdict = "drifting (unit root)" if (np.isfinite(vr2) and vr2 > 1.3) else \
                  "stationary" if (np.isfinite(vr2) and vr2 < 0.7) else "inconclusive"
    res.update(adf_t=adf_t, adf_rho=rho, adf_reject_unitroot=adf_reject_ur,
               kpss=kpss, kpss_reject_stationary=kpss_reject_stat,
               vr2=vr2, vr4=vr4, lag1_autocorr=_lag1(x), verdict=verdict)
    return res


def _lag1(x):
    if len(x) < 3:
        return np.nan
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def diagnose_dynamics(recon: pd.DataFrame, cfg: DiagnosticConfig) -> pd.DataFrame:
    """Per (group, output dim) stationarity verdict on the realized disturbance.

    NOTE: the series is ordered by ``order_time_col`` but treated as evenly
    spaced; metrology is irregular, so read the verdict as a guide, not gospel.
    """
    n_out = recon.attrs.get("n_out", 1)
    rows = []
    for keys, g in recon.groupby(list(cfg.group_cols)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for j in range(n_out):
            s = g[f"d_real_{j}"].to_numpy()
            if len(s) < cfg.min_n_for_dynamics:
                rec = {"n": len(s), "verdict": "insufficient_data"}
            else:
                rec = stationarity_test(s)
            row = dict(zip(cfg.group_cols, keys))
            row.update(output_dim=j, recommended_subtraction=(
                "cumulative" if str(rec.get("verdict", "")).startswith("drift")
                else "per-lot" if rec.get("verdict") == "stationary" else "review"))
            row.update(rec)
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# STEP 3 (gate) — validate the supplied M / g / tool_state against the data
# ----------------------------------------------------------------------------
def _ff_contribution(df: pd.DataFrame, cfg: DiagnosticConfig, n_out: int) -> np.ndarray:
    ff = _stack_vec(df[cfg.ff_col])  # (n, n_ff)
    if cfg.ff_already_scaled:
        contrib = ff
    else:
        g_first = _as_vec(df[cfg.g_col].iloc[0])
        if g_first is not None and g_first.size == 1:           # scalar gain
            contrib = float(g_first[0]) * ff
        else:                                                    # per-dim / matrix
            g_stack = _stack_vec(df[cfg.g_col])                  # (n, n_out) assumed
            contrib = g_stack * ff if g_stack.shape == ff.shape else ff
    # pad/truncate to n_out if FF dim differs from output dim
    if contrib.shape[1] != n_out:
        fixed = np.zeros((contrib.shape[0], n_out))
        k = min(contrib.shape[1], n_out)
        fixed[:, :k] = contrib[:, :k]
        contrib = fixed
    return contrib


def model_consistency(df: pd.DataFrame, cfg: DiagnosticConfig) -> pd.DataFrame:
    """Check y_obs ~ M@u_used + g*ff + tool_state on measured lots.

    Structured / large residuals flag that the *supplied* M (and g, tool_state)
    do not match reality -- i.e. the gains you'd subtract are inaccurate. This
    is the practical stand-in for the identifiability gate when M, g are given
    rather than fit.
    """
    m = df[df[cfg.y_col].apply(_is_measured)].copy()
    M = _stack_mat(m[cfg.M_col])
    u_used = _stack_vec(m[cfg.knob_used_col])
    y_obs = _stack_vec(m[cfg.y_col])
    n_out = y_obs.shape[1]
    ts = _stack_vec(m[cfg.tool_state_col]) if cfg.tool_state_col in m else np.zeros_like(y_obs)
    ff_c = _ff_contribution(m, cfg, n_out)
    pred = np.einsum("ijk,ik->ij", M, u_used) + ff_c + ts
    resid = y_obs - pred

    rows = []
    m2 = m.reset_index(drop=True)
    for keys, idx in m2.groupby(list(cfg.group_cols)).groups.items():
        keys = keys if isinstance(keys, tuple) else (keys,)
        rr = resid[list(idx)]
        for j in range(n_out):
            col = rr[:, j]
            row = dict(zip(cfg.group_cols, keys))
            row.update(output_dim=j, n=len(col),
                       resid_rms=float(np.sqrt(np.mean(col ** 2))),
                       resid_mean=float(np.mean(col)),
                       resid_lag1_autocorr=_lag1(col),
                       frac_resid_var_of_y=float(np.var(col) / (np.var(y_obs[list(idx), j]) + 1e-12)))
            rows.append(row)
    return pd.DataFrame(rows)


def ff_effectiveness(df: pd.DataFrame, cfg: DiagnosticConfig) -> pd.DataFrame:
    """How much of the realized disturbance the FF term explains (R^2)."""
    m = df[df[cfg.y_col].apply(_is_measured)].copy()
    M = _stack_mat(m[cfg.M_col])
    u_used = _stack_vec(m[cfg.knob_used_col])
    y_obs = _stack_vec(m[cfg.y_col])
    n_out = y_obs.shape[1]
    d_real = y_obs - np.einsum("ijk,ik->ij", M, u_used)
    ff_c = _ff_contribution(m, cfg, n_out)
    m2 = m.reset_index(drop=True)
    rows = []
    for keys, idx in m2.groupby(list(cfg.group_cols)).groups.items():
        keys = keys if isinstance(keys, tuple) else (keys,)
        idx = list(idx)
        for j in range(n_out):
            d = d_real[idx, j]; f = ff_c[idx, j]
            ss_tot = np.var(d) * len(d)
            ss_res = np.sum((d - f) ** 2)
            r2 = 1.0 - ss_res / (ss_tot + 1e-12)
            row = dict(zip(cfg.group_cols, keys))
            row.update(output_dim=j, n=len(d), ff_r2=float(r2),
                       resid_after_ff_std=float(np.std(d - f)),
                       disturbance_std=float(np.std(d)))
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Performance metrics: controlled (y_obs) vs reconstructed no-control (y_nc)
# ----------------------------------------------------------------------------
def _cpk(y, tgt, usl, lsl):
    mu, sd = np.mean(y), np.std(y, ddof=1)
    if sd == 0 or usl is None or lsl is None:
        return np.nan, 0.0
    cpk = min(usl - mu, mu - lsl) / (3 * sd)
    oos = float(np.mean((y > usl) | (y < lsl)))
    return float(cpk), oos


def performance(recon: pd.DataFrame, cfg: DiagnosticConfig) -> pd.DataFrame:
    n_out = recon.attrs.get("n_out", 1)
    tgt = cfg.target; usl = cfg.usl; lsl = cfg.lsl
    rows = []
    for keys, g in recon.groupby(list(cfg.group_cols)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for j in range(n_out):
            yo = g[f"y_obs_{j}"].to_numpy()
            yn = g[f"y_nc_{j}"].to_numpy()
            t = None if tgt is None else np.atleast_1d(tgt)[j]
            u = None if usl is None else np.atleast_1d(usl)[j]
            l = None if lsl is None else np.atleast_1d(lsl)[j]
            cpk_c, oos_c = _cpk(yo, t, u, l)
            cpk_n, oos_n = _cpk(yn, t, u, l)
            row = dict(zip(cfg.group_cols, keys))
            row.update(output_dim=j, n=len(yo),
                       std_controlled=float(np.std(yo, ddof=1)),
                       std_nocontrol=float(np.std(yn, ddof=1)),
                       var_reduction_pct=float(100 * (1 - np.var(yo) / (np.var(yn) + 1e-12))),
                       cpk_controlled=cpk_c, cpk_nocontrol=cpk_n,
                       oos_controlled=oos_c, oos_nocontrol=oos_n)
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# STEP 6 — propagate process-gain (M) uncertainty by Monte Carlo
# ----------------------------------------------------------------------------
def mc_uncertainty(df: pd.DataFrame, cfg: DiagnosticConfig) -> pd.DataFrame:
    """Resample M by a shared (systematic) relative error and re-derive the
    headline metric (variance reduction %, and Cpk_nocontrol if specs given).
    Returns per (group, dim) the 5/50/95 percentiles across draws.
    """
    if cfg.m_rel_sigma <= 0 or cfg.n_mc <= 0:
        return pd.DataFrame()
    m = df[df[cfg.y_col].apply(_is_measured)].copy().reset_index(drop=True)
    M = _stack_mat(m[cfg.M_col])
    u_used = _stack_vec(m[cfg.knob_used_col])
    y_obs = _stack_vec(m[cfg.y_col])
    n_knob = u_used.shape[1]; n_out = y_obs.shape[1]
    u0 = _baseline_knob(m, cfg, n_knob)
    delta_u = u_used - u0
    rng = np.random.default_rng(cfg.random_state)
    groups = {k: list(v) for k, v in m.groupby(list(cfg.group_cols)).groups.items()}

    acc = {k: {j: {"vr": [], "cpk": []} for j in range(n_out)} for k in groups}
    for _ in range(cfg.n_mc):
        # one systematic multiplicative perturbation per M element, all lots
        factor = 1.0 + rng.normal(0.0, cfg.m_rel_sigma, size=M.shape[1:])
        Mk = M * factor
        yn = y_obs - np.einsum("ijk,ik->ij", Mk, delta_u)
        for k, idx in groups.items():
            for j in range(n_out):
                yo = y_obs[idx, j]; ynj = yn[idx, j]
                acc[k][j]["vr"].append(100 * (1 - np.var(yo) / (np.var(ynj) + 1e-12)))
                if cfg.usl is not None and cfg.lsl is not None:
                    c, _ = _cpk(ynj, None, np.atleast_1d(cfg.usl)[j], np.atleast_1d(cfg.lsl)[j])
                    acc[k][j]["cpk"].append(c)
    rows = []
    for k, dims in acc.items():
        keys = k if isinstance(k, tuple) else (k,)
        for j, d in dims.items():
            vr = np.array(d["vr"]); row = dict(zip(cfg.group_cols, keys))
            row.update(output_dim=j,
                       var_reduction_p05=float(np.percentile(vr, 5)),
                       var_reduction_p50=float(np.percentile(vr, 50)),
                       var_reduction_p95=float(np.percentile(vr, 95)))
            if d["cpk"]:
                cpk = np.array(d["cpk"])
                row.update(cpk_nc_p05=float(np.percentile(cpk, 5)),
                           cpk_nc_p50=float(np.percentile(cpk, 50)),
                           cpk_nc_p95=float(np.percentile(cpk, 95)))
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------
def run_diagnostic(df: pd.DataFrame, cfg: Optional[DiagnosticConfig] = None) -> dict:
    """Run the full diagnostic. Returns a dict of DataFrames."""
    cfg = cfg or DiagnosticConfig()
    recon = reconstruct(df, cfg)
    result = {
        "reconstruction": recon,
        "reconstruction_trace": reconstruct_trace(df, cfg),
        "dynamics": diagnose_dynamics(recon, cfg),
        "model_consistency": model_consistency(df, cfg),
        "ff_effectiveness": ff_effectiveness(df, cfg),
        "performance": performance(recon, cfg),
        "uncertainty": mc_uncertainty(df, cfg),
    }
    return result


def summarize(result: dict) -> str:
    lines = ["=== R2R counterfactual diagnostic ==="]
    r = result["reconstruction"]
    lines.append(f"measured lots reconstructed: {len(r)}")
    lines.append("\n-- dynamics (per group/dim) --")
    lines.append(result["dynamics"][[c for c in result["dynamics"].columns
                  if c in ("process_id", "tool_id", "output_dim", "verdict",
                           "recommended_subtraction", "vr2", "lag1_autocorr")]
                  ].to_string(index=False))
    lines.append("\n-- model consistency (supplied M/g check) --")
    lines.append(result["model_consistency"][["process_id", "tool_id", "output_dim",
                  "resid_rms", "frac_resid_var_of_y", "resid_lag1_autocorr"]].to_string(index=False))
    lines.append("\n-- performance: controlled vs no-control --")
    lines.append(result["performance"][["process_id", "tool_id", "output_dim",
                  "std_controlled", "std_nocontrol", "var_reduction_pct",
                  "cpk_controlled", "cpk_nocontrol"]].to_string(index=False))
    if not result["uncertainty"].empty:
        lines.append("\n-- variance-reduction % with M uncertainty (p05/p50/p95) --")
        lines.append(result["uncertainty"].to_string(index=False))
    return "\n".join(lines)


# ============================================================================
# SELF-TEST on synthetic data (run: python r2r_counterfactual_diagnostic.py)
# ============================================================================
def _make_synthetic(n_lots=400, meas_fraction=0.15, drift=True, seed=1):
    rng = np.random.default_rng(seed)
    M_true = np.array([[1.0, 0.2], [0.1, 0.8]])      # 2 outputs x 2 knobs
    u_init = np.array([10.0, 5.0])                    # recipe nominal (no-control)
    target = np.array([100.0, 50.0])
    lam = 0.3                                         # EWMA gain
    g = 0.6
    tool_state = np.zeros(2)
    d = np.zeros(2)
    rows, truth_ync = [], []
    t0 = np.datetime64("2026-01-01T00:00:00")
    for i in range(n_lots):
        # exogenous disturbance: drift (random walk) + FF-explainable from incoming
        if drift:
            d = d + rng.normal(0, 0.6, 2)             # random walk component
        incoming = rng.normal(0, 1.0, 3)              # incoming matrix (flattened)
        ff_dist = np.array([0.5 * incoming[0], 0.4 * incoming[1]])  # f(incoming)
        ff_contrib = g * ff_dist
        d_total = d + ff_contrib                       # realized disturbance
        # controller solves M u = target - ff_contrib - tool_state
        u_rec = np.linalg.solve(M_true, target - ff_contrib - tool_state)
        u_used = u_rec + rng.normal(0, 0.05, 2)        # knob precision / manual
        noise = rng.normal(0, 0.4, 2)                  # metrology + process noise
        y = M_true @ u_used + d_total + noise
        measured = rng.random() < meas_fraction
        if measured:                                   # FB updates state on measure
            d_obs = y - M_true @ u_used                # back-out disturbance
            tool_state = (1 - lam) * tool_state + lam * (d_obs - ff_contrib)
        rows.append(dict(
            process_id="P1", tool_id="T1", material_id=f"L{i:04d}",
            process_dynamics_m=M_true.copy(),
            initial_knob=u_init.copy(), recommended_knob=u_rec, used_knob=u_used,
            ff_disturbance=ff_dist, ff_gain_g=g, last_tool_state=tool_state.copy(),
            y_pred=target.copy(),
            post_meas=(y if measured else None),
            process_complete_dt=t0 + np.timedelta64(i, "h"),
        ))
        truth_ync.append(M_true @ u_init + d_total + noise if measured else None)
    return pd.DataFrame(rows), M_true, u_init, truth_ync


if __name__ == "__main__":
    import sys
    # explain_lot/trace output uses Unicode math symbols (Σ, Δ, ·); make sure
    # they survive a legacy (cp1252) Windows console.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    warnings.filterwarnings("ignore")
    df, M_true, u_init, truth = _make_synthetic()
    cfg = DiagnosticConfig(
        target=np.array([100.0, 50.0]),
        usl=np.array([103.0, 52.0]), lsl=np.array([97.0, 48.0]),
        m_rel_sigma=0.05, n_mc=400,
    )
    res = run_diagnostic(df, cfg)

    # --- correctness check: with M_assumed == M_true, y_nc must equal truth ---
    recon = res["reconstruction"]
    truth_arr = np.array([t for t in truth if t is not None])
    truth_arr = truth_arr[np.argsort(  # match recon ordering
        df[df["post_meas"].apply(_is_measured)]["process_complete_dt"].to_numpy())]
    ync = recon[["y_nc_0", "y_nc_1"]].to_numpy()
    max_err = np.max(np.abs(ync - truth_arr))
    print(f"[self-test] max |reconstructed y_nc - true y_nc| = {max_err:.2e} "
          f"(should be ~0 when supplied M is exact)\n")
    assert max_err < 1e-9, "reconstruction does not match ground truth!"

    # --- trace consistency: term-level expansion must rebuild reconstruct() ---
    trace = res["reconstruction_trace"]
    rebuilt = (trace.groupby([*cfg.group_cols, "material_id", "output_dim"])
               .agg(ctrl_eff_sum=("term", "sum"),
                    y_obs=("y_obs", "first"),
                    y_nc=("y_nc", "first")).reset_index())
    trace_err = np.max(np.abs(
        (rebuilt["y_obs"] - rebuilt["ctrl_eff_sum"]) - rebuilt["y_nc"]))
    print(f"[self-test] max |Σ(terms) rebuild - y_nc| = {trace_err:.2e} "
          f"(trace reproduces the reconstruction)\n")
    assert trace_err < 1e-9, "trace does not rebuild the reconstruction!"

    # --- show the per-lot arithmetic for one example lot ---------------------
    print("=== per-lot arithmetic trace (example lot) ===")
    print(explain_lot(df, cfg, index=0))
    print()

    print(summarize(res))
