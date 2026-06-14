# R2R Counterfactual ("No-Control") Analysis

Counterfactual reconstruction and diagnostics for a high-mix **run-to-run (R2R)**
process controller with feedforward (FF) + feedback (FB) control, MIMO process
gain, and delayed / sparse metrology.

The question it answers: **what would process performance have been if there had
been no control?** — given only operating data from the running controller.

## Core idea

The counterfactual is a *reconstruction*, not a forward simulation. Control acts
**only through the control knob** `u`; FF and FB both decide a knob move, and the
wafer only ever sees the knob. So the realized effect of all control on a measured
lot is captured entirely by how far the knob moved from its no-control baseline `u0`:

```
control_effect = M @ (u_used - u0)          # M = process gain ("dynamics" matrix)
y_nocontrol    = y_observed - control_effect # per measured lot
d_realized     = y_observed - M @ u_used     # disturbance backed out of y
```

The disturbance, the process's sensitivity to it, and the per-unit noise are all
already realized inside `y_observed` and ride along for free. **The only quantity
that must be accurate is `M`** (it is what we subtract) — its uncertainty is
propagated by Monte Carlo. The FF gain `g`, the FF function, and the controller's
tool-state estimate are *not* needed for the reconstruction; they feed only
secondary diagnostics.

See [r2r_counterfactual_reconstruction_reference.md](r2r_counterfactual_reconstruction_reference.md)
for the full reasoning, derivations, data mapping, and open threads.

## Files

| File | Purpose |
| --- | --- |
| [process_control_counterfactual_analysis.py](process_control_counterfactual_analysis.py) | The diagnostic library: reconstruction, term-level trace, dynamics tests, model-consistency / FF checks, performance metrics, and MC uncertainty. |
| [generate_html_report.py](generate_html_report.py) | Renders the diagnostic to a single self-contained HTML report you can open in a browser to inspect the calculation lot-by-lot. |
| [r2r_counterfactual_reconstruction_reference.md](r2r_counterfactual_reconstruction_reference.md) | Working reference doc — the *why* behind the method. |
| [r2r_report.html](r2r_report.html) | Example generated report (built from the synthetic self-test data). |

## Requirements

- Python 3.x
- `numpy`, `pandas`, `scipy` (no `statsmodels` needed)

```
pip install numpy pandas scipy
```

## Usage

Run the self-test on built-in synthetic data (verifies the reconstruction matches
ground truth, the trace rebuilds the reconstruction, and prints an example per-lot
derivation plus summary tables):

```
python process_control_counterfactual_analysis.py
```

Generate the HTML report from the synthetic data:

```
python generate_html_report.py        # writes r2r_report.html
```

On your own data:

```python
import numpy as np
from process_control_counterfactual_analysis import DiagnosticConfig, run_diagnostic, summarize

cfg = DiagnosticConfig(
    target=np.array([100.0, 50.0]),
    usl=np.array([103.0, 52.0]), lsl=np.array([97.0, 48.0]),
    m_rel_sigma=0.05, n_mc=400,   # propagate process-gain uncertainty
)
result = run_diagnostic(df, cfg)   # dict of DataFrames
print(summarize(result))
```

```python
from generate_html_report import build_report
build_report(df, cfg, out_path="report.html")
```

### Input data

One row per lot. Array-bearing cells hold lists / `np.ndarray`. Map your column
names via `DiagnosticConfig` (see its docstring for what is **required** vs
**optional**); key fields:

- `process_dynamics_m` — `(n_out, n_knob)` process gain `M` **(must be accurate)**
- `used_knob` — `(n_knob,)` actually-used knob setting `u_used`
- `initial_knob` / `recommended_knob` — for the no-control baseline `u0`
- `post_meas` — `(n_out,)` post-process measurement; `None`/`NaN` when unmeasured
- `ff_disturbance`, `ff_gain_g`, `last_tool_state`, `y_pred` — feed secondary diagnostics only
- `process_complete_dt` — ordering for the disturbance sequence

## What `run_diagnostic` returns

A dict of DataFrames:

- **reconstruction** — per measured lot: `y_obs`, `y_nc` (no-control), `d_real`, `ctrl_eff` for each output dim.
- **reconstruction_trace** — term-level expansion (one row per lot × output × knob) so any number can be audited; `explain_lot()` prints a fully-substituted derivation for a single lot.
- **dynamics** — per group/dim stationarity verdict on the realized disturbance (ADF + KPSS + variance-ratio), with a per-lot vs cumulative subtraction recommendation.
- **model_consistency** — checks `y_obs ≈ M@u + g·ff + tool_state`; structured/large residuals flag that the supplied `M`/`g` don't match reality.
- **ff_effectiveness** — how much of the realized disturbance the FF term explains (R²).
- **performance** — controlled (`y_obs`) vs reconstructed no-control (`y_nc`): std, variance reduction %, Cpk, out-of-spec fraction.
- **uncertainty** — variance-reduction % (and Cpk) percentiles under Monte-Carlo perturbation of `M` (`m_rel_sigma`).
