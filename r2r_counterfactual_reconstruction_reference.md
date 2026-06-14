# Counterfactual ("No-Control") Reconstruction for Run-to-Run Process Control
### Working reference — reasoning, method, data mapping, implementation, and open threads

> **Purpose of this doc.** Capture the full reasoning and decisions from a working
> session so that a future Claude (or a human researcher) can continue without the
> original chat. It records *why*, not just *what*. Companion artifact:
> `r2r_counterfactual_diagnostic.py` (validated, runs on numpy/pandas/scipy; no
> statsmodels needed).

---

## 0. TL;DR — if you read only this

We want to estimate **what process performance would have been if there had been no
control**, given operating data from a feedforward (FF) + feedback (FB) run-to-run
(R2R) controller.

The key realization that collapses the problem:

> **The counterfactual is a *reconstruction*, not a forward simulation. You take what
> actually happened and subtract only the control action's effect. Everything else —
> the disturbance, the process's sensitivity to it, the per-unit noise — is already
> realized inside the observed output and rides along for free.**

For this specific system, control acts **only through the control knob** `u`. FF and FB
both compute a knob move; the wafer only ever sees the knob. So:

```
control_effect = M @ (u_used - u0)          # M = process gain (the "dynamics" matrix)
y_nocontrol    = y_observed - control_effect # per measured lot
```

where `u0` is the no-control baseline knob (recipe nominal). **The only quantity that
must be accurate is `M`** (it is what we subtract). The FF gain `g`, the FF function,
and the controller's tool-state estimate are **not needed** for the reconstruction —
they were only used to *decide* `u`, and we observe `u_used` directly.

This directly answers the two original worries:
- "We don't know `m`/process sensitivity accurately" → **it cancels; never used.**
- "We don't know the unexplained per-unit incoming variation" → **already realized in
  `y_observed`; reconstruction keeps each unit's true incoming condition automatically.**

---

## 1. Problem statement

Starting control model (as originally posed):

```
y = m·x + g·FF + c + error
```

- `y` — post-process measurement (process output / quality), deviation from target.
- `x` — incoming variation / incoming condition (measured disturbance).
- `m` — process sensitivity to incoming variation (how much uncontrolled `x` moves `y`).
- `FF` — feedforward control action; `g` — its gain.
- `c` — offset; `error` — unexplained residual.
- FB (feedback) action is also present in the data.

**Goal:** simulate the counterfactual — process performance with **no control** (FF and
FB both off).

**Two stated concerns:**
1. We don't know the accuracy of the process behavior `m` and the FF gain `g`.
2. We don't know the potential unexplained incoming variation; each processing unit may
   have different incoming conditions.

**Context (revealed later):** high-mix semiconductor-style R2R control. MIMO. Metrology
is **sparse and delayed** — only a small fraction of lots are measured, and a lot's
measurement can arrive after several later lots have already been processed.

---

## 2. The core insight (derivation)

Write the controlled (observed) output with all control terms explicit:

```
y_t = m·x_t + g·FF_t + g_fb·FB_t + c + ε_t        (observed, controlled)
```

The **no-control** version of *that same realized run* (same incoming material, same
realized disturbance, same noise) is by definition the same expression with the control
actions set to zero:

```
y_nc_t = m·x_t + c + ε_t                            (counterfactual)
```

Subtract:

```
y_nc_t = y_t − g·FF_t − g_fb·FB_t
```

`m`, `c`, and `ε_t` **cancel**. They are carried inside `y_t`. So:
- Not knowing `m` does not hurt reconstruction at all (it never appears).
- The unexplained per-unit variation for the *observed* units is in `y_t`, so each unit's
  realized incoming condition is preserved without modeling it.
- The only thing we subtract — hence the only thing that must be accurate — is the
  **control-action effect** (`g`, `g_fb`, and the dynamics).

### 2.1 System-specific sharpening: the actuator is the knob

In this R2R system the controller's sole actuator is the **control knob** `u`. FF and FB
do not act on the output directly; each produces a recommended knob move, and the wafer
only ever experiences the knob. The plant is `y = M·u + d + noise` (`M` = process gain
matrix, `d` = disturbance). Therefore the **entire** realized control effect is the knob
deviation from the no-control baseline:

```
control_effect_t = M·(u_used_t − u0_t)
y_nc_t           = y_t − control_effect_t
d_realized_t     = y_t − M·u_used_t            # disturbance backed out of the output
```

**Consequence:** `g`, the FF disturbance estimate `f(·)`, and the tool-state estimate are
*irrelevant to the reconstruction*. They only explain how `u` was chosen. We observe
`u_used` directly, so we don't need them. (They remain useful for secondary diagnostics
— see §6–7.)

---

## 3. Conceptual distinctions and caveats (the parts that can make the number wrong)

### 3.1 Static vs. integrating — per-lot vs. cumulative subtraction
The simple per-lot subtraction is exact only if the process is **memoryless** (control at
`t` affects output only at `t`). If the disturbance is **integrating/drifting** (random
walk) *and the control action feeds a state that physically carries forward*, then
removing control removes the **cumulative** effect of all past moves and the disturbance
runs away:

```
y_nc_t ≈ y_t − g·Σ_{i≤t} FF_i − g_fb·Σ_{i≤t} FB_i      (integrating case)
```

A static `y = m·x + c` model badly **understates** how bad no-control is, because it can't
show drift accumulating — and drift is the usual *reason* R2R control exists.

**Resolution for this system:** the integration lives in the **controller's tool-state
estimator** (an EWMA of the disturbance), *not* in the physical process — provided the
disturbance is exogenous to the knob (next point). The knob sets conditions for one lot
and does not physically integrate the disturbance. We bypass the estimator entirely by
observing `u_used`. **So per-lot reconstruction is valid here.** Drift still shows up
naturally as a trend across the sequence of reconstructed `d_realized`.

### 3.2 Exogeneity assumption (the validity gate for §3.1)
Per-lot reconstruction requires the disturbance be **exogenous to the knob** — the knob
must not physically alter tool drift (e.g., a setting that accelerates wear/clean cycles).
If such coupling exists, you need a dynamic / state-space reconstruction instead.
**Confirm this with process engineers.**

### 3.3 Closed-loop identification (only if *estimating* gains)
If you ever estimate `M`/`g` from operating data (rather than using supplied values), two
traps:
- **FF/x collinearity:** if `FF = −k·x` deterministically, regression identifies only the
  combination `(m − g·k)`, never `g` alone. You need episodes where FF varies independently
  of `x` (FF disabled or gain-changed on some units).
- **FB endogeneity:** `FB_t` is computed from past outputs, which contain past
  disturbances; if the disturbance is autocorrelated, `FB_t` correlates with the current
  error → OLS gains are **biased**, and the bias flows into the counterfactual.
- **Fixes (in order):** designed step/bump/dither tests (immune to disturbance
  correlation) > physical/engineering priors > proper closed-loop ID / instrumental
  variables. **Never** naive OLS on closed-loop data.

In the current setup `M` and `g` are **supplied** as data columns, so identification is
moot — but their **accuracy must be validated** against the data (see §7 model-consistency,
and the caveat in §9).

### 3.4 Two goals — reconstruction vs. generative simulation
- **Reconstruction / retrodiction:** "for the units I actually ran, what would they have
  done with no control?" → the subtraction above. Assumption-light; concern (2) does not
  bite (realized disturbance reused). Best, most defensible "what control bought us over
  this period" number. **This is the primary deliverable.**
- **Generative simulation:** "what would process capability be in general / on future
  units?" → cannot reuse realized disturbances; must **fit a stochastic disturbance model**
  (ARIMA / state-space: AR(1) stationary, IMA(1,1)/random walk for drift) + per-unit random
  effect + measurement noise, then Monte-Carlo new trajectories with control = 0. **Only
  here** does concern (2) genuinely re-enter.

### 3.5 Uncertainty propagation
Do not report point gains. Draw `M` (and dynamics) from its uncertainty
(covariance / bootstrap / posterior), rerun, report the spread.
- `m`'s uncertainty contributes **~nothing** (it cancels).
- Gain (`M`) uncertainty **dominates**, scaling per sample as `~ Δu² · Var(M̂)`.

---

## 4. Analysis workflow (condensed, 8 steps)

1. **Frame & metric.** Reconstruction vs. generative? Pick one metric (σ, Cpk, %OOS,
   yield); it drives the precision needed downstream.
2. **Assemble & clean data.** One time-ordered table per control group. Fix deadtime
   alignment (pair each action with the output it influenced). Remove sensor faults.
   **Flag any control-off episodes — these are validation gold; set them aside.**
3. **Gains (linchpin).** Prefer designed tests > priors > closed-loop ID. Output point
   estimates **and covariance**. *Identifiability gate:* check `corr(FF, x)`; if FF is a
   fixed function of `x` you cannot recover `g` from observation alone. (Here: validate
   the *supplied* `M`/`g` instead — §7, §9.)
4. **Diagnose dynamics.** Form `d_realized`, run ADF + KPSS + variance-ratio, look at
   ACF/PACF. *Static/integrating gate:* stationary → subtract per-lot; unit root →
   cumulative / state-space.
5. **Reconstruct.** Stationary: `y_nc = y − M·(u_used − u0)`. Integrating: cumulative, or
   Kalman smoother on a state-space model (cleaner; separates measurement noise; gives
   uncertainty).
6. **Propagate gain uncertainty.** Monte-Carlo over `M` draws → distribution of the metric.
   (Generative branch: simulate disturbance trajectories with control = 0.)
7. **Validate.** Compare reconstruction on control-off episodes to actuals. Sanity gates:
   `var(no-control) ≥ var(controlled)`; drift appears if expected. Fail → back to 3/4.
8. **Report.** Metric controlled vs. no-control with intervals; state the three
   assumptions (which gains + how obtained; static vs. integrating; exogeneity); lead with
   the validation result.

---

## 5. Data schema (the actual R2R system)

Grouped by lifecycle stage. **Cells hold arrays/matrices** (MIMO).

**Control context (grouping key — all data in a group shares the control calculation /
controller state):** `process_id`, `tool_id`.

**Data key:** `material_id` (lot).

**Pre-staging (from prior steps):**
- incoming measurement datetime
- incoming measurement values — **matrix** (the incoming variation `x`).

**Staging (before processing):**
- staging datetime
- FF variable → **FF disturbance** = `f(measurement)`, a possibly **nonlinear** map from
  incoming + tool variables (wafers-in-tool, time-since-clean, vibration, …) to a
  disturbance estimate.
- FF gain `g`.
- process dynamics `M` — **matrix** (MIMO process gain, knob→output).
- last estimated **tool state** (from the most recent FB calc — the FB/EWMA memory).
- initial control knob setting.
- recommended control knob setting.
- predicted post-meas `y_pred` (ideally on target; deviates due to knob precision /
  deadband; stored so FB can later account for "recommendation wasn't the math optimum").

**Tool data (after processing):**
- process completion datetime
- **actual used** control knob setting (can differ from recommended due to manual changes).

**Post-meas (sparse — only a small fraction of lots):**
- post-meas datetime
- post-meas values — **array** (the output `y`).
- updated tool state (accounts for used≠recommended and `y_pred`≠target).
- recommended control knob (FB-alone, clean FB; next staging handles FF).

**Critical timing note:** post-meas of lot A can arrive after lots B/C/D have already been
processed. Metrology is delayed/sparse by design — keep material moving, assume knob
adjustments help future lots. This trades quality (low variation) against throughput.

---

## 6. Data → diagnostic mapping

**Assumed controller model** (output units; *sign/scaling conventions are an open
question — see §8*):

```
y_pred = M·u + g·(FF disturbance) + tool_state           (controller's internal model)
u_recommended = M⁻¹·(target − g·FF − tool_state)         (drive y_pred to target; MIMO inverse/pinv)
```

FB update at a measured lot (schematic EWMA):
```
d_obs       = y_actual − M·u_used                        (back out disturbance)
tool_state ← (1−λ)·tool_state + λ·(d_obs − g·FF)
```

**Reconstruction (what the diagnostic actually computes):**
```
u0             = initial_knob              # no-control baseline (CONFIRM — §8)
control_effect = M·(u_used − u0)
y_nocontrol    = y_actual − control_effect
d_realized     = y_actual − M·u_used       # disturbance signal for dynamics analysis
```

**Field usage map:**
- **Used by reconstruction:** `M`, `used_knob`, baseline knob (`initial_knob`).
- **NOT used by reconstruction (only secondary diagnostics):** `g`, FF disturbance,
  `tool_state`, `y_pred`, `recommended_knob`.
- **Needed but absent from schema:** target, spec limits, `M` uncertainty/covariance.

---

## 7. Implementation — `r2r_counterfactual_diagnostic.py`

**Status:** written and **validated**. Self-test on synthetic R2R data confirms the
reconstruction reproduces ground-truth `y_nc` to ~1e-14 (machine precision) when the
supplied `M` equals the true `M`. Pure numpy/pandas/scipy (statsmodels unavailable in the
environment, so the stationarity tests are implemented from scratch).

**Functions → workflow steps:**
- `reconstruct(df, cfg)` — §5: per-measured-lot `y_obs`, `y_nc`, `d_realized`,
  `control_effect`. Core.
- `diagnose_dynamics` / `stationarity_test` — §4: ADF t-stat, KPSS LM stat, variance-ratio
  VR(2)/VR(4), lag-1 autocorrelation → verdict (`stationary` / `drifting (unit root)` /
  `inconclusive`) and `recommended_subtraction` (`per-lot` / `cumulative`).
- `model_consistency` — §3.3 gate (validates supplied `M`/`g`/`tool_state`): residual of
  `y − (M·u + g·FF + tool_state)`; RMS, mean, lag-1 autocorr, residual-variance fraction.
- `ff_effectiveness` — how much of `d_realized` the FF term explains (R²).
- `performance` — controlled vs. no-control: σ, variance-reduction %, Cpk, %OOS (Cpk/%OOS
  require specs).
- `mc_uncertainty` — §6: systematic relative perturbation of `M` (`m_rel_sigma`),
  re-derives variance-reduction % and `Cpk_nc`, returns p05/p50/p95.
- `run_diagnostic` orchestrates; `summarize` prints a digest. `_make_synthetic` builds the
  self-test dataset.

**`DiagnosticConfig` — key knobs:** column-name mapping; `baseline`
(`initial`/`recommended`/`zero`/`custom`) defining the no-control knob; `ff_already_scaled`;
`target`/`usl`/`lsl`; `m_rel_sigma` + `n_mc`; `min_n_for_dynamics`.

**Usage:** map columns into `DiagnosticConfig`, then `run_diagnostic(df, cfg)`.
`post_meas` is `None`/`NaN` for unmeasured lots; the code filters to measured lots itself.

**Synthetic-run sanity numbers (illustrative):** control cut σ ~3.2 → ~1.8 (≈65% variance
reduction); `Cpk_controlled` modest-positive, `Cpk_nocontrol` strongly negative (drift runs
off-target); MC band on variance-reduction tight at 5% gain uncertainty.

---

## 8. Open questions / information still needed

**Blocking (core reconstruction):**
1. **No-control baseline knob.** Confirm `initial_knob` *is* the open-loop/recipe-nominal
   setting (knob with the controller off). The entire counterfactual = "knob held at this
   baseline."
2. **Shape contract.** #knobs, #output dims in `post_meas`, layout of `M` as
   `(n_out × n_knob)`. If incoming is a site-map matrix and post-meas is per-site, define
   the mapping (per-site independent vs. averaged) — it decides per-site vs. summary
   analysis.

**For the full diagnostic (metrics + uncertainty):**
3. **Targets and spec limits** per output (`target`, `usl`, `lsl`) — absent from schema;
   needed for Cpk / %OOS (usually the headline number).
4. **Uncertainty on `M`** (`m_rel_sigma` or a per-element covariance) — the dominant
   uncertainty; get a relative 1σ from step-test history.

**Validity confirmations (decide per-lot vs. cumulative):**
5. **Is the disturbance exogenous to the knob?** (No knob→drift physical coupling.) If not,
   move to a state-space/Kalman reconstruction.
6. **Ordering timestamp** for the disturbance sequence — defaulted to
   `process_complete_dt` (physical order), not `post_meas_dt` (when learned). Confirm.

**Validation data:**
7. Any **control-off episodes** (outages, maintenance, pre-deployment lots) to test the
   reconstruction against reality.

---

## 9. Known caveats / gotchas (discovered during the session)

- **`model_consistency` conflates a wrong `M` with FB tracking lag.** Because `tool_state`
  is an EWMA that lags under sparse/delayed metrology, residuals are inflated even when `M`
  is perfect. Read a large residual as "something is off (gain, FF model, *or* tracking),"
  not specifically "M is wrong." **To isolate `M` you need knob moves decorrelated from the
  disturbance.** The schema's `y_pred` is the better lever: compare `y_pred` (corrected for
  `used − recommended` knob via `M`) against actual `y` for a cleaner controller-prediction
  error. *(Not yet implemented — see §10.)*
- **Drift detection needs enough measured lots per group.** On ~60 measured points the
  verdict can come back `inconclusive`. Treat the stationarity call as a guide; lean on the
  plotted `d_realized` trajectory.
- **Irregular spacing.** The disturbance series is irregularly spaced in time (sparse
  metrology) but the ADF/KPSS/ACF treat it as evenly spaced. Acceptable as a diagnostic;
  note the caveat, or resample/interpolate if rigor demands.
- **Baseline sensitivity.** The whole counterfactual hinges on `u0`; a wrong baseline
  silently biases every number.

---

## 10. Suggested next steps

1. **Resolve §8 blocking items** (baseline knob, shape contract) and hand over targets/specs
   + a rough `M` uncertainty → the script runs cleanly and produces Cpk and an uncertainty
   band on the real frame.
2. **Add a `y_pred`-based consistency check** to isolate `M` accuracy from FB tracking lag
   (§9). Residual `= y − y_pred − M·(u_used − u_recommended)`.
3. **Per-site handling** if incoming/post-meas are site maps (loop the diagnostic per site
   or per principal component).
4. **Validate against any control-off periods** (§8.7) — the single best calibration of
   whether `M` and the dynamics assumption are right.
5. **If §3.2 exogeneity fails (or §4 says unit root with coupling):** build the state-space
   / Kalman-smoother version (disturbance state with an integrator; observation =
   control-effect + disturbance + noise) for a cumulative-correct reconstruction with
   built-in uncertainty.
6. **If the goal shifts to process *capability* (not just these lots):** fit the disturbance
   stochastic model from `d_realized` and Monte-Carlo a generative no-control simulation
   (§3.4) — this is the only path where the unexplained-variation distribution must be
   modeled explicitly.

---

*End of reference. Companion file: `r2r_counterfactual_diagnostic.py`.*
