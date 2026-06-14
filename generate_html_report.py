"""
generate_html_report.py
=======================

Render the R2R counterfactual diagnostic as a single self-contained HTML file
you can open in a browser to *inspect the calculation logic* lot-by-lot.

The report contains:
  1. The per-lot arithmetic derivations (explain_lot) — the fully substituted
     y_nc = y_obs - M @ (u_used - u0) for each measured lot.
  2. The term-level reconstruction_trace table (one row per lot x output x knob).
  3. The summary diagnostic tables (dynamics, model consistency, performance,
     uncertainty).

Usage
-----
    # self-contained demo on the built-in synthetic data:
    python generate_html_report.py

    # or from your own code:
    from generate_html_report import build_report
    build_report(df, cfg, out_path="report.html")
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Optional

import pandas as pd

import process_control_counterfactual_analysis as pcca
from process_control_counterfactual_analysis import (
    DiagnosticConfig,
    explain_lot,
    run_diagnostic,
)

_CSS = """
:root { --fg:#1b1f24; --muted:#586069; --line:#e1e4e8; --accent:#0366d6;
        --code-bg:#f6f8fa; --term:#0a7d33; }
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       color: var(--fg); margin: 0; padding: 0 0 4rem; line-height: 1.5; }
header { background:#24292e; color:#fff; padding: 1.5rem 2rem; }
header h1 { margin: 0; font-size: 1.4rem; }
header p { margin: .35rem 0 0; color:#c8cdd2; font-size:.9rem; }
main { max-width: 1100px; margin: 0 auto; padding: 0 2rem; }
section { margin-top: 2.5rem; }
h2 { font-size: 1.15rem; border-bottom: 2px solid var(--line);
     padding-bottom: .3rem; }
.note { color: var(--muted); font-size: .88rem; }
details { border: 1px solid var(--line); border-radius: 6px; margin: .5rem 0;
          background:#fff; }
details > summary { cursor: pointer; padding: .6rem .9rem; font-weight: 600;
                    list-style: none; }
details > summary::-webkit-details-marker { display:none; }
details > summary:hover { background: var(--code-bg); }
details[open] > summary { border-bottom: 1px solid var(--line); }
pre.derivation { margin: 0; padding: .9rem 1.1rem; background: var(--code-bg);
                 font-family: "Cascadia Code", Consolas, monospace;
                 font-size: .82rem; overflow-x: auto;
                 border-radius: 0 0 6px 6px; }
table { border-collapse: collapse; width: 100%; font-size: .82rem;
        background:#fff; }
th, td { border: 1px solid var(--line); padding: .35rem .55rem;
         text-align: right; white-space: nowrap; }
th { background: var(--code-bg); position: sticky; top: 0; }
td:first-child, th:first-child { text-align: left; }
.table-wrap { max-height: 460px; overflow: auto; border: 1px solid var(--line);
              border-radius: 6px; }
.controls { margin: .5rem 0 1rem; }
.controls input { padding: .4rem .6rem; width: 100%; max-width: 320px;
                  border: 1px solid var(--line); border-radius: 6px; }
"""

# Tiny JS: live-filter the derivation blocks by lot id typed in the search box.
_JS = """
<script>
function filterLots(boxId, containerId) {
  const q = document.getElementById(boxId).value.toLowerCase();
  document.querySelectorAll('#' + containerId + ' > details').forEach(d => {
    const label = d.querySelector('summary').textContent.toLowerCase();
    d.style.display = label.includes(q) ? '' : 'none';
  });
}
</script>
"""


def _table_html(df: pd.DataFrame, float_fmt: str = "{:.4g}") -> str:
    if df is None or df.empty:
        return '<p class="note">(no rows)</p>'
    styled = df.to_html(index=False, border=0,
                        float_format=lambda x: float_fmt.format(x),
                        na_rep="")
    return f'<div class="table-wrap">{styled}</div>'


def build_report(df: pd.DataFrame, cfg: Optional[DiagnosticConfig] = None,
                 out_path: str = "report.html",
                 max_derivations: Optional[int] = None) -> Path:
    """Run the diagnostic on ``df`` and write a self-contained HTML report.

    Parameters
    ----------
    df, cfg          : same inputs as run_diagnostic.
    out_path         : where to write the .html file.
    max_derivations  : cap on how many per-lot derivations to embed
                       (None = all measured lots).
    """
    cfg = cfg or DiagnosticConfig()
    res = run_diagnostic(df, cfg)
    trace = res["reconstruction_trace"]

    # one explain_lot block per measured lot, in reconstruction order
    lots = (trace[[cfg.material_col]].drop_duplicates()
            [cfg.material_col].tolist())
    if max_derivations is not None:
        lots = lots[:max_derivations]

    deriv_blocks = []
    for mat in lots:
        text = explain_lot(df, cfg, material_id=mat)
        summary = html.escape(text.splitlines()[0])
        deriv_blocks.append(
            f"<details><summary>{summary}</summary>"
            f"<pre class='derivation'>{html.escape(text)}</pre></details>"
        )
    derivations = "\n".join(deriv_blocks)

    n_measured = len(lots)
    generated = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>R2R counterfactual diagnostic</title>",
        f"<style>{_CSS}</style></head><body>",
        "<header><h1>R2R counterfactual diagnostic &mdash; calculation trace</h1>",
        f"<p>Reconstruction: y_nc = y_obs &minus; M @ (u_used &minus; u0)"
        f" &nbsp;|&nbsp; baseline u0 = '{cfg.baseline}'"
        f" &nbsp;|&nbsp; generated {generated}</p></header>",
        "<main>",

        # 1. per-lot derivations
        "<section><h2>1. Per-lot arithmetic derivations</h2>",
        f"<p class='note'>Each block expands the reconstruction for one measured "
        f"lot, term by term. {n_measured} lot(s) shown.</p>",
        "<div class='controls'>"
        "<input id='lotSearch' placeholder='filter by lot id&hellip;' "
        "oninput=\"filterLots('lotSearch','derivs')\"></div>",
        f"<div id='derivs'>{derivations}</div></section>",

        # 2. term-level trace table
        "<section><h2>2. Reconstruction trace (term level)</h2>",
        "<p class='note'>One row per (lot, output dim, knob). "
        "<code>term = M &times; delta_u</code>; summing <code>term</code> over "
        "knobs gives <code>ctrl_eff</code>, and <code>y_obs &minus; ctrl_eff = "
        "y_nc</code>.</p>",
        _table_html(trace.drop(columns=[cfg.order_time_col], errors="ignore")),
        "</section>",

        # 3. summary tables
        "<section><h2>3. Diagnostic summaries</h2>",
        "<h3 class='note'>Performance: controlled vs no-control</h3>",
        _table_html(res["performance"]),
        "<h3 class='note'>Disturbance dynamics</h3>",
        _table_html(res["dynamics"]),
        "<h3 class='note'>Model consistency (supplied M/g check)</h3>",
        _table_html(res["model_consistency"]),
        "<h3 class='note'>FF effectiveness</h3>",
        _table_html(res["ff_effectiveness"]),
        "<h3 class='note'>Variance reduction with M uncertainty</h3>",
        _table_html(res["uncertainty"]),
        "</section>",

        "</main>",
        _JS,
        "</body></html>",
    ]

    path = Path(out_path)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


if __name__ == "__main__":
    import numpy as np

    df, *_ = pcca._make_synthetic()
    cfg = DiagnosticConfig(
        target=np.array([100.0, 50.0]),
        usl=np.array([103.0, 52.0]), lsl=np.array([97.0, 48.0]),
        m_rel_sigma=0.05, n_mc=400,
    )
    out = build_report(df, cfg, out_path="r2r_report.html")
    print(f"wrote {out.resolve()}")
