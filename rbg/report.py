"""HTML report generator for RBG audit results."""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Optional

from rbg.checks.reproducibility import ReproducibilityReport
from rbg.checks.statistics import StatisticalReport
from rbg.checks.data_quality import DataQualityReport


def _severity_badge(severity: str) -> str:
    colors = {
        "critical": "#dc2626",
        "high": "#ea580c",
        "medium": "#ca8a04",
        "low": "#2563eb",
    }
    color = colors.get(severity, "#6b7280")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:12px;font-weight:600;text-transform:uppercase;">'
        f'{severity}</span>'
    )


def _severity_icon(severity: str) -> str:
    icons = {
        "critical": "&#x1F6A8;",  # siren
        "high": "&#x26A0;&#xFE0F;",  # warning
        "medium": "&#x1F50D;",  # magnifying glass
        "low": "&#x1F4AC;",  # speech bubble
    }
    return icons.get(severity, "")


def _score_ring(score: int) -> str:
    """SVG ring showing audit score 0-100."""
    if score >= 80:
        color = "#16a34a"  # green
    elif score >= 60:
        color = "#ca8a04"  # yellow
    elif score >= 40:
        color = "#ea580c"  # orange
    else:
        color = "#dc2626"  # red

    circumference = 2 * 3.14159 * 45
    offset = circumference * (1 - score / 100)

    return f"""
    <svg width="140" height="140" viewBox="0 0 140 140">
      <circle cx="70" cy="70" r="45" fill="none" stroke="#e5e7eb" stroke-width="10"/>
      <circle cx="70" cy="70" r="45" fill="none" stroke="{color}" stroke-width="10"
              stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"
              stroke-linecap="round" transform="rotate(-90 70 70)"
              style="transition: stroke-dashoffset 1s ease;"/>
      <text x="70" y="65" text-anchor="middle" font-size="32" font-weight="700" fill="{color}">{score}</text>
      <text x="70" y="85" text-anchor="middle" font-size="12" fill="#6b7280">/ 100</text>
    </svg>
    """


def compute_score(
    repro: ReproducibilityReport,
    stats: StatisticalReport,
    dq: Optional[DataQualityReport] = None,
) -> int:
    """Compute an overall audit score (100 = clean, 0 = disaster)."""
    score = 100

    # Reproducibility deductions
    score -= min(30, len(repro.seed_findings) * 5)
    score -= min(15, len(repro.convergence_findings) * 10)
    score -= min(10, len(repro.execution_findings) * 3)
    score -= min(15, len(repro.overwrite_findings) * 5)

    # Statistics deductions
    for f in stats.findings:
        if f.severity == "critical":
            score -= 15
        elif f.severity == "high":
            score -= 8
        elif f.severity == "medium":
            score -= 3

    # Data quality deductions
    if dq:
        for f in dq.findings:
            if f.severity == "critical":
                score -= 12
            elif f.severity == "high":
                score -= 6
            elif f.severity == "medium":
                score -= 2

    return max(0, min(100, score))


def generate_html(
    notebook_path: str,
    repro: ReproducibilityReport,
    stats: StatisticalReport,
    dq: Optional[DataQualityReport] = None,
    dataset_path: Optional[str] = None,
) -> str:
    """Generate a self-contained HTML audit report."""

    score = compute_score(repro, stats, dq)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    nb_name = Path(notebook_path).name
    ds_name = Path(dataset_path).name if dataset_path else "None"

    # Count totals
    total_critical = repro.n_critical + stats.n_critical + (dq.n_critical if dq else 0)
    total_high = repro.n_warnings + stats.n_high + (dq.n_high if dq else 0)

    # Build finding rows
    def _finding_row(finding, layer: str) -> str:
        summary = html.escape(getattr(finding, "summary", str(finding)))
        detail = html.escape(getattr(finding, "detail", ""))
        severity = getattr(finding, "severity", "medium")
        cell_idx = getattr(finding, "cell_index", None)
        cell_str = f" <code>Cell [{cell_idx}]</code>" if cell_idx is not None else ""

        return f"""
        <details class="finding">
          <summary>
            {_severity_badge(severity)}
            <span class="finding-layer">{layer}</span>
            {summary}{cell_str}
          </summary>
          <div class="finding-detail">{detail}</div>
        </details>
        """

    # Layer 1 findings
    layer1_html = ""
    for f in repro.seed_findings:
        layer1_html += _finding_row(f, "Reproducibility")
    for f in repro.convergence_findings:
        layer1_html += _finding_row(f, "Reproducibility")
    for f in repro.execution_findings:
        layer1_html += _finding_row(f, "Execution Order")
    for f in repro.overwrite_findings:
        layer1_html += _finding_row(f, "State Pollution")

    # Layer 2 findings
    layer2_html = ""
    for f in stats.findings:
        layer2_html += _finding_row(f, "Statistics")

    # Layer 3 findings
    layer3_html = ""
    if dq:
        for f in dq.findings:
            layer3_html += _finding_row(f, "Data Quality")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>RBG Audit: {html.escape(nb_name)}</title>
<style>
  :root {{
    --bg: #0f172a;
    --surface: #1e293b;
    --surface2: #334155;
    --text: #f1f5f9;
    --text-dim: #94a3b8;
    --border: #475569;
    --accent: #3b82f6;
    --critical: #dc2626;
    --high: #ea580c;
    --medium: #ca8a04;
    --green: #16a34a;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 24px;
    max-width: 960px;
    margin: 0 auto;
  }}
  h1 {{
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
  }}
  h1 .rbg {{ color: var(--critical); }}
  h2 {{
    font-size: 18px;
    font-weight: 700;
    margin: 32px 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}
  .subtitle {{
    color: var(--text-dim);
    font-size: 14px;
    margin-bottom: 24px;
  }}
  .header {{
    display: flex;
    align-items: center;
    gap: 32px;
    margin-bottom: 32px;
    padding: 24px;
    background: var(--surface);
    border-radius: 12px;
    border: 1px solid var(--border);
  }}
  .header-meta {{
    flex: 1;
  }}
  .meta-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 16px;
    font-size: 13px;
  }}
  .meta-grid dt {{ color: var(--text-dim); }}
  .meta-grid dd {{ color: var(--text); font-weight: 600; }}
  .stats-bar {{
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
  }}
  .stat-card {{
    flex: 1;
    padding: 16px;
    background: var(--surface);
    border-radius: 8px;
    border: 1px solid var(--border);
    text-align: center;
  }}
  .stat-card .number {{
    font-size: 28px;
    font-weight: 800;
  }}
  .stat-card .label {{
    font-size: 12px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 1px;
  }}
  .stat-card.critical .number {{ color: var(--critical); }}
  .stat-card.high .number {{ color: var(--high); }}
  .stat-card.medium .number {{ color: var(--medium); }}
  .stat-card.green .number {{ color: var(--green); }}

  .layer-section {{
    margin-bottom: 24px;
  }}
  .layer-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 12px;
  }}
  .layer-count {{
    background: var(--surface2);
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 12px;
    color: var(--text-dim);
  }}
  .finding {{
    margin-bottom: 4px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }}
  .finding > summary {{
    padding: 10px 16px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    list-style: none;
  }}
  .finding > summary::-webkit-details-marker {{ display: none; }}
  .finding > summary::before {{
    content: "\\25B6";
    font-size: 10px;
    color: var(--text-dim);
    transition: transform 0.2s;
  }}
  .finding[open] > summary::before {{
    transform: rotate(90deg);
  }}
  .finding-layer {{
    color: var(--text-dim);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    min-width: 100px;
  }}
  .finding-detail {{
    padding: 12px 16px 16px 42px;
    font-size: 13px;
    color: var(--text-dim);
    line-height: 1.7;
    border-top: 1px solid var(--border);
  }}
  code {{
    background: var(--surface2);
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }}
  .empty-state {{
    padding: 24px;
    text-align: center;
    color: var(--text-dim);
    font-size: 14px;
    background: var(--surface);
    border-radius: 8px;
    border: 1px dashed var(--border);
  }}
  .footer {{
    margin-top: 48px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    font-size: 11px;
    color: var(--text-dim);
    text-align: center;
  }}
  .footer a {{ color: var(--accent); text-decoration: none; }}
</style>
</head>
<body>

<div class="header">
  <div>{_score_ring(score)}</div>
  <div class="header-meta">
    <h1><span class="rbg">RBG</span> Audit Report</h1>
    <div class="subtitle">Rigorous Baseline Governance</div>
    <dl class="meta-grid">
      <dt>Notebook</dt><dd>{html.escape(nb_name)}</dd>
      <dt>Dataset</dt><dd>{html.escape(ds_name)}</dd>
      <dt>Audit Date</dt><dd>{now}</dd>
      <dt>RBG Version</dt><dd>0.1.0</dd>
    </dl>
  </div>
</div>

<div class="stats-bar">
  <div class="stat-card critical">
    <div class="number">{total_critical}</div>
    <div class="label">Critical</div>
  </div>
  <div class="stat-card high">
    <div class="number">{total_high}</div>
    <div class="label">High</div>
  </div>
  <div class="stat-card medium">
    <div class="number">{len(repro.all_findings) + len(stats.findings) + (len(dq.findings) if dq else 0) - total_critical - total_high}</div>
    <div class="label">Medium / Low</div>
  </div>
  <div class="stat-card green">
    <div class="number">{score}</div>
    <div class="label">Score</div>
  </div>
</div>

<h2>Layer 1 &mdash; Reproducibility</h2>
<div class="layer-section">
  <div class="layer-header">
    Seed &amp; Execution Audit
    <span class="layer-count">{len(repro.all_findings)} findings</span>
  </div>
  {layer1_html if layer1_html else '<div class="empty-state">No reproducibility issues detected.</div>'}
</div>

<h2>Layer 2 &mdash; Statistical Integrity</h2>
<div class="layer-section">
  <div class="layer-header">
    Baselines, Metrics &amp; Validation
    <span class="layer-count">{len(stats.findings)} findings</span>
  </div>
  {layer2_html if layer2_html else '<div class="empty-state">No statistical integrity issues detected.</div>'}
</div>

<h2>Layer 3 &mdash; Data Quality</h2>
<div class="layer-section">
  <div class="layer-header">
    Missingness, Impossible Values &amp; Artifacts
    <span class="layer-count">{len(dq.findings) if dq else 0} findings</span>
  </div>
  {layer3_html if layer3_html else '<div class="empty-state">No data files provided for Layer 3 audit, or no issues detected.</div>'}
</div>

<div class="footer">
  <strong>RBG</strong> &mdash; Rigorous Baseline Governance v0.1.0<br/>
  Generated {now} &bull;
  <a href="https://github.com/SerialMiller/rbg">github.com/SerialMiller/rbg</a>
</div>

</body>
</html>"""
