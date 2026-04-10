"""Main audit orchestrator — runs all three layers and produces the report."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from rbg.parsers.notebook import parse_notebook
from rbg.parsers.dataset import profile_dataset
from rbg.checks.reproducibility import check_reproducibility
from rbg.checks.statistics import check_statistics
from rbg.checks.data_quality import check_data_quality
from rbg.report import generate_html, compute_score


def run_audit(
    notebook_path: str,
    data_paths: Optional[list[str]] = None,
    output_path: Optional[str] = None,
) -> dict:
    """Run a full three-layer audit on a notebook and optional data files.

    Args:
        notebook_path: Path to a .ipynb file
        data_paths: Optional list of CSV data file paths
        output_path: Where to write the HTML report (default: <notebook>.audit.html)

    Returns:
        dict with score, findings count, and output path
    """
    nb_path = Path(notebook_path)
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")

    # Parse notebook
    nb = parse_notebook(nb_path)

    # Layer 1: Reproducibility
    repro = check_reproducibility(nb)

    # Parse datasets if provided
    dataset_profile = None
    df = None
    if data_paths:
        # Use the first CSV as the primary dataset
        for dp in data_paths:
            p = Path(dp)
            if p.suffix.lower() == ".csv" and p.exists():
                dataset_profile = profile_dataset(p)
                df = pd.read_csv(p)
                break

    # Layer 2: Statistical integrity
    stats = check_statistics(nb, dataset_profile)

    # Layer 3: Data quality
    dq = None
    if dataset_profile is not None:
        dq = check_data_quality(dataset_profile, df)

    # Generate HTML report
    html = generate_html(
        notebook_path=str(nb_path),
        repro=repro,
        stats=stats,
        dq=dq,
        dataset_path=str(data_paths[0]) if data_paths else None,
    )

    # Write report
    if output_path is None:
        output_path = str(nb_path.with_suffix(".audit.html"))

    out = Path(output_path)
    out.write_text(html)

    score = compute_score(repro, stats, dq)
    total_findings = (
        len(repro.all_findings)
        + len(stats.findings)
        + (len(dq.findings) if dq else 0)
    )

    return {
        "score": score,
        "total_findings": total_findings,
        "critical": repro.n_critical + stats.n_critical + (dq.n_critical if dq else 0),
        "output_path": str(out),
        "notebook": str(nb_path),
        "layers": {
            "reproducibility": len(repro.all_findings),
            "statistics": len(stats.findings),
            "data_quality": len(dq.findings) if dq else 0,
        },
    }
