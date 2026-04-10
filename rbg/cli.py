"""CLI for RBG — Rigorous Baseline Governance."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from rbg.audit import run_audit


BANNER = """
\033[1;31m ____  ____   ____
|  _ \\| __ ) / ___|
| |_) |  _ \\| |  _
|  _ <| |_) | |_| |
|_| \\_\\____/ \\____|
\033[0m
\033[2mRigorous Baseline Governance v0.1.0\033[0m
"""


@click.group()
def main():
    """RBG — Automated scientific paper auditor."""
    pass


@main.command()
@click.argument("notebook", type=click.Path(exists=True))
@click.option(
    "--data", "-d",
    multiple=True,
    type=click.Path(exists=True),
    help="CSV data file(s) to audit alongside the notebook.",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Output path for the HTML report.",
)
@click.option(
    "--open/--no-open",
    "open_browser",
    default=True,
    help="Open the report in a browser after generation.",
)
def audit(notebook: str, data: tuple[str, ...], output: str | None, open_browser: bool):
    """Run a three-layer audit on a Jupyter notebook.

    \b
    Examples:
      rbg audit analysis.ipynb
      rbg audit analysis.ipynb -d data.csv
      rbg audit analysis.ipynb -d data.csv -d controls.csv -o report.html
    """
    click.echo(BANNER)

    data_paths = list(data) if data else None

    click.echo(f"  Notebook:  {Path(notebook).name}")
    if data_paths:
        for dp in data_paths:
            click.echo(f"  Dataset:   {Path(dp).name}")
    click.echo()

    # Run audit
    with click.progressbar(
        length=3,
        label="  Auditing",
        bar_template="  %(label)s  [%(bar)s]  %(info)s",
        fill_char=click.style("#", fg="red"),
        empty_char=" ",
    ) as bar:
        result = run_audit(
            notebook_path=notebook,
            data_paths=data_paths,
            output_path=output,
        )
        bar.update(3)

    click.echo()

    # Score display
    score = result["score"]
    if score >= 80:
        score_color = "green"
        verdict = "CLEAN"
    elif score >= 60:
        score_color = "yellow"
        verdict = "NEEDS WORK"
    elif score >= 40:
        score_color = "red"
        verdict = "SIGNIFICANT ISSUES"
    else:
        score_color = "red"
        verdict = "CRITICAL ISSUES"

    click.echo(f"  Score:     {click.style(str(score), fg=score_color, bold=True)}/100 — {verdict}")
    click.echo(f"  Findings:  {result['total_findings']} total ({result['critical']} critical)")
    click.echo()

    layers = result["layers"]
    click.echo(f"    Layer 1 (Reproducibility):   {layers['reproducibility']} findings")
    click.echo(f"    Layer 2 (Statistics):         {layers['statistics']} findings")
    click.echo(f"    Layer 3 (Data Quality):       {layers['data_quality']} findings")
    click.echo()

    report_path = result["output_path"]
    click.echo(f"  Report:    {report_path}")

    # Open in browser
    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{Path(report_path).resolve()}")


@main.command()
@click.argument("directory", type=click.Path(exists=True), default=".")
@click.option("--output", "-o", type=click.Path(), default="batch_audit.html")
def batch(directory: str, output: str):
    """Audit all notebooks in a directory."""
    click.echo(BANNER)

    notebooks = list(Path(directory).rglob("*.ipynb"))
    if not notebooks:
        click.echo("  No .ipynb files found.")
        return

    click.echo(f"  Found {len(notebooks)} notebooks\n")

    results = []
    for nb_path in notebooks:
        try:
            # Find co-located CSV files
            csvs = list(nb_path.parent.glob("*.csv"))
            result = run_audit(
                notebook_path=str(nb_path),
                data_paths=[str(c) for c in csvs] if csvs else None,
            )
            results.append(result)
            score = result["score"]
            icon = "+" if score >= 60 else "!"
            click.echo(f"  [{icon}] {nb_path.name}: {score}/100 ({result['total_findings']} findings)")
        except Exception as e:
            click.echo(f"  [x] {nb_path.name}: ERROR — {e}")

    click.echo(f"\n  Individual reports written alongside each notebook.")


if __name__ == "__main__":
    main()
