# RBG

**Rigorous Baseline Governance** — automated audit for computational research.

Point it at a Jupyter notebook and a dataset. Get back a scored HTML report with every reproducibility, statistical, and data quality issue ranked by severity. Runs in seconds, no API key required.

```
rbg audit analysis.ipynb -d data.csv
```

<img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python 3.10+"/>

---

## Install

```bash
pip install rbg
```

For screenshot capture (optional):

```bash
pip install playwright
playwright install chromium
```

## Usage

### Audit a notebook

```bash
rbg audit notebook.ipynb
```

### Audit with a dataset (enables Layer 3)

```bash
rbg audit notebook.ipynb -d dataset.csv
```

### Multiple data files

```bash
rbg audit notebook.ipynb -d train.csv -d test.csv -o report.html
```

### Capture a PNG screenshot

```bash
rbg audit notebook.ipynb -d data.csv --screenshot --no-open
```

### Batch audit a directory

```bash
rbg batch ./notebooks/
```

## What It Checks

### Layer 1 — Reproducibility

- Stochastic operations without `random_state` (train/test splits, classifiers, cross-validation)
- Convergence warnings in model outputs
- Out-of-order cell execution and execution count gaps
- Variable overwrites across cells (kernel state pollution)

### Layer 2 — Statistical Integrity

- Naive majority-class baseline vs. reported accuracy
- Class imbalance without mitigation (`class_weight`, SMOTE, stratified splits)
- Confidence interval width on small test sets
- ROC curves computed on hard labels instead of probability scores
- Multiple statistical tests without correction
- 100% training accuracy (overfitting signal)

### Layer 3 — Data Quality

- Asymmetric missingness between groups (not MCAR)
- Physiologically impossible values (negative concentrations, etc.)
- Floor / limit-of-detection substitution artifacts
- Non-compositional percentage columns
- Raw + log-transformed feature redundancy

## Output

A self-contained HTML file. No external dependencies, no JavaScript CDN calls. Open it in any browser, email it to a collaborator, attach it to a review.

The report includes:

- **Score** (0–100) with severity breakdown
- **Expandable findings** organized by audit layer
- **Cell references** linking each finding to the notebook source

## Scoring

| Range | Verdict |
|-------|---------|
| 80–100 | Clean |
| 60–79 | Needs work |
| 40–59 | Significant issues |
| 0–39 | Critical issues |

Deductions scale with severity. Missing random seeds cost 5 points each (capped at 30). A model that scores below naive baseline costs 15. Asymmetric missingness above 10x costs 12.

## Project Structure

```
rbg/
  __init__.py
  audit.py           # Orchestrator — runs all three layers
  cli.py             # Click CLI
  report.py          # HTML report generator
  parsers/
    notebook.py      # .ipynb parser
    dataset.py       # CSV profiler
  checks/
    reproducibility.py   # Layer 1
    statistics.py        # Layer 2
    data_quality.py      # Layer 3
```

## Development

```bash
git clone https://github.com/SerialMiller/rbg.git
cd rbg
pip install -e ".[dev]"
pytest
```

## License

MIT
