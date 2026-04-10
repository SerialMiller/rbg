# RBG — Rigorous Baseline Governance

## The Problem

Computational research papers ship with Jupyter notebooks and datasets that no one audits mechanically. Reviewers read the manuscript — they rarely clone the repo, rerun the code, and check whether a `train_test_split` was seeded or a confusion matrix implies the model underperforms a coin flip. The checks are straightforward. They just take 20 hours that nobody has.

## The Product

RBG is a CLI tool that audits a Jupyter notebook and its associated data in under 5 seconds. No API key. No cloud. Pure Python.

```
rbg audit notebook.ipynb -d dataset.csv
```

Output: a self-contained HTML report with a 0–100 score and every finding ranked by severity.

## Three Audit Layers

**Layer 1 — Reproducibility**
Missing random seeds, convergence failures, out-of-order execution, variable overwrites across cells. Mechanical checks that catch whether results can be reproduced at all.

**Layer 2 — Statistical Integrity**
Naive baseline comparison, class imbalance without mitigation, confidence interval width on small test sets, ROC curves on hard labels, multiple comparisons without correction. The math that separates a real finding from a misleading number.

**Layer 3 — Data Quality**
Asymmetric missingness between groups, physiologically impossible values, limit-of-detection substitution, non-compositional percentages, raw + log-transformed feature redundancy. The data problems that invalidate downstream analysis.

## Who It's For

- **Authors** — run before submission, fix what's fixable, own what's not in your limitations section
- **Reviewers** — triage in 5 seconds instead of 5 hours, focus human attention on domain questions
- **Editors** — attach an RBG report to any computational submission as a baseline quality check
- **Teaching** — show students what a rigorous methods audit looks like, automatically

## What's Next

**Interactive playground.** Upload a notebook and data in the browser. RBG reruns the notebook with randomized seeds to test result stability. Shareable report URLs for review workflows. No local install required.

## Technical

- Python 3.10+
- Dependencies: pandas, numpy, scipy, click, nbformat
- Optional: Playwright for PNG screenshot capture
- MIT license
- github.com/SerialMiller/rbg
