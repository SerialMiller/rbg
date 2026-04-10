# What We Built Today

## The Session (April 9, 2026)

Started with a feature request to document 7 BHDX research files.
Ended with a working product that didn't exist 2 hours ago.

---

## Phase 1: The BHDX Documentation Factory

Dispatched 14 AI agents (7 Scalia + 7 RBG) to produce dual-perspective
documentation for every data file in John Miller's HDL/CHD paper.

**Output:** 7 complete documents (139KB total), each containing:
- A Majority Opinion grounded in the actual data
- A Dissent that found real, non-trivial issues

**What the Dissents found (not hypothetical):**
- Zero random seeds across all notebooks
- A kernel state pollution bug in ML_notebook_v3 (male importance table is actually the female model's data, displayed twice)
- A copy-paste bug in CAD Workbook v3 cell 39 (male model evaluated on female data)
- 51.5% case-missingness in the headline subfraction analysis
- ROC curves computed on hard labels (not probability scores)
- No medication data in a study where statins confound HDL/CRP findings
- A null classifier that beats the ASCVD model by 12.6 percentage points

**Pushed to:** github.com/SerialMiller/BHDX

---

## Phase 2: RBG — Rigorous Baseline Governance

Realized the Dissent process was a prototype for a general scientific
auditor. Built it as a standalone Python tool in one session.

**What it is:** `pip install rbg` then `rbg audit notebook.ipynb -d data.csv`

**Three audit layers:**
1. **Reproducibility** — seeds, convergence, execution order, variable overwrites
2. **Statistical Integrity** — naive baselines, class balance, ROC validity, CI width
3. **Data Quality** — asymmetric missingness, impossible values, floor substitution

**Output:** Self-contained HTML report with:
- Score ring (0-100)
- Severity-ranked findings (critical/high/medium)
- Expandable detail for each finding
- Dark theme, no dependencies, shareable as a single file

**First dogfood results (on our own data):**
- ML_notebook_v1: 0/100, 20 findings (10 critical)
- CAD Workbook v3: 12/100, 20 findings (7 critical)
- COVID notebook: 0/100, 19 findings (8 critical)

It found everything the RBG agents found, mechanically, in under 2 seconds per notebook.

---

## The Product Vision

### Today: RBG CLI (done, working)
Automated mechanical audit for any Jupyter notebook + dataset.
No API key required. Pure Python. Runs in 2 seconds.

### Next: RBG HTML Playground (the moonshot)
Think PR review environments but for scientific papers:
- Upload a notebook + data
- RBG runs the 3-layer audit automatically
- Rerun notebooks with randomized seeds (does the result hold?)
- Interactive HTML report with code diffs, data profiles, and findings
- Shareable URL for reviewers, editors, and co-authors
- Replaces the "download, install dependencies, rerun" reviewer workflow

### The Market
- 400K+ papers/year in predatory journals with no real peer review
- Legitimate journals can't find reviewers (unpaid, 20+ hours each)
- Computational reproducibility crisis: 70% of researchers can't reproduce others' work
- No existing tool does what RBG does in 2 seconds

### The Thesis
Peer review is a human bottleneck on a mechanical problem.
80% of what a good reviewer checks can be automated.
The remaining 20% (domain expertise, novelty assessment) is what humans should spend their time on.

RBG handles the 80%. Humans handle the 20%.

---

**Built:** April 9, 2026
**By:** John Miller + Claude
**Status:** Working CLI, 3 HTML reports generated, shipped to GitHub
**Next step:** HTML playground with seed randomization and notebook re-execution
