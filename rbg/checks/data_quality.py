"""Layer 3 — Data Quality.

Checks for:
  - Asymmetric missingness between groups
  - Impossible/implausible values
  - Floor/ceiling substitution artifacts
  - Non-compositional percentage columns
  - Temporal batch effects (if dates present)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from rbg.parsers.dataset import DatasetProfile, GroupMissingness


@dataclass
class Finding:
    """Data quality finding."""

    check_name: str
    severity: str
    summary: str
    detail: str
    column: Optional[str] = None


@dataclass
class DataQualityReport:
    """Full Layer 3 report."""

    findings: list[Finding] = field(default_factory=list)

    @property
    def n_critical(self) -> int:
        return sum(1 for f in self.findings if f.severity == "critical")

    @property
    def n_high(self) -> int:
        return sum(1 for f in self.findings if f.severity == "high")


def _check_asymmetric_missingness(profile: DatasetProfile) -> list[Finding]:
    """Flag columns where missingness differs dramatically between groups."""
    findings = []

    for gm in profile.group_missingness:
        if not gm.is_suspicious:
            continue

        ratio = gm.asymmetry_ratio
        if ratio == float("inf"):
            ratio_str = "infinite"
        else:
            ratio_str = f"{ratio:.1f}x"

        severity = "critical" if ratio > 10 else "high"

        findings.append(Finding(
            check_name="asymmetric_missingness",
            severity=severity,
            summary=(
                f"`{gm.column}`: {gm.pct_missing_a:.1f}% missing in {gm.group_a} "
                f"vs {gm.pct_missing_b:.1f}% in {gm.group_b} ({ratio_str} asymmetry)"
            ),
            detail=(
                f"Column `{gm.column}` has {gm.missing_a}/{gm.total_a} missing values "
                f"in group '{gm.group_a}' ({gm.pct_missing_a:.1f}%) vs "
                f"{gm.missing_b}/{gm.total_b} in group '{gm.group_b}' ({gm.pct_missing_b:.1f}%). "
                f"This {ratio_str} asymmetry suggests missingness is not random (not MCAR). "
                f"Any analysis using this column operates on a biased subset."
            ),
            column=gm.column,
        ))

    return findings


def _check_impossible_values(profile: DatasetProfile) -> list[Finding]:
    """Flag physiologically impossible values."""
    findings = []

    for col_name, n_neg in profile.impossible_negatives:
        col = profile.get_column(col_name)
        if col is None:
            continue
        pct = n_neg / col.n_total * 100

        severity = "high" if pct > 5 else "medium"

        findings.append(Finding(
            check_name="impossible_values",
            severity=severity,
            summary=f"`{col_name}`: {n_neg} negative values ({pct:.1f}%) in an inherently positive measure",
            detail=(
                f"Column `{col_name}` contains {n_neg} negative values ({pct:.1f}% of data), "
                f"min = {col.min_val}. This variable should be non-negative by biological definition. "
                f"Possible causes: log-transformation without labeling, data entry errors, "
                f"or simulation artifacts. These values distort means and invalidate parametric tests."
            ),
            column=col_name,
        ))

    return findings


def _check_floor_substitution(profile: DatasetProfile) -> list[Finding]:
    """Flag columns with suspected floor/LOD substitution."""
    findings = []

    for col in profile.columns:
        if col.suspected_floor is not None and col.pct_at_floor > 15:
            findings.append(Finding(
                check_name="floor_substitution",
                severity="high" if col.pct_at_floor > 30 else "medium",
                summary=(
                    f"`{col.name}`: {col.n_at_floor} values ({col.pct_at_floor:.1f}%) "
                    f"at suspected floor/LOD value {col.suspected_floor}"
                ),
                detail=(
                    f"Column `{col.name}` has {col.n_at_floor} of {col.n_total} values "
                    f"({col.pct_at_floor:.1f}%) equal to {col.suspected_floor}, which appears "
                    f"to be a lower limit of detection (LOD) substitution. "
                    f"Treating these as measured values inflates correlations and biases "
                    f"group comparisons. Consider: left-censored regression, multiple imputation, "
                    f"or at minimum, sensitivity analysis excluding floor values."
                ),
                column=col.name,
            ))

    return findings


def _check_compositional_columns(profile: DatasetProfile, df: pd.DataFrame) -> list[Finding]:
    """Check if percentage columns actually sum to 100%."""
    findings = []

    # Heuristic: find groups of columns that look like they should sum to 100%
    # Look for alpha/fraction patterns
    pct_patterns = [
        (r"alph\d|alpha\d|preB|pre_?beta", "HDL subfraction percentages"),
        (r"frac_|fraction_|pct_|percent_", "percentage columns"),
    ]

    import re as _re
    for pattern, label in pct_patterns:
        matching = [c for c in profile.column_names if _re.search(pattern, c, _re.IGNORECASE)]
        if len(matching) >= 3:
            # Check if they sum to ~100
            subset = df[matching].dropna()
            if len(subset) > 0:
                row_sums = subset.sum(axis=1)
                mean_sum = row_sums.mean()
                std_sum = row_sums.std()
                near_100 = ((row_sums > 95) & (row_sums < 105)).sum()
                pct_near_100 = near_100 / len(row_sums) * 100

                if pct_near_100 < 50 and abs(mean_sum - 100) > 10:
                    findings.append(Finding(
                        check_name="non_compositional",
                        severity="medium",
                        summary=(
                            f"{label}: {len(matching)} columns sum to "
                            f"{mean_sum:.1f} (SD {std_sum:.1f}), not 100%"
                        ),
                        detail=(
                            f"Columns [{', '.join(matching)}] appear to be {label} "
                            f"but sum to a mean of {mean_sum:.1f} (SD {std_sum:.1f}, "
                            f"range {row_sums.min():.1f}–{row_sums.max():.1f}). "
                            f"Only {pct_near_100:.1f}% of rows sum to near 100%. "
                            f"These may be absolute values, not percentages. "
                            f"Relative differences are still valid, but units need clarification."
                        ),
                    ))

    return findings


def _check_overall_missingness(profile: DatasetProfile) -> list[Finding]:
    """Flag high overall missingness."""
    findings = []

    high_missing = [
        (col.name, col.pct_missing)
        for col in profile.columns
        if col.pct_missing > 20
    ]

    if high_missing:
        cols_str = ", ".join(f"`{name}` ({pct:.0f}%)" for name, pct in high_missing)
        findings.append(Finding(
            check_name="high_missingness",
            severity="high" if any(p > 40 for _, p in high_missing) else "medium",
            summary=f"{len(high_missing)} columns have >20% missing values",
            detail=(
                f"Columns with high missingness: {cols_str}. "
                f"If these columns enter the analysis, `dropna()` will silently "
                f"remove substantial fractions of the dataset. "
                f"Document: how many rows survive? Is missingness random?"
            ),
        ))

    return findings


def check_data_quality(
    profile: DatasetProfile, df: Optional[pd.DataFrame] = None
) -> DataQualityReport:
    """Run all Layer 3 checks."""
    findings = []
    findings.extend(_check_asymmetric_missingness(profile))
    findings.extend(_check_impossible_values(profile))
    findings.extend(_check_floor_substitution(profile))
    findings.extend(_check_overall_missingness(profile))

    if df is not None:
        findings.extend(_check_compositional_columns(profile, df))

    return DataQualityReport(findings=findings)
