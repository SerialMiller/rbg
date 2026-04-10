"""Parse and profile CSV/tabular datasets for audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ColumnProfile:
    """Statistical profile of a single column."""

    name: str
    dtype: str
    n_total: int
    n_missing: int
    n_unique: int
    # Numeric stats (None for non-numeric)
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    median: Optional[float] = None
    # Flags
    has_negative: bool = False
    n_negative: int = 0
    suspected_floor: Optional[float] = None  # Repeated minimum value
    n_at_floor: int = 0

    @property
    def pct_missing(self) -> float:
        return self.n_missing / self.n_total * 100 if self.n_total > 0 else 0.0

    @property
    def pct_negative(self) -> float:
        return self.n_negative / self.n_total * 100 if self.n_total > 0 else 0.0

    @property
    def pct_at_floor(self) -> float:
        return self.n_at_floor / self.n_total * 100 if self.n_total > 0 else 0.0


@dataclass
class GroupMissingness:
    """Missingness comparison between groups."""

    column: str
    group_col: str
    group_a: str
    group_b: str
    missing_a: int
    total_a: int
    missing_b: int
    total_b: int

    @property
    def pct_missing_a(self) -> float:
        return self.missing_a / self.total_a * 100 if self.total_a > 0 else 0.0

    @property
    def pct_missing_b(self) -> float:
        return self.missing_b / self.total_b * 100 if self.total_b > 0 else 0.0

    @property
    def asymmetry_ratio(self) -> float:
        """Ratio of missingness rates. >1 means A is more missing."""
        rate_a = self.pct_missing_a
        rate_b = self.pct_missing_b
        if rate_b == 0:
            return float("inf") if rate_a > 0 else 1.0
        return rate_a / rate_b

    @property
    def is_suspicious(self) -> bool:
        """Missingness asymmetry > 2x and at least 5% in one group."""
        return (
            self.asymmetry_ratio > 2.0
            and max(self.pct_missing_a, self.pct_missing_b) > 5.0
        )


@dataclass
class DatasetProfile:
    """Full profile of a tabular dataset."""

    path: Path
    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]
    column_names: list[str]
    # Detected structure
    suspected_target: Optional[str] = None
    suspected_group_col: Optional[str] = None
    group_missingness: list[GroupMissingness] = field(default_factory=list)
    # Feature pairs
    raw_log_pairs: list[tuple[str, str]] = field(default_factory=list)
    # Impossible values
    impossible_negatives: list[tuple[str, int]] = field(default_factory=list)

    def get_column(self, name: str) -> Optional[ColumnProfile]:
        for c in self.columns:
            if c.name == name:
                return c
        return None


def _is_numeric_dtype(dtype) -> bool:
    """Check if a dtype is numeric, handling pandas extension types safely."""
    try:
        return np.issubdtype(dtype, np.number)
    except (TypeError, AttributeError):
        # Handle pandas extension types like StringDtype, etc.
        dtype_str = str(dtype).lower()
        return any(t in dtype_str for t in ("int", "float", "decimal"))


def _detect_floor_values(series: pd.Series, threshold: float = 0.10) -> tuple[Optional[float], int]:
    """Detect if a column has floor/ceiling substitution.

    Returns (floor_value, count) if >threshold fraction share the min value.
    """
    if series.isna().all():
        return None, 0
    clean = series.dropna()
    if len(clean) == 0:
        return None, 0
    min_val = clean.min()
    count = (clean == min_val).sum()
    if count / len(clean) > threshold and count > 5:
        return float(min_val), int(count)
    return None, 0


def _detect_raw_log_pairs(columns: list[str]) -> list[tuple[str, str]]:
    """Find columns that appear to be raw + log-transformed pairs."""
    pairs = []
    log_pattern = set()
    for col in columns:
        low = col.lower()
        if low.startswith("log") or low.startswith("ln"):
            # Strip prefix and try to match
            stripped = low.lstrip("log").lstrip("ln").lstrip("_").lstrip(".")
            log_pattern.add((col, stripped))

    for log_col, stripped in log_pattern:
        for raw_col in columns:
            raw_low = raw_col.lower().replace(".", "").replace("_", "")
            stripped_clean = stripped.replace(".", "").replace("_", "")
            if raw_low == stripped_clean and raw_col != log_col:
                pairs.append((raw_col, log_col))
                break

    return pairs


def _detect_group_column(df: pd.DataFrame) -> Optional[str]:
    """Detect the likely case/control or group column."""
    candidates = []
    for col in df.columns:
        try:
            if df[col].dtype == object or str(df[col].dtype) in ("category", "string"):
                nunique = df[col].nunique()
                if 2 <= nunique <= 5:
                    low_vals = {str(v).lower() for v in df[col].unique() if pd.notna(v)}
                    # Score by how "groupy" the values look
                    group_words = {
                        "case", "control", "patient", "healthy",
                        "positive", "negative", "yes", "no",
                        "inpatient", "outpatient", "0", "1",
                    }
                    overlap = len(low_vals & group_words)
                    candidates.append((col, overlap, nunique))
        except Exception:
            continue

    if not candidates:
        return None
    # Prefer columns with group-like values, then fewer unique values
    candidates.sort(key=lambda x: (-x[1], x[2]))
    return candidates[0][0]


def _detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """Detect the likely prediction target / response variable."""
    target_names = {"response", "target", "label", "y", "outcome", "class",
                    "casecontrol", "case_control", "event", "event5"}
    for col in df.columns:
        if col.lower().replace("_", "").replace(".", "") in target_names:
            return col
    # Fall back to binary int columns named suggestively
    for col in df.columns:
        if df[col].nunique() == 2 and col.lower() in {"y", "response", "target"}:
            return col
    return None


INHERENTLY_POSITIVE = {
    "age", "bmi", "body_mass_index", "weight", "height",
    "hdl", "ldl", "tc", "tg", "chol", "trig",
    "creatinine", "creat", "egfr",
    "crp", "hscrp", "saa", "mpo", "ferritin",
    "il6", "il-6",
    "iga", "igg", "igm",
}


def _check_impossible_negatives(df: pd.DataFrame) -> list[tuple[str, int]]:
    """Find columns that should be positive but contain negative values."""
    results = []
    for col in df.columns:
        if not _is_numeric_dtype(df[col].dtype):
            continue
        col_lower = col.lower().replace(".", "").replace("_", "").replace("-", "")
        # Skip log-transformed columns
        if col_lower.startswith("log") or col_lower.startswith("ln"):
            continue
        for pos_name in INHERENTLY_POSITIVE:
            clean = pos_name.replace("_", "").replace("-", "")
            if clean in col_lower:
                try:
                    n_neg = int((df[col] < 0).sum())
                    if n_neg > 0:
                        results.append((col, n_neg))
                except Exception:
                    pass
                break
    return results


def profile_dataset(path: str | Path) -> DatasetProfile:
    """Profile a CSV dataset for audit."""
    path = Path(path)
    df = pd.read_csv(path)

    columns = []
    for col in df.columns:
        n_total = len(df)
        n_missing = int(df[col].isna().sum())
        n_unique = int(df[col].nunique())

        profile = ColumnProfile(
            name=col,
            dtype=str(df[col].dtype),
            n_total=n_total,
            n_missing=n_missing,
            n_unique=n_unique,
        )

        if _is_numeric_dtype(df[col].dtype):
            clean = df[col].dropna()
            if len(clean) > 0:
                try:
                    profile.mean = float(clean.mean())
                    profile.std = float(clean.std())
                    profile.min_val = float(clean.min())
                    profile.max_val = float(clean.max())
                    profile.median = float(clean.median())
                    profile.has_negative = bool((clean < 0).any())
                    profile.n_negative = int((clean < 0).sum())
                    profile.suspected_floor, profile.n_at_floor = _detect_floor_values(clean)
                except Exception:
                    pass

        columns.append(profile)

    # Detect structure
    group_col = _detect_group_column(df)
    target_col = _detect_target_column(df)
    raw_log_pairs = _detect_raw_log_pairs(list(df.columns))
    impossible_negs = _check_impossible_negatives(df)

    # Compute group missingness if group column found
    group_miss = []
    if group_col is not None:
        groups = df[group_col].dropna().unique()
        if len(groups) == 2:
            ga, gb = str(groups[0]), str(groups[1])
            df_a = df[df[group_col] == groups[0]]
            df_b = df[df[group_col] == groups[1]]
            for col_name in df.columns:
                if col_name == group_col:
                    continue
                gm = GroupMissingness(
                    column=col_name,
                    group_col=group_col,
                    group_a=ga,
                    group_b=gb,
                    missing_a=int(df_a[col_name].isna().sum()),
                    total_a=len(df_a),
                    missing_b=int(df_b[col_name].isna().sum()),
                    total_b=len(df_b),
                )
                if gm.is_suspicious:
                    group_miss.append(gm)

    return DatasetProfile(
        path=path,
        n_rows=len(df),
        n_cols=len(df.columns),
        columns=columns,
        column_names=list(df.columns),
        suspected_target=target_col,
        suspected_group_col=group_col,
        group_missingness=group_miss,
        raw_log_pairs=raw_log_pairs,
        impossible_negatives=impossible_negs,
    )
