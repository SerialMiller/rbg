"""Layer 1 — Reproducibility Autopsy.

Checks for:
  - Missing random seeds in stochastic operations
  - Convergence warnings in outputs
  - Execution order anomalies
  - Unexecuted cells with code
  - Kernel state pollution (variable overwrites)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rbg.parsers.notebook import Notebook, Cell


@dataclass
class SeedFinding:
    """A stochastic operation missing a random seed."""

    cell_index: int
    line_number: int
    code_line: str
    function_name: str
    severity: str = "high"  # high for model/split, medium for sampling

    @property
    def summary(self) -> str:
        return (
            f"Cell [{self.cell_index}] line {self.line_number}: "
            f"`{self.function_name}` called without `random_state`"
        )


@dataclass
class ConvergenceFinding:
    """A model that did not converge."""

    cell_index: int
    warning_text: str
    severity: str = "high"

    @property
    def summary(self) -> str:
        return (
            f"Cell [{self.cell_index}]: Model did not converge — "
            f"results from an unfinished training run"
        )


@dataclass
class ExecutionOrderFinding:
    """Execution order anomaly."""

    finding_type: str  # "out_of_order", "gap", "unexecuted"
    cell_index: int
    detail: str
    severity: str = "medium"

    @property
    def summary(self) -> str:
        return f"Cell [{self.cell_index}]: {self.detail}"


@dataclass
class OverwriteFinding:
    """Variable overwritten between cells, risking kernel state pollution."""

    variable: str
    first_cell: int
    second_cell: int
    severity: str = "high"

    @property
    def summary(self) -> str:
        return (
            f"Variable `{self.variable}` assigned in cell [{self.first_cell}] "
            f"and overwritten in cell [{self.second_cell}] — "
            f"if cells ran out of order, later reads may use stale data"
        )


@dataclass
class ReproducibilityReport:
    """Full Layer 1 report."""

    seed_findings: list[SeedFinding] = field(default_factory=list)
    convergence_findings: list[ConvergenceFinding] = field(default_factory=list)
    execution_findings: list[ExecutionOrderFinding] = field(default_factory=list)
    overwrite_findings: list[OverwriteFinding] = field(default_factory=list)

    @property
    def all_findings(self) -> list:
        return (
            self.seed_findings
            + self.convergence_findings
            + self.execution_findings
            + self.overwrite_findings
        )

    @property
    def n_critical(self) -> int:
        return sum(1 for f in self.all_findings if f.severity == "high")

    @property
    def n_warnings(self) -> int:
        return sum(1 for f in self.all_findings if f.severity == "medium")


# Functions/classes that need random_state
STOCHASTIC_PATTERNS = [
    # (regex pattern, function name, severity)
    (r"train_test_split\s*\(", "train_test_split", "high"),
    (r"RandomForestClassifier\s*\(", "RandomForestClassifier", "high"),
    (r"RandomForestRegressor\s*\(", "RandomForestRegressor", "high"),
    (r"GradientBoostingClassifier\s*\(", "GradientBoostingClassifier", "high"),
    (r"GradientBoostingRegressor\s*\(", "GradientBoostingRegressor", "high"),
    (r"AdaBoostClassifier\s*\(", "AdaBoostClassifier", "high"),
    (r"AdaBoostRegressor\s*\(", "AdaBoostRegressor", "high"),
    (r"DecisionTreeClassifier\s*\(", "DecisionTreeClassifier", "high"),
    (r"DecisionTreeRegressor\s*\(", "DecisionTreeRegressor", "high"),
    (r"ExtraTreesClassifier\s*\(", "ExtraTreesClassifier", "high"),
    (r"KFold\s*\(", "KFold", "high"),
    (r"StratifiedKFold\s*\(", "StratifiedKFold", "high"),
    (r"ShuffleSplit\s*\(", "ShuffleSplit", "high"),
    (r"XGBClassifier\s*\(", "XGBClassifier", "high"),
    (r"XGBRegressor\s*\(", "XGBRegressor", "high"),
    (r"LogisticRegression\s*\(", "LogisticRegression", "medium"),
    (r"MLPClassifier\s*\(", "MLPClassifier", "high"),
    (r"MLPRegressor\s*\(", "MLPRegressor", "high"),
    (r"SGDClassifier\s*\(", "SGDClassifier", "medium"),
    (r"KMeans\s*\(", "KMeans", "medium"),
    (r"SMOTE\s*\(", "SMOTE", "high"),
    (r"\.sample\s*\(", "DataFrame.sample", "medium"),
    (r"np\.random\.", "numpy.random (global)", "medium"),
]


def _check_seeds(nb: Notebook) -> list[SeedFinding]:
    """Find stochastic operations without random_state."""
    findings = []

    # Check for global seed setting
    all_code = nb.all_code
    has_global_seed = bool(
        re.search(r"np\.random\.seed\s*\(", all_code)
        or re.search(r"random\.seed\s*\(", all_code)
        or re.search(r"torch\.manual_seed\s*\(", all_code)
    )

    for cell in nb.code_cells:
        for line_no, line in enumerate(cell.source.split("\n"), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("'''"):
                continue

            for pattern, func_name, severity in STOCHASTIC_PATTERNS:
                if re.search(pattern, line):
                    # Check if random_state is set in the same call
                    # Look for the full call (might span multiple lines)
                    # Simple heuristic: check this line and next few
                    call_text = line
                    # Extend to closing paren if not found
                    lines = cell.source.split("\n")
                    idx = line_no - 1
                    paren_depth = 0
                    for i in range(idx, min(idx + 10, len(lines))):
                        call_text += " " + lines[i]
                        paren_depth += lines[i].count("(") - lines[i].count(")")
                        if paren_depth <= 0:
                            break

                    has_seed = bool(
                        re.search(r"random_state\s*=", call_text)
                        or re.search(r"seed\s*=", call_text)
                    )

                    # Global numpy seed partially mitigates
                    if not has_seed and not (has_global_seed and func_name == "numpy.random (global)"):
                        findings.append(SeedFinding(
                            cell_index=cell.index,
                            line_number=line_no,
                            code_line=stripped,
                            function_name=func_name,
                            severity=severity,
                        ))
                    break  # Don't double-count same line

    return findings


def _check_convergence(nb: Notebook) -> list[ConvergenceFinding]:
    """Find convergence warnings in outputs."""
    findings = []
    for cell in nb.code_cells:
        for warning in cell.convergence_warnings:
            findings.append(ConvergenceFinding(
                cell_index=cell.index,
                warning_text=warning,
            ))
    return findings


def _check_execution_order(nb: Notebook) -> list[ExecutionOrderFinding]:
    """Check for execution order anomalies."""
    findings = []

    # Out-of-order execution
    if nb.has_out_of_order_execution:
        counts = [(c.index, c.execution_count) for c in nb.code_cells
                   if c.execution_count is not None]
        for i in range(1, len(counts)):
            idx, curr = counts[i]
            prev_idx, prev = counts[i - 1]
            if curr < prev:
                findings.append(ExecutionOrderFinding(
                    finding_type="out_of_order",
                    cell_index=idx,
                    detail=(
                        f"Execution count [{curr}] follows [{prev}] — "
                        f"cell was run before the previous cell"
                    ),
                ))

    # Execution gaps
    for idx, prev, curr in nb.execution_gaps:
        gap = curr - prev
        if gap > 5:  # Only flag large gaps
            findings.append(ExecutionOrderFinding(
                finding_type="gap",
                cell_index=idx,
                detail=(
                    f"Execution count gap: [{prev}] to [{curr}] "
                    f"({gap} hidden executions between cells)"
                ),
                severity="medium",
            ))

    # Unexecuted code cells
    for cell in nb.unexecuted_cells:
        if cell.source.strip():  # Skip empty cells
            findings.append(ExecutionOrderFinding(
                finding_type="unexecuted",
                cell_index=cell.index,
                detail="Code cell was never executed in this session",
                severity="medium" if len(cell.source) < 50 else "high",
            ))

    return findings


def _check_variable_overwrites(nb: Notebook) -> list[OverwriteFinding]:
    """Detect variables assigned in multiple cells (state pollution risk)."""
    # Track assignments: variable -> list of cell indices
    assignments: dict[str, list[int]] = {}

    # Important ML variable patterns
    important_vars = re.compile(
        r"^(X_train|X_test|y_train|y_test|model|clf|rf|"
        r"node_purity|feature_importance|importance|"
        r"RFmod|RFmod\d|predictions?|pred|y_pred|"
        r"accuracy|score|cm|conf_matrix|results?)\s*="
    )

    for cell in nb.code_cells:
        for line in cell.source.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            match = important_vars.match(stripped)
            if match:
                var_name = stripped.split("=")[0].strip()
                if var_name not in assignments:
                    assignments[var_name] = []
                assignments[var_name].append(cell.index)

    findings = []
    for var, cells in assignments.items():
        if len(cells) > 1:
            for i in range(1, len(cells)):
                findings.append(OverwriteFinding(
                    variable=var,
                    first_cell=cells[0],
                    second_cell=cells[i],
                ))

    return findings


def check_reproducibility(nb: Notebook) -> ReproducibilityReport:
    """Run all Layer 1 checks."""
    return ReproducibilityReport(
        seed_findings=_check_seeds(nb),
        convergence_findings=_check_convergence(nb),
        execution_findings=_check_execution_order(nb),
        overwrite_findings=_check_variable_overwrites(nb),
    )
