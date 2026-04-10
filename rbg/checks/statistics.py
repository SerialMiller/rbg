"""Layer 2 — Statistical Integrity.

Checks for:
  - Naive baseline accuracy (majority class)
  - Class imbalance without mitigation
  - Confidence intervals on small test sets
  - ROC curves on hard labels
  - Multiple comparison corrections
  - Feature redundancy (raw + log pairs)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from rbg.parsers.notebook import Notebook, Cell
from rbg.parsers.dataset import DatasetProfile


@dataclass
class Finding:
    """Generic statistical finding."""

    check_name: str
    severity: str  # "critical", "high", "medium", "low"
    summary: str
    detail: str
    cell_index: Optional[int] = None


@dataclass
class StatisticalReport:
    """Full Layer 2 report."""

    findings: list[Finding] = field(default_factory=list)

    @property
    def n_critical(self) -> int:
        return sum(1 for f in self.findings if f.severity == "critical")

    @property
    def n_high(self) -> int:
        return sum(1 for f in self.findings if f.severity == "high")


def _extract_confusion_matrices(nb: Notebook) -> list[tuple[int, list[list[int]]]]:
    """Extract confusion matrices from notebook outputs.

    Returns list of (cell_index, matrix) tuples.
    """
    matrices = []
    cm_pattern = re.compile(
        r"\[\[?\s*(\d+)\s*[,\s]+(\d+)\s*\]?\s*[,\s]*\[?\s*(\d+)\s*[,\s]+(\d+)\s*\]?\]?"
    )

    for cell in nb.code_cells:
        text = cell.output_text + "\n" + cell.source
        for match in cm_pattern.finditer(text):
            try:
                tn, fp, fn, tp = [int(match.group(i)) for i in range(1, 5)]
                matrices.append((cell.index, [[tn, fp], [fn, tp]]))
            except (ValueError, IndexError):
                continue

    return matrices


def _check_naive_baseline(
    matrices: list[tuple[int, list[list[int]]]]
) -> list[Finding]:
    """Compare reported accuracy against naive majority baseline."""
    findings = []

    for cell_idx, cm in matrices:
        tn, fp = cm[0]
        fn, tp = cm[1]
        total = tn + fp + fn + tp
        if total == 0:
            continue

        actual_acc = (tn + tp) / total
        majority_class = max(tn + fp, fn + tp)  # Negatives or positives
        naive_acc = majority_class / total

        # Class balance
        n_positive = fn + tp
        n_negative = tn + fp
        imbalance_ratio = max(n_positive, n_negative) / min(n_positive, n_negative) if min(n_positive, n_negative) > 0 else float("inf")

        gain_over_naive = actual_acc - naive_acc

        if gain_over_naive < 0:
            findings.append(Finding(
                check_name="naive_baseline",
                severity="critical",
                summary=f"Model accuracy ({actual_acc:.1%}) is WORSE than naive baseline ({naive_acc:.1%})",
                detail=(
                    f"Cell [{cell_idx}]: A classifier that always predicts the majority class "
                    f"achieves {naive_acc:.1%}. The model achieves {actual_acc:.1%} — "
                    f"it is destroying information."
                ),
                cell_index=cell_idx,
            ))
        elif gain_over_naive < 0.05 and imbalance_ratio > 3:
            findings.append(Finding(
                check_name="naive_baseline",
                severity="high",
                summary=f"Model barely beats naive baseline: {actual_acc:.1%} vs {naive_acc:.1%} (+{gain_over_naive:.1%})",
                detail=(
                    f"Cell [{cell_idx}]: With {imbalance_ratio:.1f}:1 class imbalance, "
                    f"naive baseline is {naive_acc:.1%}. Model adds only {gain_over_naive:.1%} — "
                    f"headline accuracy is misleading without this context."
                ),
                cell_index=cell_idx,
            ))

        # Sensitivity check
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        if sensitivity < 0.5 and actual_acc > 0.85:
            findings.append(Finding(
                check_name="sensitivity",
                severity="high",
                summary=f"High accuracy ({actual_acc:.1%}) masks low sensitivity ({sensitivity:.1%})",
                detail=(
                    f"Cell [{cell_idx}]: The model correctly identifies only {tp}/{tp+fn} "
                    f"positive cases ({sensitivity:.1%} sensitivity). "
                    f"A clinical deployment would miss {fn} of {tp+fn} cases."
                ),
                cell_index=cell_idx,
            ))

        # Confidence interval width on small test sets
        if n_positive > 0 and n_positive < 30:
            # Wilson interval approximation
            z = 1.96
            n = tp + fn
            p_hat = tp / n if n > 0 else 0
            if n > 0:
                denominator = 1 + z**2 / n
                center = (p_hat + z**2 / (2 * n)) / denominator
                spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator
                ci_width = 2 * spread
                if ci_width > 0.30:
                    findings.append(Finding(
                        check_name="small_test_set",
                        severity="high",
                        summary=f"Sensitivity {sensitivity:.1%} has a 95% CI width of {ci_width:.0%} (n={n} cases)",
                        detail=(
                            f"Cell [{cell_idx}]: With only {n} positive cases in the test set, "
                            f"the sensitivity estimate of {sensitivity:.1%} has a 95% Wilson CI of "
                            f"approximately [{max(0, center-spread):.0%}, {min(1, center+spread):.0%}]. "
                            f"Reporting to decimal precision is false precision."
                        ),
                        cell_index=cell_idx,
                    ))

    return findings


def _check_roc_on_hard_labels(nb: Notebook) -> list[Finding]:
    """Check if roc_curve is called with hard predictions instead of probabilities."""
    findings = []

    for cell in nb.code_cells:
        source = cell.source
        if "roc_curve" not in source:
            continue

        # Look for roc_curve called with a variable that was set by .predict()
        # rather than .predict_proba()
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if "roc_curve" in line:
                # Check if the second argument looks like hard predictions
                # Simple heuristic: look for predict() without _proba in nearby code
                has_predict = bool(re.search(r"\.predict\s*\(", source))
                has_proba = bool(re.search(r"\.predict_proba\s*\(", source))
                has_decision = bool(re.search(r"\.decision_function\s*\(", source))

                if has_predict and not has_proba and not has_decision:
                    findings.append(Finding(
                        check_name="roc_hard_labels",
                        severity="high",
                        summary="ROC curve computed on hard predictions, not probability scores",
                        detail=(
                            f"Cell [{cell.index}]: `roc_curve` is called but only `.predict()` "
                            f"is used (no `.predict_proba()` or `.decision_function()`). "
                            f"A ROC curve on binary labels has exactly one interior operating point — "
                            f"it is not a curve, and the AUC equals balanced accuracy."
                        ),
                        cell_index=cell.index,
                    ))
                break

    return findings


def _check_class_imbalance_handling(nb: Notebook) -> list[Finding]:
    """Check if class imbalance is addressed."""
    findings = []
    all_code = nb.all_code

    # Detect if there's class imbalance
    has_classifiers = bool(re.search(
        r"(RandomForest|AdaBoost|Logistic|MLP|KNeighbors|SVM|SVC).*Classifier",
        all_code
    ))

    if not has_classifiers:
        return findings

    # Check for imbalance handling
    has_class_weight = "class_weight" in all_code
    has_smote = "SMOTE" in all_code or "smote" in all_code
    has_stratify = "stratify" in all_code
    has_balanced_acc = "balanced_accuracy" in all_code
    has_f1 = "f1_score" in all_code
    has_mcc = "matthews_corrcoef" in all_code or "MCC" in all_code

    mitigations = []
    if has_class_weight:
        mitigations.append("class_weight")
    if has_smote:
        mitigations.append("SMOTE")
    if has_stratify:
        mitigations.append("stratified split")
    if has_balanced_acc or has_f1 or has_mcc:
        mitigations.append("balanced metric")

    if not mitigations:
        findings.append(Finding(
            check_name="class_imbalance",
            severity="medium",
            summary="No class imbalance mitigation detected",
            detail=(
                "The notebook uses classifiers but does not appear to employ "
                "class_weight='balanced', SMOTE, stratified splitting, or balanced "
                "evaluation metrics (balanced accuracy, F1, MCC). "
                "If classes are imbalanced, accuracy is misleading."
            ),
        ))

    return findings


def _check_feature_redundancy(dataset: Optional[DatasetProfile]) -> list[Finding]:
    """Check for raw + log-transformed feature pairs."""
    findings = []

    if dataset is None or not dataset.raw_log_pairs:
        return findings

    pairs = dataset.raw_log_pairs
    if len(pairs) > 0:
        pair_strs = [f"`{raw}` + `{log}`" for raw, log in pairs]
        findings.append(Finding(
            check_name="feature_redundancy",
            severity="medium",
            summary=f"{len(pairs)} raw + log-transformed feature pairs detected",
            detail=(
                f"Both raw and log-transformed versions of the same variables are present: "
                f"{', '.join(pair_strs)}. "
                f"If both enter a tree-based model, feature importance is split between them, "
                f"inflating the apparent importance of that biomarker."
            ),
        ))

    return findings


def _check_multiple_comparisons(nb: Notebook) -> list[Finding]:
    """Check for multiple comparisons without correction."""
    findings = []
    all_code = nb.all_code

    # Count t-tests or similar
    n_tests = len(re.findall(r"ttest_ind|mannwhitneyu|chi2_contingency|pearsonr|spearmanr", all_code))

    if n_tests >= 5:
        has_correction = bool(re.search(
            r"bonferroni|holm|fdr|benjamini|multipletests|p_adjust|p\.adjust",
            all_code, re.IGNORECASE
        ))
        if not has_correction:
            findings.append(Finding(
                check_name="multiple_comparisons",
                severity="medium",
                summary=f"{n_tests} statistical tests without multiple comparison correction",
                detail=(
                    f"The notebook performs {n_tests} statistical tests but does not apply "
                    f"Bonferroni, Holm, or FDR correction. At a nominal alpha of 0.05, "
                    f"the family-wise error rate is approximately "
                    f"{1 - (0.95 ** n_tests):.0%} — "
                    f"the probability of at least one false positive."
                ),
            ))

    return findings


def _check_train_accuracy(nb: Notebook) -> list[Finding]:
    """Flag 100% training accuracy as overfitting signal."""
    findings = []

    for cell in nb.code_cells:
        text = cell.output_text
        # Look for 1.0 or 100% training accuracy patterns
        if re.search(r"train.*(?:accuracy|score).*(?:1\.0|100)", text, re.IGNORECASE):
            findings.append(Finding(
                check_name="overfitting",
                severity="high",
                summary="100% training accuracy — classic overfitting signal",
                detail=(
                    f"Cell [{cell.index}]: Training accuracy of 100% typically indicates "
                    f"the model has memorized the training data rather than learning "
                    f"generalizable patterns. This is especially concerning with "
                    f"tree-based models with no depth constraint."
                ),
                cell_index=cell.index,
            ))

    return findings


def check_statistics(
    nb: Notebook, dataset: Optional[DatasetProfile] = None
) -> StatisticalReport:
    """Run all Layer 2 checks."""
    matrices = _extract_confusion_matrices(nb)
    findings = []
    findings.extend(_check_naive_baseline(matrices))
    findings.extend(_check_roc_on_hard_labels(nb))
    findings.extend(_check_class_imbalance_handling(nb))
    findings.extend(_check_feature_redundancy(dataset))
    findings.extend(_check_multiple_comparisons(nb))
    findings.extend(_check_train_accuracy(nb))
    return StatisticalReport(findings=findings)
