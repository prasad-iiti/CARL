# ============================================================
# CARL - Metrics
# ============================================================

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score as sk_f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix as sk_confusion_matrix,
    multilabel_confusion_matrix,
)

# ============================================================
# Constants
# ============================================================

EPS = 1e-12


# ============================================================
# Conversion Utilities
# ============================================================

def _to_numpy(x):
    """
    Convert torch tensors, lists, tuples, or NumPy arrays
    to a NumPy array.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def _get_pred_labels(y_pred):
    """
    Convert predictions to class labels.

    Supported inputs
    ----------------
    [N]     : predicted class indices
    [N, C]  : logits or probabilities
    """
    y_pred = _to_numpy(y_pred)

    if y_pred.ndim == 1:
        return y_pred.astype(int)

    if y_pred.ndim != 2:
        raise ValueError(
            "Predictions must have shape [N] or [N, C]."
        )

    return np.argmax(y_pred, axis=1)


def _get_probabilities(y_pred):
    """
    Convert logits/probabilities to a probability matrix.

    Input
    -----
    y_pred : [N, C]

    If values already represent probabilities, they are returned
    unchanged. Otherwise, numerically stable softmax is applied.
    """
    y_pred = _to_numpy(y_pred)

    if y_pred.ndim != 2:
        raise ValueError(
            "Probability/logit matrix expected with shape [N, C]."
        )

    if y_pred.shape[0] == 0:
        raise ValueError(
            "Prediction array is empty."
        )

    if not np.all(np.isfinite(y_pred)):
        raise ValueError(
            "Predictions contain NaN or infinite values."
        )

    row_sums = np.sum(
        y_pred,
        axis=1,
    )

    is_probability = (
        np.all(y_pred >= 0.0)
        and
        np.all(y_pred <= 1.0)
        and
        np.allclose(
            row_sums,
            1.0,
            atol=1e-5,
        )
    )

    if is_probability:
        return y_pred

    # Stable softmax
    logits = (
        y_pred
        - np.max(
            y_pred,
            axis=1,
            keepdims=True,
        )
    )

    exp_scores = np.exp(logits)

    probabilities = (
        exp_scores
        /
        np.sum(
            exp_scores,
            axis=1,
            keepdims=True,
        )
    )

    return probabilities


# ============================================================
# Input Validation
# ============================================================

def _validate_inputs(
    y_pred,
    y_true,
    group_labels=None,
):
    """
    Validate prediction, target, and optional group arrays.
    """
    y_true = _to_numpy(y_true)
    y_pred_np = _to_numpy(y_pred)

    if y_true.ndim != 1:
        y_true = y_true.reshape(-1)

    if len(y_true) == 0:
        raise ValueError(
            "Ground-truth labels are empty."
        )

    if y_pred_np.ndim == 0:
        raise ValueError(
            "Predictions are empty."
        )

    if y_pred_np.shape[0] != len(y_true):
        raise ValueError(
            "Prediction and label sizes do not match."
        )

    if group_labels is not None:

        group_labels = _to_numpy(
            group_labels
        )

        if len(group_labels) != len(y_true):
            raise ValueError(
                "Group labels and ground-truth labels "
                "must have the same length."
            )


# ============================================================
# Basic Classification Metrics
# ============================================================

def accuracy(
    y_pred,
    y_true,
):
    """
    Standard classification accuracy.
    """
    _validate_inputs(
        y_pred,
        y_true,
    )

    return float(
        accuracy_score(
            _to_numpy(y_true),
            _get_pred_labels(y_pred),
        )
    )


def balanced_accuracy(
    y_pred,
    y_true,
):
    """
    Balanced accuracy.

    This is CARL's primary evaluation metric.
    """
    _validate_inputs(
        y_pred,
        y_true,
    )

    return float(
        balanced_accuracy_score(
            _to_numpy(y_true),
            _get_pred_labels(y_pred),
        )
    )


def precision(
    y_pred,
    y_true,
    average="macro",
):
    """
    Precision.

    Default:
        macro precision
    """
    _validate_inputs(
        y_pred,
        y_true,
    )

    return float(
        precision_score(
            _to_numpy(y_true),
            _get_pred_labels(y_pred),
            average=average,
            zero_division=0,
        )
    )


def precision_macro(
    y_pred,
    y_true,
):
    """
    Macro precision.

    CARL config name:
        precision_macro
    """
    return precision(
        y_pred,
        y_true,
        average="macro",
    )


def recall(
    y_pred,
    y_true,
    average="macro",
):
    """
    Recall.

    Default:
        macro recall
    """
    _validate_inputs(
        y_pred,
        y_true,
    )

    return float(
        recall_score(
            _to_numpy(y_true),
            _get_pred_labels(y_pred),
            average=average,
            zero_division=0,
        )
    )


def recall_macro(
    y_pred,
    y_true,
):
    """
    Macro recall.

    CARL config name:
        recall_macro
    """
    return recall(
        y_pred,
        y_true,
        average="macro",
    )


def f1_score(
    y_pred,
    y_true,
    average="macro",
):
    """
    F1 score.

    Default:
        macro F1
    """
    _validate_inputs(
        y_pred,
        y_true,
    )

    return float(
        sk_f1_score(
            _to_numpy(y_true),
            _get_pred_labels(y_pred),
            average=average,
            zero_division=0,
        )
    )


def macro_f1(
    y_pred,
    y_true,
):
    """
    Macro F1.

    CARL config name:
        macro_f1
    """
    return f1_score(
        y_pred,
        y_true,
        average="macro",
    )


def sensitivity(
    y_pred,
    y_true,
):
    """
    Macro sensitivity.

    Sensitivity is equivalent to macro recall
    in the multiclass one-vs-rest setting.
    """
    return recall_macro(
        y_pred,
        y_true,
    )


# ============================================================
# Specificity
# ============================================================

def specificity(
    y_pred,
    y_true,
):
    """
    Macro-averaged multiclass specificity.

    Uses one-vs-rest confusion matrices.
    """
    _validate_inputs(
        y_pred,
        y_true,
    )

    predicted = _get_pred_labels(
        y_pred
    )

    true = _to_numpy(
        y_true
    )

    matrices = multilabel_confusion_matrix(
        true,
        predicted,
    )

    specificities = []

    for cm in matrices:

        tn, fp, fn, tp = cm.ravel()

        denominator = tn + fp

        if denominator == 0:
            continue

        specificities.append(
            tn / denominator
        )

    if not specificities:
        return 0.0

    return float(
        np.mean(
            specificities
        )
    )


# ============================================================
# ROC-AUC
# ============================================================

def roc_auc(
    y_pred,
    y_true,
):
    """
    Macro ROC-AUC.

    Supports:
        binary classification
        multiclass classification

    Multiclass:
        One-vs-rest
        Macro averaging

    Returns None when ROC-AUC cannot be computed because
    the ground truth does not contain sufficient classes.
    """
    _validate_inputs(
        y_pred,
        y_true,
    )

    probabilities = _get_probabilities(
        y_pred
    )

    true = _to_numpy(
        y_true
    )

    unique_classes = np.unique(
        true
    )

    try:

        if len(unique_classes) == 2:

            return float(
                roc_auc_score(
                    true,
                    probabilities[:, 1],
                )
            )

        # Important:
        # sklearn requires the number of probability columns
        # to match the labels/classes being evaluated.
        #
        # For normal CARL test sets containing all classes,
        # this is the desired behavior.

        return float(
            roc_auc_score(
                true,
                probabilities,
                multi_class="ovr",
                average="macro",
            )
        )

    except ValueError:
        return None


# ============================================================
# PR-AUC
# ============================================================

def pr_auc(
    y_pred,
    y_true,
):
    """
    Macro Precision-Recall AUC.

    Supports binary and multiclass classification.
    """
    _validate_inputs(
        y_pred,
        y_true,
    )

    probabilities = _get_probabilities(
        y_pred
    )

    true = _to_numpy(
        y_true
    )

    try:

        if len(np.unique(true)) == 2:

            return float(
                average_precision_score(
                    true,
                    probabilities[:, 1],
                )
            )

        return float(
            average_precision_score(
                true,
                probabilities,
                average="macro",
            )
        )

    except ValueError:
        return None


# ============================================================
# Confusion Matrix
# ============================================================

def confusion_matrix(
    y_pred,
    y_true,
):
    """
    Return the multiclass confusion matrix.
    """
    _validate_inputs(
        y_pred,
        y_true,
    )

    return sk_confusion_matrix(
        _to_numpy(y_true),
        _get_pred_labels(y_pred),
    )


def normalized_confusion_matrix(
    y_pred,
    y_true,
):
    """
    Return row-normalized confusion matrix.

    Each row represents a true class.
    """
    _validate_inputs(
        y_pred,
        y_true,
    )

    return sk_confusion_matrix(
        _to_numpy(y_true),
        _get_pred_labels(y_pred),
        normalize="true",
    )


# ============================================================
# Classification Summary
# ============================================================

def classification_summary(
    y_pred,
    y_true,
):
    """
    Compute CARL classification metrics.
    """
    return {
        "accuracy": accuracy(
            y_pred,
            y_true,
        ),

        "balanced_accuracy": balanced_accuracy(
            y_pred,
            y_true,
        ),

        "precision_macro": precision_macro(
            y_pred,
            y_true,
        ),

        "recall_macro": recall_macro(
            y_pred,
            y_true,
        ),

        "macro_f1": macro_f1(
            y_pred,
            y_true,
        ),

        "sensitivity": sensitivity(
            y_pred,
            y_true,
        ),

        "specificity": specificity(
            y_pred,
            y_true,
        ),

        "roc_auc": roc_auc(
            y_pred,
            y_true,
        ),

        "pr_auc": pr_auc(
            y_pred,
            y_true,
        ),

        "confusion_matrix": confusion_matrix(
            y_pred,
            y_true,
        ).tolist(),

        "normalized_confusion_matrix":
            normalized_confusion_matrix(
                y_pred,
                y_true,
            ).tolist(),
    }


# ============================================================
# Fairness Metrics
# ============================================================

def subgroup_accuracy(
    y_pred,
    y_true,
    group_labels,
):
    """
    Accuracy for each sensitive subgroup.
    """
    _validate_inputs(
        y_pred,
        y_true,
        group_labels,
    )

    predicted = _get_pred_labels(
        y_pred
    )

    true = _to_numpy(
        y_true
    )

    groups = _to_numpy(
        group_labels
    )

    results = {}

    for group in np.unique(groups):

        mask = (
            groups == group
        )

        if not np.any(mask):
            continue

        results[str(group)] = float(
            np.mean(
                predicted[mask]
                ==
                true[mask]
            )
        )

    return results


def worst_group_accuracy(
    y_pred,
    y_true,
    group_labels,
):
    """
    Minimum subgroup accuracy.
    """
    results = subgroup_accuracy(
        y_pred,
        y_true,
        group_labels,
    )

    if not results:
        return 0.0

    return float(
        min(results.values())
    )


def bias_risk_score(
    y_pred,
    y_true,
    group_labels,
):
    """
    Difference between best and worst subgroup accuracy.

    Lower is better.
    """
    results = subgroup_accuracy(
        y_pred,
        y_true,
        group_labels,
    )

    if not results:
        return 0.0

    return float(
        max(results.values())
        -
        min(results.values())
    )


def equalized_odds_difference(
    y_pred,
    y_true,
    group_labels,
):
    """
    Multiclass Equalized Odds Difference.

    Computes the maximum gap across:
        TPR
        FPR

    across all classes and demographic/sensitive groups.

    Lower is better.
    """
    _validate_inputs(
        y_pred,
        y_true,
        group_labels,
    )

    predicted = _get_pred_labels(
        y_pred
    )

    true = _to_numpy(
        y_true
    )

    groups = _to_numpy(
        group_labels
    )

    classes = np.unique(
        true
    )

    unique_groups = np.unique(
        groups
    )

    max_gap = 0.0

    for cls in classes:

        tprs = []
        fprs = []

        for group in unique_groups:

            mask = (
                groups == group
            )

            if not np.any(mask):
                continue

            yt = (
                true[mask] == cls
            ).astype(int)

            yp = (
                predicted[mask] == cls
            ).astype(int)

            tp = np.sum(
                (yt == 1)
                &
                (yp == 1)
            )

            fn = np.sum(
                (yt == 1)
                &
                (yp == 0)
            )

            fp = np.sum(
                (yt == 0)
                &
                (yp == 1)
            )

            tn = np.sum(
                (yt == 0)
                &
                (yp == 0)
            )

            tpr_denominator = (
                tp + fn
            )

            fpr_denominator = (
                fp + tn
            )

            if tpr_denominator > 0:
                tprs.append(
                    tp / tpr_denominator
                )

            if fpr_denominator > 0:
                fprs.append(
                    fp / fpr_denominator
                )

        if len(tprs) > 1:

            max_gap = max(
                max_gap,
                max(tprs) - min(tprs),
            )

        if len(fprs) > 1:

            max_gap = max(
                max_gap,
                max(fprs) - min(fprs),
            )

    return float(
        max_gap
    )


# ============================================================
# Fairness Summary
# ============================================================

def fairness_summary(
    y_pred,
    y_true,
    metadata,
):
    """
    Compute fairness metrics for metadata attributes.

    CARL currently uses:
        age
        sex
        location

    Non-sensitive identifiers are excluded.
    """
    results = {}

    excluded = {
        "image_name",
        "image_id",
        "patient_id",
        "lesion_id",
        "path",
        "filename",
        "label",
        "binary_label",
    }

    for attribute, groups in metadata.items():

        if attribute in excluded:
            continue

        if groups is None:
            continue

        results[attribute] = {
            "subgroup_accuracy":
                subgroup_accuracy(
                    y_pred,
                    y_true,
                    groups,
                ),

            "worst_group_accuracy":
                worst_group_accuracy(
                    y_pred,
                    y_true,
                    groups,
                ),

            "bias_risk_score":
                bias_risk_score(
                    y_pred,
                    y_true,
                    groups,
                ),

            "equalized_odds_difference":
                equalized_odds_difference(
                    y_pred,
                    y_true,
                    groups,
                ),
        }

    return results


# ============================================================
# Uncertainty Metrics
# ============================================================

def predictive_entropy(
    y_pred,
):
    """
    Predictive entropy for every sample.

    Higher entropy indicates greater uncertainty.
    """
    probabilities = _get_probabilities(
        y_pred
    )

    probabilities = np.clip(
        probabilities,
        EPS,
        1.0,
    )

    entropy = -np.sum(
        probabilities
        *
        np.log(probabilities),
        axis=1,
    )

    return entropy


def maximum_softmax_probability(
    y_pred,
):
    """
    Maximum Softmax Probability (MSP).

    Lower MSP indicates greater uncertainty.
    """
    probabilities = _get_probabilities(
        y_pred
    )

    return np.max(
        probabilities,
        axis=1,
    )


def expected_calibration_error(
    y_pred,
    y_true,
    n_bins=15,
):
    """
    Expected Calibration Error (ECE).
    """
    _validate_inputs(
        y_pred,
        y_true,
    )

    probabilities = _get_probabilities(
        y_pred
    )

    true = _to_numpy(
        y_true
    )

    confidences = np.max(
        probabilities,
        axis=1,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    correctness = (
        predictions == true
    )

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0

    for i in range(n_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:

            mask = (
                (confidences >= lower)
                &
                (confidences <= upper)
            )

        else:

            mask = (
                (confidences >= lower)
                &
                (confidences < upper)
            )

        count = np.sum(
            mask
        )

        if count == 0:
            continue

        accuracy_bin = np.mean(
            correctness[mask]
        )

        confidence_bin = np.mean(
            confidences[mask]
        )

        ece += (
            count
            /
            (len(confidences) + EPS)
        ) * abs(
            accuracy_bin
            -
            confidence_bin
        )

    return float(
        ece
    )


# ============================================================
# Uncertainty Summary
# ============================================================

def uncertainty_summary(
    y_pred,
    y_true,
    n_bins=15,
):
    """
    CARL uncertainty summary.
    """
    entropy = predictive_entropy(
        y_pred
    )

    msp = maximum_softmax_probability(
        y_pred
    )

    return {
        "expected_calibration_error":
            expected_calibration_error(
                y_pred,
                y_true,
                n_bins=n_bins,
            ),

        "mean_entropy":
            float(
                np.mean(entropy)
            ),

        "std_entropy":
            float(
                np.std(entropy)
            ),

        "min_entropy":
            float(
                np.min(entropy)
            ),

        "max_entropy":
            float(
                np.max(entropy)
            ),

        "mean_msp":
            float(
                np.mean(msp)
            ),

        "std_msp":
            float(
                np.std(msp)
            ),
    }


# ============================================================
# Unified CARL Evaluation
# ============================================================

def evaluate_all(
    y_pred,
    y_true,
    metadata=None,
):
    """
    Unified CARL evaluation.

    Returns:
        classification metrics
        uncertainty metrics
        optional fairness metrics
    """
    results = {}

    results.update(
        classification_summary(
            y_pred,
            y_true,
        )
    )

    results.update(
        uncertainty_summary(
            y_pred,
            y_true,
        )
    )

    if metadata is not None:

        results["fairness"] = (
            fairness_summary(
                y_pred,
                y_true,
                metadata,
            )
        )

    return results


# ============================================================
# CARL Metric Selection
# ============================================================

def get_primary_metric(
    y_pred,
    y_true,
):
    """
    Return CARL's primary metric.

    CARL configuration:
        primary: balanced_accuracy
    """
    return balanced_accuracy(
        y_pred,
        y_true,
    )


def get_configured_metrics(
    y_pred,
    y_true,
):
    """
    Return the metrics explicitly configured in CARL.
    """
    return {
        "balanced_accuracy":
            balanced_accuracy(
                y_pred,
                y_true,
            ),

        "macro_f1":
            macro_f1(
                y_pred,
                y_true,
            ),

        "roc_auc":
            roc_auc(
                y_pred,
                y_true,
            ),

        "precision_macro":
            precision_macro(
                y_pred,
                y_true,
            ),

        "recall_macro":
            recall_macro(
                y_pred,
                y_true,
            ),
    }


# ============================================================
# Public API
# ============================================================

__all__ = [

    # Basic classification
    "accuracy",
    "balanced_accuracy",
    "precision",
    "precision_macro",
    "recall",
    "recall_macro",
    "f1_score",
    "macro_f1",
    "sensitivity",
    "specificity",

    # Probability metrics
    "roc_auc",
    "pr_auc",

    # Confusion matrices
    "confusion_matrix",
    "normalized_confusion_matrix",

    # Classification summaries
    "classification_summary",
    "get_primary_metric",
    "get_configured_metrics",

    # Fairness
    "subgroup_accuracy",
    "worst_group_accuracy",
    "bias_risk_score",
    "equalized_odds_difference",
    "fairness_summary",

    # Uncertainty
    "predictive_entropy",
    "maximum_softmax_probability",
    "expected_calibration_error",
    "uncertainty_summary",

    # Unified evaluation
    "evaluate_all",
]


# ============================================================
# Sanity Check
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CARL METRICS SANITY CHECK")
    print("=" * 70)

    # --------------------------------------------------------
    # Synthetic multiclass example
    # --------------------------------------------------------

    torch.manual_seed(42)

    num_samples = 40
    num_classes = 10

    y_true = torch.randint(
        low=0,
        high=num_classes,
        size=(num_samples,),
    )

    logits = torch.randn(
        num_samples,
        num_classes,
    )

    print(
        f"Num samples   : {num_samples}"
    )

    print(
        f"Num classes   : {num_classes}"
    )

    print()

    # --------------------------------------------------------
    # Configured metrics
    # --------------------------------------------------------

    metrics = get_configured_metrics(
        logits,
        y_true,
    )

    for name, value in metrics.items():

        if value is None:

            print(
                f"{name:<22}: None"
            )

        else:

            print(
                f"{name:<22}: "
                f"{value:.6f}"
            )

    print()

    # --------------------------------------------------------
    # Additional classification metrics
    # --------------------------------------------------------

    print(
        f"Accuracy              : "
        f"{accuracy(logits, y_true):.6f}"
    )

    print(
        f"Sensitivity            : "
        f"{sensitivity(logits, y_true):.6f}"
    )

    print(
        f"Specificity            : "
        f"{specificity(logits, y_true):.6f}"
    )

    print(
        f"PR-AUC                 : "
        f"{pr_auc(logits, y_true)}"
    )

    print()

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        logits,
        y_true,
    )

    ncm = normalized_confusion_matrix(
        logits,
        y_true,
    )

    print(
        f"Confusion matrix shape : "
        f"{cm.shape}"
    )

    print(
        f"Normalized CM shape    : "
        f"{ncm.shape}"
    )

    assert cm.shape == (
        num_classes,
        num_classes,
    )

    assert ncm.shape == (
        num_classes,
        num_classes,
    )

    print(
        "Confusion matrix       : OK"
    )

    # --------------------------------------------------------
    # Uncertainty
    # --------------------------------------------------------

    uncertainty = uncertainty_summary(
        logits,
        y_true,
    )

    print()

    print(
        f"ECE                   : "
        f"{uncertainty['expected_calibration_error']:.6f}"
    )

    print(
        f"Mean entropy          : "
        f"{uncertainty['mean_entropy']:.6f}"
    )

    print(
        f"Mean MSP              : "
        f"{uncertainty['mean_msp']:.6f}"
    )

    print(
        "Uncertainty metrics   : OK"
    )

    # --------------------------------------------------------
    # Fairness
    # --------------------------------------------------------

    metadata = {

        "sex": np.array([
            "M", "F"
        ] * (num_samples // 2)),

        "location": np.array([
            "face", "trunk"
        ] * (num_samples // 2)),
    }

    fairness = fairness_summary(
        logits,
        y_true,
        metadata,
    )

    assert "sex" in fairness
    assert "location" in fairness

    print(
        "Fairness metrics      : OK"
    )

    # --------------------------------------------------------
    # Unified evaluation
    # --------------------------------------------------------

    results = evaluate_all(
        logits,
        y_true,
        metadata=metadata,
    )

    assert (
        "balanced_accuracy"
        in results
    )

    assert (
        "macro_f1"
        in results
    )

    assert (
        "roc_auc"
        in results
    )

    assert (
        "fairness"
        in results
    )

    print(
        "Unified evaluation    : OK"
    )

    print()
    print(
        "CARL metrics sanity check completed."
    )

    print("=" * 70)