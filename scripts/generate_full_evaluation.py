"""
Complete evaluation of a multiclass skin-lesion classifier from
test_predictions.csv.

Computes:
    1. Confusion matrix
    2. Classification report
    3. Sensitivity / specificity
    4. Balanced accuracy
    5. MCC
    6. Cohen's kappa
    7. Multiclass Brier score
    8. ECE / MCE
    9. Class-wise ECE
   10. Confidence/error analysis
   11. High-confidence error analysis
   12. Bootstrap 95% confidence intervals
   13. ROC-AUC
   14. PR-AUC / Average Precision
   15. ROC curves
   16. Precision-recall curves

Input:
    predictions/test_predictions.csv

Output:
    evaluation/
        confusion_matrix.pdf
        confusion_matrix.png
        classification_report.csv
        sensitivity_specificity.csv
        summary_metrics.csv
        calibration_metrics.csv
        classwise_ece.csv
        confidence_analysis.csv
        high_confidence_errors.csv
        bootstrap_ci.csv
        roc_auc_results.csv
        pr_auc_results.csv
        roc_curves.pdf
        roc_curves.png
        pr_curves.pdf
        pr_curves.png
        brier_score.txt
        evaluation_summary.txt
"""

import os
import ast
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")


# ============================================================
# Default HAM10000 class mapping
# ============================================================

DEFAULT_HAM_CLASSES = {
    0: "AKIEC",
    1: "BCC",
    2: "BKL",
    3: "DF",
    4: "MEL",
    5: "NV",
    6: "VASC",
}


# ============================================================
# Utility functions
# ============================================================

def parse_probabilities(x):
    """Convert stringified probability list into numpy array."""
    if isinstance(x, str):
        return np.asarray(ast.literal_eval(x), dtype=float)

    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x, dtype=float)

    raise ValueError(f"Cannot parse probability value: {x}")


def expected_calibration_error(
    y_true,
    y_pred,
    probabilities,
    n_bins=10
):
    """
    Top-label Expected Calibration Error (ECE)
    and Maximum Calibration Error (MCE).

    Confidence = maximum predicted probability.
    Accuracy   = whether predicted class is correct.
    """

    confidence = np.max(probabilities, axis=1)
    correctness = (y_pred == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    rows = []

    ece = 0.0
    mce = 0.0

    n = len(y_true)

    for b in range(n_bins):
        lower = bin_edges[b]
        upper = bin_edges[b + 1]

        if b == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)

        count = int(mask.sum())

        if count == 0:
            rows.append({
                "bin": b + 1,
                "lower": lower,
                "upper": upper,
                "count": 0,
                "mean_confidence": np.nan,
                "empirical_accuracy": np.nan,
                "absolute_gap": np.nan,
            })
            continue

        mean_conf = confidence[mask].mean()
        accuracy = correctness[mask].mean()
        gap = abs(mean_conf - accuracy)

        ece += (count / n) * gap
        mce = max(mce, gap)

        rows.append({
            "bin": b + 1,
            "lower": lower,
            "upper": upper,
            "count": count,
            "mean_confidence": mean_conf,
            "empirical_accuracy": accuracy,
            "absolute_gap": gap,
        })

    return ece, mce, pd.DataFrame(rows)


def classwise_ece(
    y_true,
    probabilities,
    n_bins=10,
    n_classes=None
):
    """
    One-vs-rest class-wise ECE.

    For each class:
        confidence = probability assigned to that class
        correctness = whether the true label is that class
    """

    if n_classes is None:
        n_classes = probabilities.shape[1]

    rows = []

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for cls in range(n_classes):

        confidence = probabilities[:, cls]
        correctness = (y_true == cls).astype(float)

        ece = 0.0
        mce = 0.0

        for b in range(n_bins):

            lower = bin_edges[b]
            upper = bin_edges[b + 1]

            if b == n_bins - 1:
                mask = (confidence >= lower) & (confidence <= upper)
            else:
                mask = (confidence >= lower) & (confidence < upper)

            count = int(mask.sum())

            if count == 0:
                continue

            mean_conf = confidence[mask].mean()
            accuracy = correctness[mask].mean()
            gap = abs(mean_conf - accuracy)

            ece += (count / len(y_true)) * gap
            mce = max(mce, gap)

        rows.append({
            "class_index": cls,
            "ECE": ece,
            "MCE": mce,
        })

    return pd.DataFrame(rows)


def multiclass_brier_score(y_true, probabilities):
    """
    Multiclass Brier score.

    Brier = mean over samples of the sum of squared differences
            between predicted probability vector and one-hot target.
    """

    n_classes = probabilities.shape[1]

    y_onehot = np.zeros_like(probabilities, dtype=float)
    y_onehot[np.arange(len(y_true)), y_true] = 1.0

    return np.mean(np.sum((probabilities - y_onehot) ** 2, axis=1))


def bootstrap_metrics(
    y_true,
    y_pred,
    probabilities,
    n_bootstrap=2000,
    seed=42
):
    """
    Bootstrap 95% confidence intervals for major metrics.
    """

    rng = np.random.default_rng(seed)

    n = len(y_true)

    metric_names = [
        "Accuracy",
        "Balanced Accuracy",
        "Macro F1",
        "MCC",
        "Cohen Kappa",
    ]

    values = {name: [] for name in metric_names}

    for _ in range(n_bootstrap):

        indices = rng.integers(0, n, size=n)

        yt = y_true[indices]
        yp = y_pred[indices]

        # Avoid degenerate bootstrap samples.
        if len(np.unique(yt)) < len(np.unique(y_true)):
            continue

        values["Accuracy"].append(
            accuracy_score(yt, yp)
        )

        values["Balanced Accuracy"].append(
            balanced_accuracy_score(yt, yp)
        )

        report = classification_report(
            yt,
            yp,
            output_dict=True,
            zero_division=0
        )

        values["Macro F1"].append(
            report["macro avg"]["f1-score"]
        )

        values["MCC"].append(
            matthews_corrcoef(yt, yp)
        )

        values["Cohen Kappa"].append(
            cohen_kappa_score(yt, yp)
        )

    rows = []

    for metric, vals in values.items():

        vals = np.asarray(vals)

        rows.append({
            "metric": metric,
            "estimate": np.mean(vals),
            "lower_95_CI": np.percentile(vals, 2.5),
            "upper_95_CI": np.percentile(vals, 97.5),
            "bootstrap_samples": len(vals),
        })

    return pd.DataFrame(rows)


def save_figure(fig, output_dir, filename):
    """Save PDF and PNG."""
    pdf_path = os.path.join(output_dir, filename + ".pdf")
    png_path = os.path.join(output_dir, filename + ".png")

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# Confusion matrix
# ============================================================

def generate_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    output_dir
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names))
    )

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("Confusion Matrix")

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right"
    )

    threshold = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black"
            )

    fig.colorbar(im, ax=ax)

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "confusion_matrix"
    )

    return cm


# ============================================================
# ROC curves
# ============================================================

def generate_roc_curves(
    y_true,
    probabilities,
    class_names,
    output_dir
):

    n_classes = probabilities.shape[1]

    rows = []

    fig, ax = plt.subplots(figsize=(8, 7))

    for cls in range(n_classes):

        binary_true = (y_true == cls).astype(int)

        try:
            auc = roc_auc_score(
                binary_true,
                probabilities[:, cls]
            )

            fpr, tpr, _ = roc_curve(
                binary_true,
                probabilities[:, cls]
            )

            ax.plot(
                fpr,
                tpr,
                label=f"{class_names[cls]} (AUC={auc:.4f})"
            )

            rows.append({
                "class_index": cls,
                "class": class_names[cls],
                "ROC_AUC": auc,
            })

        except ValueError:
            rows.append({
                "class_index": cls,
                "class": class_names[cls],
                "ROC_AUC": np.nan,
            })

    macro_auc = roc_auc_score(
        y_true,
        probabilities,
        multi_class="ovr",
        average="macro"
    )

    rows.append({
        "class_index": -1,
        "class": "Macro",
        "ROC_AUC": macro_auc,
    })

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Chance"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC Curves")

    ax.legend(
        loc="lower right",
        fontsize=8
    )

    ax.grid(alpha=0.25)

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "roc_curves"
    )

    return pd.DataFrame(rows)


# ============================================================
# Precision-recall curves
# ============================================================

def generate_pr_curves(
    y_true,
    probabilities,
    class_names,
    output_dir
):

    n_classes = probabilities.shape[1]

    rows = []

    fig, ax = plt.subplots(figsize=(8, 7))

    for cls in range(n_classes):

        binary_true = (y_true == cls).astype(int)

        try:
            ap = average_precision_score(
                binary_true,
                probabilities[:, cls]
            )

            precision, recall, _ = precision_recall_curve(
                binary_true,
                probabilities[:, cls]
            )

            ax.plot(
                recall,
                precision,
                label=f"{class_names[cls]} (AP={ap:.4f})"
            )

            rows.append({
                "class_index": cls,
                "class": class_names[cls],
                "Average_Precision": ap,
            })

        except ValueError:
            rows.append({
                "class_index": cls,
                "class": class_names[cls],
                "Average_Precision": np.nan,
            })

    macro_ap = average_precision_score(
        np.eye(n_classes)[y_true],
        probabilities,
        average="macro"
    )

    rows.append({
        "class_index": -1,
        "class": "Macro",
        "Average_Precision": macro_ap,
    })

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("One-vs-Rest Precision-Recall Curves")

    ax.legend(
        loc="lower left",
        fontsize=8
    )

    ax.grid(alpha=0.25)

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "pr_curves"
    )

    return pd.DataFrame(rows)


# ============================================================
# Confidence/error analysis
# ============================================================

def confidence_analysis(
    y_true,
    y_pred,
    probabilities,
    output_dir
):

    confidence = probabilities.max(axis=1)

    correct = y_true == y_pred

    df = pd.DataFrame({
        "confidence": confidence,
        "correct": correct,
        "error": ~correct,
        "true_label": y_true,
        "predicted_label": y_pred,
    })

    summary = []

    for name, mask in [
        ("Correct", correct),
        ("Incorrect", ~correct),
        ("All", np.ones(len(df), dtype=bool)),
    ]:

        values = confidence[mask]

        summary.append({
            "group": name,
            "N": len(values),
            "mean_confidence": values.mean() if len(values) else np.nan,
            "median_confidence": np.median(values) if len(values) else np.nan,
            "std_confidence": values.std() if len(values) else np.nan,
            "min_confidence": values.min() if len(values) else np.nan,
            "max_confidence": values.max() if len(values) else np.nan,
        })

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        os.path.join(
            output_dir,
            "confidence_analysis.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # High-confidence errors
    # --------------------------------------------------------

    high_error = df[
        (~df["correct"]) &
        (df["confidence"] >= 0.90)
    ].copy()

    high_error = high_error.sort_values(
        "confidence",
        ascending=False
    )

    high_error.to_csv(
        os.path.join(
            output_dir,
            "high_confidence_errors.csv"
        ),
        index=False
    )

    return summary_df, high_error


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        default="predictions/test_predictions.csv",
        help="Path to test_predictions.csv"
    )

    parser.add_argument(
        "--output",
        default="evaluation",
        help="Output directory"
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap iterations"
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of calibration bins"
    )

    parser.add_argument(
        "--dataset",
        default="HAM10000",
        help="Dataset name"
    )

    args = parser.parse_args()

    os.makedirs(
        args.output,
        exist_ok=True
    )

    print("=" * 70)
    print("CARL COMPLETE EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    print("\nLoading:")
    print(args.csv)

    df = pd.read_csv(args.csv)

    required_columns = [
        "true_label",
        "predicted_label",
        "confidence",
        "probabilities",
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    y_true = df["true_label"].astype(int).to_numpy()

    y_pred = df["predicted_label"].astype(int).to_numpy()

    probabilities = np.vstack(
        df["probabilities"].apply(parse_probabilities).values
    )

    n_classes = probabilities.shape[1]

    print(f"\nSamples:       {len(y_true)}")
    print(f"Classes:       {n_classes}")
    print(f"Probability shape: {probabilities.shape}")

    # --------------------------------------------------------
    # Class names
    # --------------------------------------------------------

    if args.dataset.upper() == "HAM10000" and n_classes == 7:

        class_names = [
            DEFAULT_HAM_CLASSES[i]
            for i in range(n_classes)
        ]

    else:

        class_names = [
            f"Class_{i}"
            for i in range(n_classes)
        ]

    print("\nClass mapping:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

    # --------------------------------------------------------
    # Basic metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=np.arange(n_classes),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    macro_f1 = report_dict["macro avg"]["f1-score"]

    weighted_f1 = report_dict["weighted avg"]["f1-score"]

    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )

    kappa = cohen_kappa_score(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # ROC / PR
    # --------------------------------------------------------

    roc_results = generate_roc_curves(
        y_true,
        probabilities,
        class_names,
        args.output
    )

    pr_results = generate_pr_curves(
        y_true,
        probabilities,
        class_names,
        args.output
    )

    macro_roc_auc = roc_results.loc[
        roc_results["class"] == "Macro",
        "ROC_AUC"
    ].iloc[0]

    macro_ap = pr_results.loc[
        pr_results["class"] == "Macro",
        "Average_Precision"
    ].iloc[0]

    roc_results.to_csv(
        os.path.join(
            args.output,
            "roc_auc_results.csv"
        ),
        index=False
    )

    pr_results.to_csv(
        os.path.join(
            args.output,
            "pr_auc_results.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = generate_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        args.output
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report_df = pd.DataFrame(report_dict).T

    report_df.to_csv(
        os.path.join(
            args.output,
            "classification_report.csv"
        )
    )

    # --------------------------------------------------------
    # Sensitivity / specificity
    # --------------------------------------------------------

    sensitivity_specificity = []

    for cls in range(n_classes):

        tp = cm[cls, cls]

        fn = cm[cls, :].sum() - tp

        fp = cm[:, cls].sum() - tp

        tn = cm.sum() - tp - fn - fp

        sensitivity = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else np.nan
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else np.nan
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else np.nan
        )

        npv = (
            tn / (tn + fn)
            if (tn + fn) > 0
            else np.nan
        )

        sensitivity_specificity.append({
            "class_index": cls,
            "class": class_names[cls],
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Precision": precision,
            "NPV": npv,
        })

    ss_df = pd.DataFrame(
        sensitivity_specificity
    )

    ss_df.to_csv(
        os.path.join(
            args.output,
            "sensitivity_specificity.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    ece, mce, calibration_bins = expected_calibration_error(
        y_true,
        y_pred,
        probabilities,
        n_bins=args.bins
    )

    calibration_bins.to_csv(
        os.path.join(
            args.output,
            "calibration_bins.csv"
        ),
        index=False
    )

    class_ece = classwise_ece(
        y_true,
        probabilities,
        n_bins=args.bins,
        n_classes=n_classes
    )

    class_ece["class"] = class_ece[
        "class_index"
    ].map(
        dict(enumerate(class_names))
    )

    class_ece.to_csv(
        os.path.join(
            args.output,
            "classwise_ece.csv"
        ),
        index=False
    )

    brier = multiclass_brier_score(
        y_true,
        probabilities
    )

    with open(
        os.path.join(
            args.output,
            "brier_score.txt"
        ),
        "w"
    ) as f:

        f.write(
            f"Multiclass Brier Score: {brier:.8f}\n"
        )

    # --------------------------------------------------------
    # Confidence analysis
    # --------------------------------------------------------

    confidence_df, high_error_df = confidence_analysis(
        y_true,
        y_pred,
        probabilities,
        args.output
    )

    # --------------------------------------------------------
    # Bootstrap confidence intervals
    # --------------------------------------------------------

    print(
        f"\nRunning {args.bootstrap} bootstrap iterations..."
    )

    bootstrap_df = bootstrap_metrics(
        y_true,
        y_pred,
        probabilities,
        n_bootstrap=args.bootstrap,
        seed=42
    )

    bootstrap_df.to_csv(
        os.path.join(
            args.output,
            "bootstrap_ci.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Summary metrics
    # --------------------------------------------------------

    summary = pd.DataFrame([
        {
            "Dataset": args.dataset,
            "N": len(y_true),
            "Accuracy": accuracy,
            "Balanced_Accuracy": balanced_accuracy,
            "Macro_F1": macro_f1,
            "Weighted_F1": weighted_f1,
            "Macro_ROC_AUC": macro_roc_auc,
            "Macro_Average_Precision": macro_ap,
            "MCC": mcc,
            "Cohen_Kappa": kappa,
            "Brier_Score": brier,
            "ECE": ece,
            "MCE": mce,
        }
    ])

    summary.to_csv(
        os.path.join(
            args.output,
            "summary_metrics.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Text summary
    # --------------------------------------------------------

    with open(
        os.path.join(
            args.output,
            "evaluation_summary.txt"
        ),
        "w"
    ) as f:

        f.write("=" * 70 + "\n")
        f.write("CARL COMPLETE EVALUATION\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Samples: {len(y_true)}\n")
        f.write(f"Classes: {n_classes}\n\n")

        f.write("OVERALL METRICS\n")
        f.write("-" * 40 + "\n")

        f.write(f"Accuracy:              {accuracy:.6f}\n")
        f.write(f"Balanced Accuracy:     {balanced_accuracy:.6f}\n")
        f.write(f"Macro F1:              {macro_f1:.6f}\n")
        f.write(f"Weighted F1:           {weighted_f1:.6f}\n")
        f.write(f"Macro ROC-AUC:         {macro_roc_auc:.6f}\n")
        f.write(f"Macro Average Precision:{macro_ap:.6f}\n")
        f.write(f"MCC:                   {mcc:.6f}\n")
        f.write(f"Cohen Kappa:           {kappa:.6f}\n")
        f.write(f"Brier Score:           {brier:.6f}\n")
        f.write(f"ECE:                   {ece:.6f}\n")
        f.write(f"MCE:                   {mce:.6f}\n\n")

        f.write("CLASS MAPPING\n")
        f.write("-" * 40 + "\n")

        for i, name in enumerate(class_names):
            f.write(f"{i}: {name}\n")

        f.write("\nHIGH-CONFIDENCE ERRORS\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"Errors with confidence >= 0.90: "
            f"{len(high_error_df)}\n"
        )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"Accuracy:              {accuracy:.4f}")
    print(f"Balanced Accuracy:     {balanced_accuracy:.4f}")
    print(f"Macro F1:              {macro_f1:.4f}")
    print(f"Weighted F1:           {weighted_f1:.4f}")
    print(f"Macro ROC-AUC:         {macro_roc_auc:.4f}")
    print(f"Macro Average Precision:{macro_ap:.4f}")
    print(f"MCC:                   {mcc:.4f}")
    print(f"Cohen Kappa:           {kappa:.4f}")
    print(f"Brier Score:           {brier:.4f}")
    print(f"ECE:                   {ece:.4f}")
    print(f"MCE:                   {mce:.4f}")

    print("\nHigh-confidence errors (>= 0.90):",
          len(high_error_df))

    print("\n" + "=" * 70)
    print("FILES SAVED TO")
    print("=" * 70)

    print(os.path.abspath(args.output))


if __name__ == "__main__":
    main()