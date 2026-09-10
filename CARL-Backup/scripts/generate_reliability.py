import ast
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# HAM10000 class names
# These must correspond to the label mapping used by data.py.
# ============================================================
HAM_CLASSES = {
    0: "AKIEC",
    1: "BCC",
    2: "BKL",
    3: "DF",
    4: "MEL",
    5: "NV",
    6: "VASC",
}


def load_predictions(csv_path):
    df = pd.read_csv(csv_path)

    required = ["true_label", "predicted_label", "confidence", "probabilities"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    probabilities = np.array(
        [ast.literal_eval(x) for x in df["probabilities"]],
        dtype=np.float64,
    )

    true_labels = df["true_label"].to_numpy(dtype=int)
    predicted_labels = df["predicted_label"].to_numpy(dtype=int)

    confidence = df["confidence"].to_numpy(dtype=np.float64)

    # Verify probabilities
    if probabilities.ndim != 2:
        raise ValueError(
            f"Expected probability matrix [N, C], got {probabilities.shape}"
        )

    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Probability vectors do not sum to 1.")

    # Verify confidence against maximum probability
    max_prob = probabilities.max(axis=1)

    if not np.allclose(confidence, max_prob, atol=1e-5):
        print(
            "WARNING: confidence column does not exactly match "
            "maximum probability."
        )

    return df, true_labels, predicted_labels, probabilities


def compute_reliability(
    true_labels,
    predicted_labels,
    probabilities,
    n_bins=10,
):
    confidence = probabilities.max(axis=1)
    correctness = (predicted_labels == true_labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_confidence = []
    bin_accuracy = []
    bin_count = []

    for i in range(n_bins):

        if i == n_bins - 1:
            mask = (
                (confidence >= bin_edges[i])
                & (confidence <= bin_edges[i + 1])
            )
        else:
            mask = (
                (confidence >= bin_edges[i])
                & (confidence < bin_edges[i + 1])
            )

        count = mask.sum()

        if count == 0:
            bin_confidence.append(np.nan)
            bin_accuracy.append(np.nan)
            bin_count.append(0)
        else:
            bin_confidence.append(confidence[mask].mean())
            bin_accuracy.append(correctness[mask].mean())
            bin_count.append(count)

    bin_confidence = np.asarray(bin_confidence)
    bin_accuracy = np.asarray(bin_accuracy)
    bin_count = np.asarray(bin_count)

    # Expected Calibration Error
    ece = 0.0

    for i in range(n_bins):
        if bin_count[i] > 0:
            ece += (
                bin_count[i] / len(confidence)
            ) * abs(
                bin_accuracy[i] - bin_confidence[i]
            )

    # Maximum Calibration Error
    valid = bin_count > 0

    if np.any(valid):
        mce = np.max(
            np.abs(
                bin_accuracy[valid] - bin_confidence[valid]
            )
        )
    else:
        mce = np.nan

    return (
        bin_edges,
        bin_confidence,
        bin_accuracy,
        bin_count,
        ece,
        mce,
    )


def plot_reliability(
    bin_confidence,
    bin_accuracy,
    bin_count,
    ece,
    mce,
    output_pdf,
    output_png,
    dataset_name,
):
    fig, ax = plt.subplots(figsize=(6.0, 5.2))

    valid = bin_count > 0

    # Perfect calibration
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Perfect calibration",
    )

    # Observed calibration
    ax.plot(
        bin_confidence[valid],
        bin_accuracy[valid],
        marker="o",
        linewidth=2.0,
        markersize=5,
        label="CARL",
    )

    ax.set_xlabel("Mean predicted confidence", fontsize=11)
    ax.set_ylabel("Fraction of correct predictions", fontsize=11)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.grid(True, alpha=0.25)

    ax.legend(
        loc="upper left",
        frameon=True,
        fontsize=9,
    )

    ax.text(
        0.97,
        0.08,
        f"ECE = {ece:.4f}\nMCE = {mce:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9,
        ),
    )

    ax.set_title(
        f"Reliability Diagram: {dataset_name}",
        fontsize=12,
        fontweight="bold",
    )

    fig.tight_layout()

    fig.savefig(
        output_pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_calibration_table(
    bin_confidence,
    bin_accuracy,
    bin_count,
    ece,
    mce,
    output_csv,
):
    table = pd.DataFrame(
        {
            "bin": np.arange(1, len(bin_count) + 1),
            "mean_confidence": bin_confidence,
            "accuracy": bin_accuracy,
            "sample_count": bin_count,
        }
    )

    table["absolute_gap"] = (
        table["mean_confidence"] - table["accuracy"]
    ).abs()

    table.to_csv(output_csv, index=False)

    metrics_csv = Path(output_csv).with_name(
        Path(output_csv).stem + "_metrics.csv"
    )

    pd.DataFrame(
        {
            "ECE": [ece],
            "MCE": [mce],
        }
    ).to_csv(metrics_csv, index=False)


def main():

    parser = argparse.ArgumentParser(
        description="Generate reliability diagram from CARL predictions."
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to test_predictions.csv",
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["ham10000", "padufes20", "pad_ufes20"],
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/reliability",
    )

    parser.add_argument(
        "--n_bins",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = (
        "HAM10000"
        if args.dataset == "ham10000"
        else "PAD-UFES-20"
    )

    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Input:   {args.csv}")
    print("=" * 60)

    (
        df,
        true_labels,
        predicted_labels,
        probabilities,
    ) = load_predictions(args.csv)

    print(f"Number of samples: {len(df)}")
    print(f"Number of classes: {probabilities.shape[1]}")

    (
        bin_edges,
        bin_confidence,
        bin_accuracy,
        bin_count,
        ece,
        mce,
    ) = compute_reliability(
        true_labels,
        predicted_labels,
        probabilities,
        n_bins=args.n_bins,
    )

    print("\nCalibration results")
    print("-" * 60)
    print(f"ECE: {ece:.6f}")
    print(f"MCE: {mce:.6f}")

    print("\nReliability bins")
    print("-" * 60)

    for i in range(args.n_bins):
        if bin_count[i] > 0:
            print(
                f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f} | "
                f"N={bin_count[i]:4d} | "
                f"Confidence={bin_confidence[i]:.4f} | "
                f"Accuracy={bin_accuracy[i]:.4f}"
            )

    prefix = (
        "ham10000"
        if args.dataset == "ham10000"
        else "padufes20"
    )

    pdf_path = output_dir / f"{prefix}_reliability.pdf"
    png_path = output_dir / f"{prefix}_reliability.png"
    csv_path = output_dir / f"{prefix}_reliability_bins.csv"

    plot_reliability(
        bin_confidence,
        bin_accuracy,
        bin_count,
        ece,
        mce,
        pdf_path,
        png_path,
        dataset_name,
    )

    save_calibration_table(
        bin_confidence,
        bin_accuracy,
        bin_count,
        ece,
        mce,
        csv_path,
    )

    print("\nSaved:")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")
    print(f"  CSV: {csv_path}")
    print(
        f"  Metrics: "
        f"{csv_path.with_name(csv_path.stem + '_metrics.csv')}"
    )


if __name__ == "__main__":
    main()
