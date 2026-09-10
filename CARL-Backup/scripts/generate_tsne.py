"""
=========================================================
CARL - t-SNE Visualization
=========================================================

Visualizes the learned CARL embeddings on the held-out
test set for:

    - HAM10000
    - PAD-UFES-20

Pipeline
--------
Image
  ↓
EfficientNet-B0
  ↓
CA-AS3
  ↓
CAFA
  ↓
Representation Projection
  ↓
Global Average Pooling
  ↓
CARL Projection Head
  ↓
128-D normalized embedding
  ↓
t-SNE
  ↓
2-D visualization

The script uses the exact dataset class mapping defined
by data.py and does not hard-code disease labels.

Outputs
-------
For each dataset:

    tsne_<dataset>.pdf
    tsne_<dataset>.png
    tsne_<dataset>_coordinates.csv

Author: Prasad Kanhegaonkar
=========================================================
"""

from __future__ import annotations

import argparse
import ast
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from sklearn.manifold import TSNE

from src.model import build_model
from src.data import build_datasets


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed: int = 42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# Device
# =========================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# =========================================================
# Checkpoint Loading
# =========================================================

def load_checkpoint(model, checkpoint_path, device):

    print()
    print("=" * 70)
    print("LOADING CHECKPOINT")
    print("=" * 70)

    print(f"Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # -----------------------------------------------------
    # Handle common checkpoint formats
    # -----------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        elif "model" in checkpoint:

            state_dict = checkpoint["model"]

        else:

            # Some experiments save the state_dict
            # directly inside the checkpoint.
            state_dict = checkpoint

    else:

        raise ValueError(
            "Unsupported checkpoint format."
        )

    # -----------------------------------------------------
    # Remove possible DistributedDataParallel prefix
    # -----------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[len("module."):]

        cleaned_state_dict[key] = value

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False,
    )

    if missing:
        print()
        print("WARNING: Missing keys:")
        for key in missing:
            print(f"  {key}")

    if unexpected:
        print()
        print("WARNING: Unexpected keys:")
        for key in unexpected:
            print(f"  {key}")

    print()
    print("Checkpoint loaded successfully.")

    return checkpoint


# =========================================================
# Extract CARL Embeddings
# =========================================================

@torch.no_grad()
def extract_embeddings(
    model,
    dataset,
    device,
    batch_size=64,
    num_workers=4,
):
    """
    Extract the normalized CARL embeddings from the test set.

    The model output used is:

        outputs["embedding"]

    which is the normalized CARL embedding.
    """

    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    embeddings = []
    labels = []
    image_names = []
    patient_ids = []

    print()
    print("=" * 70)
    print("EXTRACTING CARL EMBEDDINGS")
    print("=" * 70)

    model.eval()

    for batch_index, batch in enumerate(loader):

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        output = model(
            images,
            epoch=0,
        )

        embedding = output["embedding"]

        if embedding is None:

            raise RuntimeError(
                "Model returned embedding=None. "
                "Make sure cfg.carl.enabled=True."
            )

        # -------------------------------------------------
        # Verify embedding dimensionality
        # -------------------------------------------------

        if embedding.ndim != 2:

            raise RuntimeError(
                "Unexpected embedding shape: "
                f"{tuple(embedding.shape)}. "
                "Expected [B, embedding_dim]."
            )

        embeddings.append(
            embedding.cpu().numpy()
        )

        labels.append(
            batch["label"].cpu().numpy()
        )

        image_names.extend(
            batch["image_name"]
        )

        patient_ids.extend(
            batch["patient_id"]
        )

        if (
            batch_index % 10 == 0
            or batch_index == len(loader) - 1
        ):

            print(
                f"Batch "
                f"{batch_index + 1:4d}/"
                f"{len(loader):4d}"
            )

    embeddings = np.concatenate(
        embeddings,
        axis=0,
    )

    labels = np.concatenate(
        labels,
        axis=0,
    )

    print()
    print(
        f"Samples   : {len(embeddings):,}"
    )

    print(
        f"Embedding : {embeddings.shape}"
    )

    print(
        f"Dimension  : {embeddings.shape[1]}"
    )

    return (
        embeddings,
        labels,
        image_names,
        patient_ids,
    )


# =========================================================
# t-SNE
# =========================================================

def compute_tsne(
    embeddings,
    seed=42,
    perplexity=30,
):

    n_samples = len(embeddings)

    # -----------------------------------------------------
    # Safety check
    # -----------------------------------------------------

    if n_samples <= perplexity:

        perplexity = max(
            5,
            min(
                30,
                (n_samples - 1) // 3,
            ),
        )

        print(
            f"Adjusted perplexity to "
            f"{perplexity}"
        )

    print()
    print("=" * 70)
    print("RUNNING t-SNE")
    print("=" * 70)

    print(
        f"Samples       : {n_samples:,}"
    )

    print(
        f"Input dim     : {embeddings.shape[1]}"
    )

    print(
        f"Perplexity    : {perplexity}"
    )

    print(
        f"Random seed   : {seed}"
    )

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=2000,
        random_state=seed,
    )

    coordinates = tsne.fit_transform(
        embeddings
    )

    print()
    print("t-SNE completed.")

    return coordinates


# =========================================================
# Plot
# =========================================================

def plot_tsne(
    coordinates,
    labels,
    class_names,
    dataset_name,
    output_path,
):
    """
    Generate publication-quality t-SNE visualization.
    """

    print()
    print("=" * 70)
    print("GENERATING t-SNE FIGURE")
    print("=" * 70)

    fig, ax = plt.subplots(
        figsize=(7.2, 5.8)
    )

    unique_labels = sorted(
        np.unique(labels)
    )

    for label_index in unique_labels:

        mask = labels == label_index

        class_name = class_names.get(
            int(label_index),
            str(label_index),
        )

        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=18,
            alpha=0.70,
            label=class_name,
            edgecolors="none",
        )

    ax.set_xlabel(
        "t-SNE dimension 1",
        fontsize=11,
    )

    ax.set_ylabel(
        "t-SNE dimension 2",
        fontsize=11,
    )

    ax.set_title(
        f"CARL Representation: {dataset_name}",
        fontsize=12,
    )

    ax.tick_params(
        axis="both",
        labelsize=9,
    )

    ax.legend(
        fontsize=8,
        frameon=True,
        loc="best",
        markerscale=1.3,
    )

    ax.grid(
        False
    )

    fig.tight_layout()

    # -----------------------------------------------------
    # PDF
    # -----------------------------------------------------

    pdf_path = output_path.with_suffix(
        ".pdf"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    # -----------------------------------------------------
    # PNG
    # -----------------------------------------------------

    png_path = output_path.with_suffix(
        ".png"
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(
        f"PDF saved : {pdf_path}"
    )

    print(
        f"PNG saved : {png_path}"
    )


# =========================================================
# Save Coordinates
# =========================================================

def save_coordinates(
    coordinates,
    labels,
    image_names,
    patient_ids,
    class_names,
    output_path,
):

    df = pd.DataFrame(
        {
            "image": image_names,
            "patient": patient_ids,
            "label": labels,
            "class": [
                class_names.get(
                    int(label),
                    str(label),
                )
                for label in labels
            ],
            "tsne_1": coordinates[:, 0],
            "tsne_2": coordinates[:, 1],
        }
    )

    csv_path = (
        output_path.parent
        / (
            output_path.stem
            + "_coordinates.csv"
        )
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    print(
        f"CSV saved  : {csv_path}"
    )


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate t-SNE visualization "
            "of CARL embeddings."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="/home/prasad/CARL/config.yaml",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "ham10000",
            "pad_ufes20",
            "padufes20",
        ],
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/prasad/CARL/outputs/tsne",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--perplexity",
        type=float,
        default=30,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Seed
    # -----------------------------------------------------

    set_seed(
        args.seed
    )

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = get_device()

    print()
    print("=" * 70)
    print("CARL t-SNE VISUALIZATION")
    print("=" * 70)

    print(
        f"Dataset : {args.dataset}"
    )

    print(
        f"Device  : {device}"
    )

    # -----------------------------------------------------
    # Load configuration
    # -----------------------------------------------------

    cfg = OmegaConf.load(
        args.config
    )

    dataset_name = args.dataset.lower()

    if dataset_name == "padufes20":
        dataset_name = "padufes20"

    cfg.dataset.name = dataset_name

    print(
        f"Config dataset: "
        f"{cfg.dataset.name}"
    )

    # -----------------------------------------------------
    # Build datasets
    # -----------------------------------------------------

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = build_datasets(
        cfg
    )

    print()
    print(
        f"Test samples: "
        f"{len(test_dataset):,}"
    )

    # -----------------------------------------------------
    # Class mapping
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("CLASS MAPPING")
    print("=" * 70)

    class_names = {
        int(index): str(label)
        for index, label
        in test_dataset.index_to_label.items()
    }

    for index in sorted(
        class_names
    ):

        print(
            f"  [{index}] "
            f"{class_names[index]}"
        )

    # -----------------------------------------------------
    # Build model
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("BUILDING MODEL")
    print("=" * 70)

    model = build_model(
        cfg
    )

    model = model.to(
        device
    )

    # -----------------------------------------------------
    # Load trained weights
    # -----------------------------------------------------

    load_checkpoint(
        model,
        Path(args.checkpoint),
        device,
    )

    model.eval()

    # -----------------------------------------------------
    # Extract embeddings
    # -----------------------------------------------------

    (
        embeddings,
        labels,
        image_names,
        patient_ids,
    ) = extract_embeddings(
        model=model,
        dataset=test_dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # -----------------------------------------------------
    # Verify embedding
    # -----------------------------------------------------

    if embeddings.shape[1] != 128:

        print()
        print(
            "WARNING:"
        )

        print(
            f"Expected CARL embedding dimension "
            f"128, but obtained "
            f"{embeddings.shape[1]}."
        )

    # -----------------------------------------------------
    # t-SNE
    # -----------------------------------------------------

    coordinates = compute_tsne(
        embeddings=embeddings,
        seed=args.seed,
        perplexity=args.perplexity,
    )

    # -----------------------------------------------------
    # Output directory
    # -----------------------------------------------------

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"tsne_{dataset_name}"
    )

    # -----------------------------------------------------
    # Save coordinates
    # -----------------------------------------------------

    save_coordinates(
        coordinates=coordinates,
        labels=labels,
        image_names=image_names,
        patient_ids=patient_ids,
        class_names=class_names,
        output_path=output_path,
    )

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------

    plot_tsne(
        coordinates=coordinates,
        labels=labels,
        class_names=class_names,
        dataset_name=dataset_name,
        output_path=output_path,
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("COMPLETED")
    print("=" * 70)

    print(
        f"Dataset       : {dataset_name}"
    )

    print(
        f"Samples       : {len(labels):,}"
    )

    print(
        f"Embedding dim : {embeddings.shape[1]}"
    )

    print(
        f"Output        : {output_path.parent}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
