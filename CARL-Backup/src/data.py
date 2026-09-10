"""
=========================================================
CARL
Generic Dataset & Transform Utilities
=========================================================

Supported datasets
------------------
- Derm7pt
- HAM10000
- PAD-UFES-20
- HIBA

Design
------
The dataset is selected through:

    cfg.dataset.name

Dataset-specific information is resolved from:

    cfg.dataset.registry

The rest of the training pipeline is dataset-agnostic.

Expected metadata fields
------------------------
image       : image filename/path
patient     : patient identifier
split       : train / val / test
label       : multiclass label
age         : patient age
sex         : sex
location    : lesion location

The loader returns:
    image
    label
    binary_label
    image_name
    patient_id
    metadata
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# =========================================================
# Worker Seeding
# =========================================================

def seed_worker(worker_id: int) -> None:
    """
    Seed every DataLoader worker independently.

    Parameters
    ----------
    worker_id : int
        Worker identifier supplied by PyTorch.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)

    random.seed(worker_seed)
    np.random.seed(worker_seed)


# =========================================================
# Dataset Configuration Resolution
# =========================================================

def resolve_dataset_config(cfg):
    """
    Resolve dataset-specific configuration from the registry.

    The selected dataset is specified by:

        cfg.dataset.name

    and dataset-specific information is stored in:

        cfg.dataset.registry

    Returns
    -------
    dataset_cfg
        OmegaConf node containing:
            root
            metadata_file
            num_classes
    """
    dataset_name = str(cfg.dataset.name).lower()

    if "registry" not in cfg.dataset:
        raise KeyError(
            "Missing 'dataset.registry' in config.yaml."
        )

    registry = cfg.dataset.registry

    if dataset_name not in registry:
        available = list(registry.keys())

        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available datasets: {available}"
        )

    return registry[dataset_name]


# =========================================================
# Image Transformations
# =========================================================

def build_transforms(cfg):
    """
    Build train / validation / test transforms from YAML.

    Training:
        Resize
        Horizontal flip
        Vertical flip
        Rotation
        Color jitter
        ToTensor
        Normalize

    Validation/Test:
        Resize
        ToTensor
        Normalize

    Returns
    -------
    train_transform
    val_transform
    test_transform
    """

    image_size = int(cfg.dataset.image_size)

    mean = list(cfg.dataset.normalization.mean)
    std = list(cfg.dataset.normalization.std)

    # -----------------------------------------------------
    # Training transformations
    # -----------------------------------------------------

    train_tf = []

    train_tf.append(
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BILINEAR,
        )
    )

    aug = cfg.augmentation.train

    # Horizontal flip
    if aug.hflip.enabled:
        train_tf.append(
            transforms.RandomHorizontalFlip(
                p=float(aug.hflip.p)
            )
        )

    # Vertical flip
    if aug.vflip.enabled:
        train_tf.append(
            transforms.RandomVerticalFlip(
                p=float(aug.vflip.p)
            )
        )

    # Rotation
    if float(aug.rotation) > 0:
        train_tf.append(
            transforms.RandomRotation(
                degrees=float(aug.rotation),
                interpolation=InterpolationMode.BILINEAR,
            )
        )

    # Color jitter
    if aug.color_jitter.enabled:
        train_tf.append(
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05,
            )
        )

    train_tf.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_transform = transforms.Compose(train_tf)

    # -----------------------------------------------------
    # Validation / Test transformations
    # -----------------------------------------------------

    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return train_transform, eval_transform, eval_transform


# =========================================================
# Generic Skin Dataset
# =========================================================

class SkinDataset(Dataset):
    """
    Generic metadata-driven skin lesion dataset.

    Dataset-specific configuration is obtained from the
    dataset registry in config.yaml.

    The dataset class itself is independent of:
        - ResNet-50
        - EfficientNet-B0
        - CA-AS3
        - CAFA
        - CARL

    It only provides images, labels and metadata.
    """

    def __init__(
        self,
        cfg,
        split: str,
        transform=None,
    ):
        self.cfg = cfg
        self.transform = transform
        self.split = split

        # -------------------------------------------------
        # Resolve selected dataset
        # -------------------------------------------------

        self.dataset_name = str(
            cfg.dataset.name
        ).lower()

        self.dataset_cfg = resolve_dataset_config(cfg)

        self.root = Path(
            str(self.dataset_cfg.root)
        )

        self.metadata_file = Path(
            str(self.dataset_cfg.metadata_file)
        )

        self.expected_num_classes = int(
            self.dataset_cfg.num_classes
        )

        # -------------------------------------------------
        # General dataset configuration
        # -------------------------------------------------

        self.task = str(
            cfg.dataset.task
        ).lower()

        self.verify_files = bool(
            cfg.dataset.verify_files
        )

        self.verify_metadata = bool(
            cfg.dataset.verify_metadata
        )

        self.return_metadata = bool(
            cfg.dataset.get("return_metadata", True)
        )

        # -------------------------------------------------
        # Metadata column configuration
        # -------------------------------------------------

        self.image_column = str(
            cfg.dataset.metadata.image
        )

        self.patient_column = str(
            cfg.dataset.metadata.patient
        )

        self.split_column = str(
            cfg.dataset.metadata.split
        )

        self.label_column = str(
            cfg.dataset.label_column
        )

        self.binary_label_column = str(
            cfg.dataset.binary_label_column
        )

        # -------------------------------------------------
        # Validate metadata file
        # -------------------------------------------------

        if not self.metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found:\n"
                f"{self.metadata_file}"
            )

        # -------------------------------------------------
        # Load metadata
        # -------------------------------------------------

        full_df = pd.read_csv(
            self.metadata_file
        )

        if full_df.empty:
            raise ValueError(
                f"Metadata file is empty:\n"
                f"{self.metadata_file}"
            )

        # -------------------------------------------------
        # Validate required columns
        # -------------------------------------------------

        required_columns = [
            self.image_column,
            self.patient_column,
            self.split_column,
            self.label_column,
        ]

        missing_columns = [
            col
            for col in required_columns
            if col not in full_df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required metadata columns "
                f"for dataset '{self.dataset_name}': "
                f"{missing_columns}\n"
                f"Available columns: "
                f"{list(full_df.columns)}"
            )

        # -------------------------------------------------
        # Validate binary label column
        # -------------------------------------------------

        if (
            self.task == "binary"
            and self.binary_label_column
            not in full_df.columns
        ):
            raise ValueError(
                f"Binary task requires column "
                f"'{self.binary_label_column}', "
                f"but it is missing from metadata."
            )

        # -------------------------------------------------
        # Validate split column
        # -------------------------------------------------

        available_splits = (
            full_df[self.split_column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if split not in available_splits:
            raise ValueError(
                f"Split '{split}' not found for "
                f"dataset '{self.dataset_name}'. "
                f"Available splits: {available_splits}"
            )

        # -------------------------------------------------
        # Filter requested split
        # -------------------------------------------------

        self.df = (
            full_df[
                full_df[self.split_column].astype(str)
                == str(split)
            ]
            .reset_index(drop=True)
        )

        if self.df.empty:
            raise ValueError(
                f"No samples found for split '{split}' "
                f"in dataset '{self.dataset_name}'."
            )

        # -------------------------------------------------
        # Validate labels
        # -------------------------------------------------

        if self.df[self.label_column].isna().any():
            n_missing = int(
                self.df[self.label_column]
                .isna()
                .sum()
            )

            raise ValueError(
                f"Found {n_missing} samples with missing "
                f"labels in split '{split}'."
            )

        # -------------------------------------------------
        # Create deterministic class mapping
        # -------------------------------------------------

        all_classes = (
            full_df[self.label_column]
            .dropna()
            .unique()
            .tolist()
        )

        all_classes = sorted(
            all_classes,
            key=lambda x: str(x)
        )

        if self.task == "multiclass":

            self.label_to_index = {
                label: idx
                for idx, label
                in enumerate(all_classes)
            }

            self.index_to_label = {
                idx: label
                for label, idx
                in self.label_to_index.items()
            }

            self.classes = all_classes

            self.num_classes = len(
                self.label_to_index
            )

            if self.num_classes != self.expected_num_classes:
                raise ValueError(
                    f"Dataset '{self.dataset_name}' "
                    f"configuration specifies "
                    f"{self.expected_num_classes} classes, "
                    f"but metadata contains "
                    f"{self.num_classes} unique labels.\n"
                    f"Labels: {self.classes}"
                )

        elif self.task == "binary":

            self.label_to_index = None
            self.index_to_label = None

            self.classes = [0, 1]
            self.num_classes = 2

        else:
            raise ValueError(
                f"Unsupported task '{self.task}'. "
                f"Expected 'multiclass' or 'binary'."
            )

        # -------------------------------------------------
        # Validate metadata columns requested by config
        # -------------------------------------------------

        self.metadata_columns = {}

        for key, column in cfg.dataset.metadata.items():

            column = str(column)

            if column not in full_df.columns:
                if self.verify_metadata:
                    raise ValueError(
                        f"Configured metadata field "
                        f"'{key}: {column}' is missing "
                        f"from dataset '{self.dataset_name}'.\n"
                        f"Available columns: "
                        f"{list(full_df.columns)}"
                    )

            else:
                self.metadata_columns[key] = column

        # -------------------------------------------------
        # Patient-level split verification
        # -------------------------------------------------

        if bool(cfg.dataset.patient_level_split):

            self._verify_patient_level_split(
                full_df
            )

        # -------------------------------------------------
        # Verify image files
        # -------------------------------------------------

        if self.verify_files:
            self._verify_image_files()

        # -------------------------------------------------
        # Convert rows to dictionaries
        # -------------------------------------------------

        self.records = self.df.to_dict(
            orient="records"
        )

    # =====================================================
    # Patient-Level Split Verification
    # =====================================================

    def _verify_patient_level_split(
        self,
        full_df: pd.DataFrame,
    ) -> None:
        """
        Verify that a patient does not occur in multiple
        dataset splits.

        This does not create splits. It only verifies that
        the already-created metadata split satisfies the
        patient-level constraint.
        """

        if self.patient_column not in full_df.columns:
            raise ValueError(
                f"Patient-level split is enabled, but "
                f"patient column '{self.patient_column}' "
                f"is missing."
            )

        patient_split_counts = (
            full_df
            .dropna(
                subset=[
                    self.patient_column,
                    self.split_column,
                ]
            )
            .groupby(self.patient_column)[
                self.split_column
            ]
            .nunique()
        )

        conflicting_patients = (
            patient_split_counts[
                patient_split_counts > 1
            ]
        )

        if len(conflicting_patients) > 0:

            examples = (
                conflicting_patients
                .head(10)
                .index
                .tolist()
            )

            raise ValueError(
                "Patient-level split violation detected. "
                f"{len(conflicting_patients)} patients "
                "occur in more than one split.\n"
                f"Example patient IDs: {examples}"
            )

    # =====================================================
    # Image File Verification
    # =====================================================

    def _verify_image_files(self) -> None:
        """
        Verify that every image referenced by the current
        dataset metadata exists on disk.
        """

        missing = []

        for image_name in self.df[
            self.image_column
        ]:

            image_path = self._resolve_image_path(
                image_name
            )

            if not image_path.exists():
                missing.append(
                    str(image_path)
                )

        if missing:

            preview = "\n".join(
                missing[:10]
            )

            raise FileNotFoundError(
                f"{len(missing)} image files missing "
                f"from dataset '{self.dataset_name}'.\n"
                f"Examples:\n{preview}"
            )

    # =====================================================
    # Image Path Resolution
    # =====================================================

    def _resolve_image_path(
        self,
        image_name,
    ) -> Path:
        """
        Resolve image filename/path against the dataset
        root.

        Handles:
            relative paths
            filenames with extensions
            filenames without extensions
        """

        image_name = str(image_name)

        image_path = Path(image_name)

        # -------------------------------------------------
        # Absolute path
        # -------------------------------------------------

        if image_path.is_absolute():
            return image_path

        # -------------------------------------------------
        # Normal relative path
        # -------------------------------------------------

        candidate = self.root / image_path

        if candidate.exists():
            return candidate

        # -------------------------------------------------
        # Try common image extensions when metadata
        # contains filename without extension.
        # -------------------------------------------------

        if candidate.suffix == "":
            extensions = [
                ".jpg",
                ".jpeg",
                ".png",
                ".JPG",
                ".JPEG",
                ".PNG",
            ]

            for extension in extensions:

                extended = candidate.with_suffix(
                    extension
                )

                if extended.exists():
                    return extended

        return candidate

    # =====================================================
    # Length
    # =====================================================

    def __len__(self) -> int:
        return len(self.records)

    # =====================================================
    # Image Loader
    # =====================================================

    def load_image(
        self,
        path: Path,
    ) -> Image.Image:
        """
        Load an RGB image using the configured backend.
        """

        backend = str(
            self.cfg.dataset.loader.backend
        ).lower()

        if backend == "opencv":

            image = cv2.imread(
                str(path)
            )

            if image is None:
                raise RuntimeError(
                    f"Cannot read image:\n{path}"
                )

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

            image = Image.fromarray(
                image
            )

        elif backend == "pil":

            image = Image.open(
                path
            ).convert("RGB")

        else:

            raise ValueError(
                f"Unknown image backend: "
                f"'{backend}'. "
                f"Expected 'opencv' or 'pil'."
            )

        return image

    # =====================================================
    # Get Item
    # =====================================================

    def __getitem__(
        self,
        index: int,
    ) -> Dict:

        row = self.records[index]

        # -------------------------------------------------
        # Image
        # -------------------------------------------------

        image_path = self._resolve_image_path(
            row[self.image_column]
        )

        image = self.load_image(
            image_path
        )

        if self.transform is not None:
            image = self.transform(
                image
            )

        # -------------------------------------------------
        # Label
        # -------------------------------------------------

        if self.task == "multiclass":

            label_value = row[
                self.label_column
            ]

            label = self.label_to_index[
                label_value
            ]

        else:

            label = int(
                row[
                    self.binary_label_column
                ]
            )

        # -------------------------------------------------
        # Basic sample
        # -------------------------------------------------

        sample = {
            "image": image,

            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),

            "image_name": str(
                row[self.image_column]
            ),

            "patient_id": str(
                row[self.patient_column]
            ),
        }

        # -------------------------------------------------
        # Binary label
        # -------------------------------------------------

        if (
            self.binary_label_column
            in self.df.columns
        ):

            binary_value = row[
                self.binary_label_column
            ]

            if pd.notna(binary_value):

                sample["binary_label"] = torch.tensor(
                    int(binary_value),
                    dtype=torch.long,
                )

        # -------------------------------------------------
        # Metadata
        # -------------------------------------------------

        if self.return_metadata:

            metadata = {}

            for key, column in (
                self.metadata_columns.items()
            ):

                value = row[column]

                # -----------------------------------------
                # Preserve metadata values.
                #
                # Numerical values remain numerical.
                # Categorical values remain strings.
                #
                # Encoding should be performed by the
                # component that actually needs it.
                # -----------------------------------------

                if pd.isna(value):
                    metadata[key] = None
                else:
                    metadata[key] = value

            sample["metadata"] = metadata

        return sample


# =========================================================
# Dataset Builder
# =========================================================

def build_datasets(
    cfg,
) -> Tuple[
    SkinDataset,
    SkinDataset,
    SkinDataset,
]:
    """
    Build train, validation and test datasets.
    """

    (
        train_transform,
        val_transform,
        test_transform,
    ) = build_transforms(cfg)

    splits = cfg.dataset.splits

    train_dataset = SkinDataset(
        cfg,
        split=str(splits.train),
        transform=train_transform,
    )

    val_dataset = SkinDataset(
        cfg,
        split=str(splits.val),
        transform=val_transform,
    )

    test_dataset = SkinDataset(
        cfg,
        split=str(splits.test),
        transform=test_transform,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


# =========================================================
# Class Weights
# =========================================================

def compute_class_weights(
    dataset: SkinDataset,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for
    weighted cross entropy.

    Formula
    -------
        w_c = N / (C * n_c)

    where:

        N = total number of samples
        C = number of classes
        n_c = number of samples in class c
    """

    if dataset.task == "multiclass":

        labels = []

        for label_value in dataset.df[
            dataset.label_column
        ]:

            labels.append(
                dataset.label_to_index[
                    label_value
                ]
            )

    else:

        labels = (
            dataset.df[
                dataset.binary_label_column
            ]
            .astype(int)
            .tolist()
        )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    if labels.size == 0:
        raise ValueError(
            "Cannot compute class weights "
            "for an empty dataset."
        )

    counts = np.bincount(
        labels,
        minlength=dataset.num_classes,
    )

    counts = np.clip(
        counts,
        1,
        None,
    )

    weights = (
        len(labels)
        / (
            dataset.num_classes
            * counts
        )
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# =========================================================
# DataLoader Builder
# =========================================================

def build_dataloaders(cfg):
    """
    Build train, validation and test DataLoaders.
    """

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = build_datasets(cfg)

    # -----------------------------------------------------
    # Loader configuration
    # -----------------------------------------------------

    batch_size = int(
        cfg.dataset.batch_size
    )

    num_workers = int(
        cfg.dataset.num_workers
    )

    pin_memory = bool(
        cfg.dataset.pin_memory
    )

    persistent_workers = bool(
        cfg.dataset.persistent_workers
    )

    # persistent_workers is only valid when workers > 0
    persistent_workers = (
        persistent_workers
        and num_workers > 0
    )

    # -----------------------------------------------------
    # Reproducible generator
    # -----------------------------------------------------

    generator = torch.Generator()

    generator.manual_seed(
        int(cfg.seed)
    )

    # -----------------------------------------------------
    # Common DataLoader arguments
    # -----------------------------------------------------

    common_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "worker_init_fn": seed_worker,
    }

    if num_workers > 0:
        common_kwargs["prefetch_factor"] = 2

    # -----------------------------------------------------
    # Train loader
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,

        batch_size=batch_size,

        shuffle=bool(
            cfg.dataset.shuffle.train
        ),

        generator=generator,

        drop_last=bool(
            cfg.dataset.drop_last
        ),

        **common_kwargs,
    )

    # -----------------------------------------------------
    # Validation loader
    # -----------------------------------------------------

    val_loader = DataLoader(
        val_dataset,

        batch_size=batch_size,

        shuffle=bool(
            cfg.dataset.shuffle.val
        ),

        drop_last=False,

        **common_kwargs,
    )

    # -----------------------------------------------------
    # Test loader
    # -----------------------------------------------------

    test_loader = DataLoader(
        test_dataset,

        batch_size=int(
            cfg.inference.batch_size
        ),

        shuffle=bool(
            cfg.dataset.shuffle.test
        ),

        drop_last=False,

        **common_kwargs,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# =========================================================
# Dataset Information
# =========================================================

def dataset_info(
    dataset: SkinDataset,
) -> None:
    """
    Print dataset statistics.
    """

    print("=" * 70)

    print(
        f"Dataset        : "
        f"{dataset.dataset_name}"
    )

    print(
        f"Split          : "
        f"{dataset.split}"
    )

    print(
        f"Task           : "
        f"{dataset.task}"
    )

    print(
        f"Samples        : "
        f"{len(dataset)}"
    )

    print(
        f"Num classes    : "
        f"{dataset.num_classes}"
    )

    print()

    # -----------------------------------------------------
    # Class distribution
    # -----------------------------------------------------

    if dataset.task == "multiclass":

        print("Classes")

        counts = (
            dataset.df[
                dataset.label_column
            ]
            .value_counts()
            .sort_index()
        )

        for label, count in counts.items():

            index = dataset.label_to_index[
                label
            ]

            print(
                f"  [{index}] "
                f"{label}: {count}"
            )

    else:

        print("Binary Classes")

        counts = (
            dataset.df[
                dataset.binary_label_column
            ]
            .value_counts()
            .sort_index()
        )

        print(counts)

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    print()
    print("Metadata Columns")

    for key, column in (
        dataset.metadata_columns.items()
    ):

        print(
            f"  {key:15s} -> {column}"
        )

    # -----------------------------------------------------
    # Paths
    # -----------------------------------------------------

    print()
    print(
        f"Image root     : "
        f"{dataset.root}"
    )

    print(
        f"Metadata file  : "
        f"{dataset.metadata_file}"
    )

    print("=" * 70)


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(
        "config.yaml"
    )

    (
        train_loader,
        val_loader,
        test_loader,
    ) = build_dataloaders(cfg)

    # -----------------------------------------------------
    # Dataset information
    # -----------------------------------------------------

    print()
    print("TRAIN DATASET")
    dataset_info(
        train_loader.dataset
    )

    print()
    print("VALIDATION DATASET")
    dataset_info(
        val_loader.dataset
    )

    print()
    print("TEST DATASET")
    dataset_info(
        test_loader.dataset
    )

    # -----------------------------------------------------
    # Class weights
    # -----------------------------------------------------

    class_weights = compute_class_weights(
        train_loader.dataset
    )

    print()
    print("Class Weights")
    print(class_weights)

    # -----------------------------------------------------
    # Test one batch
    # -----------------------------------------------------

    batch = next(
        iter(train_loader)
    )

    print()
    print("Batch Keys")
    print(batch.keys())

    print()
    print("Image Shape")
    print(
        batch["image"].shape
    )

    print()
    print("Labels")
    print(
        batch["label"][:10]
    )

    if "binary_label" in batch:

        print()
        print("Binary Labels")
        print(
            batch["binary_label"][:10]
        )

    if "metadata" in batch:

        print()
        print("Metadata")

        print(
            batch["metadata"]
        )

    print()
    print("Done.")