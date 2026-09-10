"""
=========================================================
CARL
Utility Functions

Contains
--------
1. AverageMeter
2. Reproducibility
3. Device Utilities
4. Parameter Utilities
5. Time Formatting
6. Experiment Logging
7. Output Directory Utilities
8. JSON Utilities
9. Checkpoint Utilities
10. CARL Utility Sanity Check
=========================================================
"""

from __future__ import annotations

import csv
import json
import os
import random
import time

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ==========================================================
# Average Meter
# ==========================================================

class AverageMeter:
    """
    Computes and stores a running average.
    Parameters
    ----------
    None
    Example
    -------
    meter = AverageMeter()
    meter.update(loss, batch_size)
    print(meter.avg)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the meter."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(
        self,
        value,
        n: int = 1,
    ):
        """
        Update the meter.

        Parameters
        ----------
        value : float
            Current value.

        n : int
            Number of samples represented by value.
        """

        if torch.is_tensor(value):
            value = value.detach().item()

        value = float(value)
        n = int(n)

        self.val = value
        self.sum += value * n
        self.count += n

        if self.count > 0:
            self.avg = self.sum / self.count


# ==========================================================
# Reproducibility
# ==========================================================

def set_seed(
    seed: int,
    deterministic: bool = False,
):
    """
    Set random seeds for reproducible experiments.

    Parameters
    ----------
    seed : int
        Random seed.

    deterministic : bool
        Whether to force deterministic CUDA/PyTorch behavior.
    """

    seed = int(seed)

    random.seed(seed)

    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    if deterministic:

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False

        torch.use_deterministic_algorithms(True)

    else:

        torch.backends.cudnn.deterministic = False

        torch.backends.cudnn.benchmark = True

        torch.use_deterministic_algorithms(False)


# ==========================================================
# Device Utilities
# ==========================================================

def get_device(cfg):
    """
    Resolve the training device from configuration.

    Supported
    ---------
    auto
    cpu
    cuda
    cuda:0
    etc.
    """

    device_name = str(cfg.device).lower()

    if device_name == "auto":

        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return torch.device(device_name)


def move_to_device(
    batch: Any,
    device: torch.device,
):
    """
    Recursively move tensors contained in an object to device.

    Supports
    --------
    Tensor
    dict
    list
    tuple
    """

    if torch.is_tensor(batch):

        return batch.to(
            device,
            non_blocking=True,
        )

    if isinstance(batch, dict):

        return {
            key: move_to_device(
                value,
                device,
            )
            for key, value in batch.items()
        }

    if isinstance(batch, list):

        return [
            move_to_device(
                value,
                device,
            )
            for value in batch
        ]

    if isinstance(batch, tuple):

        return tuple(
            move_to_device(
                value,
                device,
            )
            for value in batch
        )

    return batch


# ==========================================================
# Parameter Utilities
# ==========================================================

def count_parameters(
    model,
    trainable_only: bool = False,
):
    """
    Count model parameters.

    Parameters
    ----------
    model : torch.nn.Module

    trainable_only : bool
        If True, count only parameters with
        requires_grad=True.

    Returns
    -------
    int
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if (
            parameter.requires_grad
            if trainable_only
            else True
        )
    )


def print_model_summary(model):
    """
    Print model parameter statistics.
    """

    total = count_parameters(model)

    trainable = count_parameters(
        model,
        trainable_only=True,
    )

    frozen = total - trainable

    print("=" * 70)

    print("Model Summary")

    print("=" * 70)

    print(
        f"Total Parameters     : {total:,}"
    )

    print(
        f"Trainable Parameters : {trainable:,}"
    )

    print(
        f"Frozen Parameters    : {frozen:,}"
    )

    print("=" * 70)


# ==========================================================
# Time Utilities
# ==========================================================

def format_time(seconds):
    """
    Convert seconds into HH:MM:SS.
    """

    seconds = int(seconds)

    hours = seconds // 3600

    minutes = (
        seconds % 3600
    ) // 60

    seconds = seconds % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


# ==========================================================
# Logger
# ==========================================================

class Logger:
    """
    Lightweight CARL experiment logger.

    Supports
    --------
    Console logging
    Text log file
    CSV metrics
    """

    def __init__(
        self,
        log_dir,
        log_file: str = "train.log",
    ):

        self.log_dir = Path(log_dir)

        self.log_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.log_path = (
            self.log_dir / log_file
        )

        self.csv_path = (
            self.log_dir / "metrics.csv"
        )

        self.fieldnames = None

        self.header_written = False

        # --------------------------------------------------
        # Recover existing CSV header
        # --------------------------------------------------

        if self.csv_path.exists():

            with open(
                self.csv_path,
                "r",
                newline="",
                encoding="utf-8",
            ) as file:

                reader = csv.reader(file)

                try:

                    self.fieldnames = next(
                        reader
                    )

                    self.header_written = True

                except StopIteration:

                    pass

    # ------------------------------------------------------
    # Console + text log
    # ------------------------------------------------------

    def log(self, message):
        """
        Print message and write it to train.log.
        """

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        line = (
            f"[{timestamp}] "
            f"{message}"
        )

        print(line)

        with open(
            self.log_path,
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                line + "\n"
            )

    # ------------------------------------------------------

    def separator(self):
        """Write a separator."""

        self.log("=" * 80)

    # ------------------------------------------------------

    def blank(self):
        """Write a blank line."""

        self.log("")

    # ------------------------------------------------------

    def log_metrics(
        self,
        metrics: dict,
    ):
        """
        Append metrics to metrics.csv.

        Parameters
        ----------
        metrics : dict
            Dictionary containing scalar metrics.
        """

        cleaned = {}

        for key, value in metrics.items():

            if torch.is_tensor(value):

                if value.numel() == 1:

                    value = value.detach().item()

            if isinstance(
                value,
                np.floating,
            ):

                value = float(value)

            elif isinstance(
                value,
                np.integer,
            ):

                value = int(value)

            cleaned[key] = value

        metrics = cleaned

        # --------------------------------------------------
        # Create CSV header
        # --------------------------------------------------

        if not self.header_written:

            self.fieldnames = list(
                metrics.keys()
            )

            with open(
                self.csv_path,
                "w",
                newline="",
                encoding="utf-8",
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=self.fieldnames,
                )

                writer.writeheader()

            self.header_written = True

        # --------------------------------------------------
        # Handle newly appearing fields
        # --------------------------------------------------

        current_fields = list(
            metrics.keys()
        )

        if set(current_fields) != set(
            self.fieldnames
        ):

            existing_rows = []

            with open(
                self.csv_path,
                "r",
                newline="",
                encoding="utf-8",
            ) as file:

                reader = csv.DictReader(file)

                existing_rows = list(reader)

            all_fields = list(
                dict.fromkeys(
                    self.fieldnames
                    + current_fields
                )
            )

            with open(
                self.csv_path,
                "w",
                newline="",
                encoding="utf-8",
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=all_fields,
                )

                writer.writeheader()

                writer.writerows(
                    existing_rows
                )

            self.fieldnames = all_fields

        # --------------------------------------------------
        # Append current metrics
        # --------------------------------------------------

        with open(
            self.csv_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=self.fieldnames,
            )

            writer.writerow(metrics)


# ==========================================================
# Output Directory Utilities
# ==========================================================

def create_output_directories(
    root,
):
    """
    Create the standard CARL experiment directories.

    Returns
    -------
    dict
        Mapping of directory names to Path objects.
    """

    root = Path(root)

    directories = {

        "logs": root / "logs",

        "checkpoints": (
            root / "checkpoints"
        ),

        "metrics": (
            root / "metrics"
        ),

        "plots": (
            root / "plots"
        ),

        "predictions": (
            root / "predictions"
        ),

        "embeddings": (
            root / "embeddings"
        ),

        "attention": (
            root / "attention"
        ),

        "feature_weights": (
            root / "feature_weights"
        ),

        "artifacts": (
            root / "artifacts"
        ),
    }

    for directory in directories.values():

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    return directories


# ==========================================================
# JSON Utilities
# ==========================================================

def save_json(
    data,
    path,
):
    """
    Save a Python object as JSON.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            default=str,
        )


def load_json(path):
    """
    Load JSON from disk.
    """

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ==========================================================
# Checkpoint Helpers
# ==========================================================

def _build_checkpoint(
    model,
    optimizer=None,
    scheduler=None,
    epoch=0,
    score=0.0,
    config=None,
):
    """
    Build a CARL checkpoint dictionary.
    """

    checkpoint = {

        "epoch": int(epoch),

        "score": float(score),

        "model_state_dict":
            model.state_dict(),
    }

    if optimizer is not None:

        checkpoint[
            "optimizer_state_dict"
        ] = optimizer.state_dict()

    if scheduler is not None:

        checkpoint[
            "scheduler_state_dict"
        ] = scheduler.state_dict()

    if config is not None:

        try:

            from omegaconf import (
                OmegaConf,
            )

            checkpoint[
                "config"
            ] = OmegaConf.to_container(
                config,
                resolve=True,
            )

        except Exception:

            checkpoint[
                "config"
            ] = str(config)

    return checkpoint


# ==========================================================
# Save Latest Checkpoint
# ==========================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    score,
    checkpoint_dir,
    config=None,
):
    """
    Save the latest CARL checkpoint.

    File
    ----
    last_model.pth
    """

    checkpoint_dir = Path(
        checkpoint_dir
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = _build_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        score=score,
        config=config,
    )

    checkpoint_path = (
        checkpoint_dir
        / "last_model.pth"
    )

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    save_json(
        {
            "epoch": int(epoch),
            "score": float(score),
            "saved_at":
                datetime.now().isoformat(),
        },
        checkpoint_dir
        / "last_model_meta.json",
    )

    return checkpoint_path


# ==========================================================
# Save Best Model
# ==========================================================

def save_best_model(
    model,
    epoch,
    score,
    checkpoint_dir,
    optimizer=None,
    scheduler=None,
    config=None,
):
    """
    Save the best-performing CARL model.

    File
    ----
    best_model.pth
    """

    checkpoint_dir = Path(
        checkpoint_dir
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = _build_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        score=score,
        config=config,
    )

    checkpoint_path = (
        checkpoint_dir
        / "best_model.pth"
    )

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    save_json(
        {
            "epoch": int(epoch),
            "score": float(score),
            "saved_at":
                datetime.now().isoformat(),
        },
        checkpoint_dir
        / "best_model_meta.json",
    )

    return checkpoint_path


# ==========================================================
# Load Checkpoint
# ==========================================================

def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    map_location="cpu",
):
    """
    Load a CARL checkpoint.

    Returns
    -------
    model
    optimizer
    scheduler
    epoch
    score
    """

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        # Support raw state_dict
        model.load_state_dict(
            checkpoint
        )

    # ------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------

    if (
        optimizer is not None
        and isinstance(
            checkpoint,
            dict,
        )
        and "optimizer_state_dict"
        in checkpoint
    ):

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

    # ------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------

    if (
        scheduler is not None
        and isinstance(
            checkpoint,
            dict,
        )
        and "scheduler_state_dict"
        in checkpoint
    ):

        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

    epoch = (
        checkpoint.get(
            "epoch",
            0,
        )
        if isinstance(
            checkpoint,
            dict,
        )
        else 0
    )

    score = (
        checkpoint.get(
            "score",
            0.0,
        )
        if isinstance(
            checkpoint,
            dict,
        )
        else 0.0
    )

    return (
        model,
        optimizer,
        scheduler,
        epoch,
        score,
    )


# ==========================================================
# CARL Utility Sanity Check
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "CARL UTILS SANITY CHECK"
    )

    print("=" * 70)

    # ------------------------------------------------------
    # Device
    # ------------------------------------------------------

    class DummyConfig:

        device = "auto"

    cfg = DummyConfig()

    device = get_device(cfg)

    print(
        f"Device : {device}"
    )

    # ------------------------------------------------------
    # AverageMeter
    # ------------------------------------------------------

    meter = AverageMeter()

    meter.update(
        2.0,
        2,
    )

    meter.update(
        4.0,
        2,
    )

    print(
        f"AverageMeter avg : "
        f"{meter.avg:.4f}"
    )

    assert abs(
        meter.avg - 3.0
    ) < 1e-6

    # ------------------------------------------------------
    # Seed
    # ------------------------------------------------------

    set_seed(
        42,
        deterministic=False,
    )

    random_a = torch.randn(3)

    set_seed(
        42,
        deterministic=False,
    )

    random_b = torch.randn(3)

    assert torch.allclose(
        random_a,
        random_b,
    )

    print(
        "Seed reproducibility : OK"
    )

    # ------------------------------------------------------
    # Device movement
    # ------------------------------------------------------

    batch = {

        "image": torch.randn(
            2,
            3,
            224,
            224,
        ),

        "label": torch.tensor(
            [0, 1]
        ),

        "nested": [
            torch.tensor(
                [1.0, 2.0]
            )
        ],
    }

    moved = move_to_device(
        batch,
        device,
    )

    assert (
        moved["image"].device.type
        == device.type
    )

    assert (
        moved["label"].device.type
        == device.type
    )

    print(
        "move_to_device : OK"
    )

    # ------------------------------------------------------
    # Parameter counting
    # ------------------------------------------------------

    model = torch.nn.Sequential(

        torch.nn.Linear(
            10,
            20,
        ),

        torch.nn.ReLU(),

        torch.nn.Linear(
            20,
            5,
        ),
    )

    total = count_parameters(
        model
    )

    trainable = count_parameters(
        model,
        trainable_only=True,
    )

    print(
        f"Total parameters     : "
        f"{total:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable:,}"
    )

    assert total == trainable

    print(
        "Parameter counting : OK"
    )

    # ------------------------------------------------------
    # Time formatting
    # ------------------------------------------------------

    formatted = format_time(
        3661
    )

    print(
        f"Formatted time : "
        f"{formatted}"
    )

    assert formatted == "01:01:01"

    # ------------------------------------------------------
    # Temporary output directory
    # ------------------------------------------------------

    sanity_dir = (
        Path("outputs")
        / "utils_sanity_check"
    )

    directories = (
        create_output_directories(
            sanity_dir
        )
    )

    assert directories[
        "logs"
    ].exists()

    assert directories[
        "checkpoints"
    ].exists()

    assert directories[
        "embeddings"
    ].exists()

    assert directories[
        "feature_weights"
    ].exists()

    print(
        "Output directories : OK"
    )

    # ------------------------------------------------------
    # Logger
    # ------------------------------------------------------

    logger = Logger(
        directories["logs"]
    )

    logger.log(
        "CARL utility sanity check"
    )

    logger.log_metrics(
        {
            "epoch": 1,
            "loss": 1.2345,
            "balanced_accuracy":
                0.8123,
        }
    )

    assert (
        logger.log_path.exists()
    )

    assert (
        logger.csv_path.exists()
    )

    print(
        "Logger : OK"
    )

    # ------------------------------------------------------
    # JSON
    # ------------------------------------------------------

    json_path = (
        directories["artifacts"]
        / "sanity.json"
    )

    save_json(
        {
            "dataset": "hiba",
            "num_classes": 10,
        },
        json_path,
    )

    loaded = load_json(
        json_path
    )

    assert (
        loaded["dataset"]
        == "hiba"
    )

    assert (
        loaded["num_classes"]
        == 10
    )

    print(
        "JSON utilities : OK"
    )

    # ------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    checkpoint_path = (
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=1,
            score=0.75,
            checkpoint_dir=
                directories[
                    "checkpoints"
                ],
        )
    )

    assert checkpoint_path.exists()

    # ------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------

    restored_model = (
        torch.nn.Sequential(

            torch.nn.Linear(
                10,
                20,
            ),

            torch.nn.ReLU(),

            torch.nn.Linear(
                20,
                5,
            ),
        )
    )

    restored_optimizer = (
        torch.optim.AdamW(
            restored_model.parameters(),
            lr=1e-3,
        )
    )

    (
        restored_model,
        restored_optimizer,
        _,
        restored_epoch,
        restored_score,
    ) = load_checkpoint(
        checkpoint_path,
        restored_model,
        optimizer=
            restored_optimizer,
        scheduler=None,
        map_location=device,
    )

    assert restored_epoch == 1

    assert abs(
        restored_score - 0.75
    ) < 1e-8

    print(
        "Checkpoint save/load : OK"
    )

    # ------------------------------------------------------
    # Final
    # ------------------------------------------------------

    print()

    print(
        "CARL utility sanity "
        "check completed."
    )

    print("=" * 70)