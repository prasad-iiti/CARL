import json
import shutil
import time
from pathlib import Path

import pandas as pd
import torch
from torch import amp
from tqdm import tqdm

from src.evaluator import Evaluator
from src.utils import (
    AverageMeter,
    Logger,
    load_checkpoint,
    save_best_model,
    save_checkpoint,
)


class Trainer:
    """
    CARL training engine.

    Responsibilities
    ----------------
    - Experiment management
    - Training
    - Validation
    - Testing
    - CARL feedback scheduling
    - Mixed precision training
    - Gradient clipping
    - Learning-rate scheduling
    - Checkpointing
    - Early stopping
    - Training history
    - Evaluation artifact management
    """

    def __init__(
        self,
        cfg,
        model,
        loss_fn,
        device,
        optimizer=None,
        scheduler=None,
    ):
        self.cfg = cfg
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # ==================================================
        # Mixed Precision
        # ==================================================

        self.use_amp = bool(
            cfg.training.get("use_amp", True)
        )

        amp_device = (
            self.device.type
            if self.device.type in ["cuda", "cpu"]
            else "cuda"
        )

        self.scaler = amp.GradScaler(
            amp_device,
            enabled=self.use_amp and self.device.type == "cuda",
        )

        # ==================================================
        # Experiment
        # ==================================================

        self._prepare_experiment()

        self.logger = Logger(
            self.logs_dir,
            "train.log",
        )

        self.evaluator = Evaluator(
            cfg=self.cfg,
            output_dir=self.run_dir,
        )

        self._init_history()

        # ==================================================
        # Training State
        # ==================================================

        self.current_epoch = 0
        self.best_epoch = -1
        self.patience_counter = 0

        self.history = []

        mode = str(
            self.cfg.checkpoint.mode
        ).lower()

        if mode == "max":
            self.best_score = -float("inf")
        elif mode == "min":
            self.best_score = float("inf")
        else:
            raise ValueError(
                f"Unknown checkpoint mode: {mode}"
            )

    # ======================================================
    # Experiment Directories
    # ======================================================

    def _prepare_experiment(self):

        dataset = str(
            self.cfg.dataset.name
        )

        backbone = str(
            self.cfg.backbone.name
        )

        experiment = str(
            self.cfg.experiment.name
        )

        timestamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        run_name = (
            f"{timestamp}_"
            f"{dataset}_"
            f"{backbone}_"
            f"{experiment}"
        )

        self.run_dir = (
            Path("outputs")
            / "experiments"
            / run_name
        )

        self.logs_dir = (
            self.run_dir / "logs"
        )

        self.checkpoint_dir = (
            self.run_dir / "checkpoints"
        )

        self.metrics_dir = (
            self.run_dir / "metrics"
        )

        self.prediction_dir = (
            self.run_dir / "predictions"
        )

        self.artifact_dir = (
            self.run_dir / "artifacts"
        )

        directories = [
            self.logs_dir,
            self.checkpoint_dir,
            self.metrics_dir,
            self.prediction_dir,
            self.artifact_dir,
        ]

        for directory in directories:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        # Optional config backup
        config_path = self.cfg.get(
            "_config_path",
            None,
        )

        if config_path is not None:
            config_path = Path(
                str(config_path)
            )

            if config_path.exists():
                shutil.copy(
                    config_path,
                    self.run_dir / "config.yaml",
                )

    # ======================================================
    # History
    # ======================================================

    def _init_history(self):

        self.history_file = (
            self.metrics_dir
            / "history.csv"
        )

        columns = [
            "epoch",
            "train_loss",
            "val_loss",
            "train_accuracy",
            "val_accuracy",
            "focal_loss",
            "metric_loss",
            "learning_rate",
            "feedback_strength",
            "epoch_time",
            "samples_per_second",
        ]

        if not self.history_file.exists():

            pd.DataFrame(
                columns=columns
            ).to_csv(
                self.history_file,
                index=False,
            )

    # ======================================================
    # Logging
    # ======================================================

    def log(self, message):

        self.logger.log(
            message
        )

    # ======================================================
    # Learning Rate
    # ======================================================

    @property
    def learning_rate(self):

        if self.optimizer is None:
            return 0.0

        return float(
            self.optimizer.param_groups[0]["lr"]
        )

    # ======================================================
    # CARL Feedback Schedule
    # ======================================================

    def compute_feedback_strength(self):

        carl_cfg = self.cfg.carl

        if not bool(
            carl_cfg.get(
                "training_feedback",
                False,
            )
        ):
            return 0.0

        start_epoch = int(
            carl_cfg.get(
                "feedback_start_epoch",
                0,
            )
        )

        strength = float(
            carl_cfg.get(
                "feedback_strength",
                0.0,
            )
        )

        if self.current_epoch + 1 < start_epoch:
            return 0.0

        return strength

    # ======================================================
    # Train One Epoch
    # ======================================================

    def train_one_epoch(
        self,
        train_loader,
    ):

        self.model.train()

        start_time = time.time()

        total_iterations = len(
            train_loader
        )

        train_loss = AverageMeter()
        focal_meter = AverageMeter()
        metric_meter = AverageMeter()
        train_acc = AverageMeter()

        processed_samples = 0

        feedback_strength = (
            self.compute_feedback_strength()
        )

        pbar = tqdm(
            enumerate(train_loader),
            total=total_iterations,
            desc=(
                f"Epoch "
                f"{self.current_epoch + 1}"
            ),
            leave=False,
        )

        for iteration, batch in pbar:

            images = batch["image"].to(
                self.device,
                non_blocking=True,
            )

            labels = batch["label"].to(
                self.device,
                non_blocking=True,
            )

            # ------------------------------------------
            # Metadata
            # ------------------------------------------

            metadata = batch.get(
                "metadata",
                {},
            )

            moved_metadata = {}

            for key, value in metadata.items():

                if torch.is_tensor(value):

                    moved_metadata[key] = value.to(
                        self.device,
                        non_blocking=True,
                    )

                else:

                    moved_metadata[key] = value

            # ------------------------------------------
            # Optimizer
            # ------------------------------------------

            self.optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------
            # Forward + Loss
            # ------------------------------------------

            with amp.autocast(
                device_type=self.device.type,
                enabled=(
                    self.use_amp
                    and self.device.type == "cuda"
                ),
            ):

                outputs = self.model(
                    images,
                    epoch=self.current_epoch,
                )

                losses = self.loss_fn(
                    outputs=outputs,
                    labels=labels,
                    metadata=moved_metadata,
                )

                total_loss = losses["loss"]

            # ------------------------------------------
            # Backward
            # ------------------------------------------

            self.scaler.scale(
                total_loss
            ).backward()

            self.scaler.unscale_(
                self.optimizer
            )

            grad_clip = float(
                self.cfg.training.get(
                    "grad_clip",
                    0.0,
                )
            )

            if grad_clip > 0:

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=grad_clip,
                )

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

            # ------------------------------------------
            # Statistics
            # ------------------------------------------

            batch_size = labels.size(0)

            processed_samples += batch_size

            train_loss.update(
                total_loss.item(),
                batch_size,
            )

            focal_loss = losses.get(
                "focal_loss",
                0.0,
            )

            metric_loss = losses.get(
                "metric_loss",
                0.0,
            )

            if torch.is_tensor(
                focal_loss
            ):
                focal_loss = focal_loss.item()

            if torch.is_tensor(
                metric_loss
            ):
                metric_loss = metric_loss.item()

            focal_meter.update(
                focal_loss,
                batch_size,
            )

            metric_meter.update(
                metric_loss,
                batch_size,
            )

            # ------------------------------------------
            # Classification accuracy
            # ------------------------------------------

            logits = outputs["logits"]

            predictions = logits.argmax(
                dim=1
            )

            accuracy = (
                predictions == labels
            ).float().mean().item()

            train_acc.update(
                accuracy,
                batch_size,
            )

            # ------------------------------------------
            # Progress bar
            # ------------------------------------------

            pbar.set_postfix(
                {
                    "Loss":
                        f"{train_loss.avg:.4f}",

                    "Focal":
                        f"{focal_meter.avg:.4f}",

                    "Metric":
                        f"{metric_meter.avg:.4f}",

                    "Acc":
                        f"{train_acc.avg:.4f}",

                    "LR":
                        f"{self.learning_rate:.2e}",

                    "FB":
                        f"{feedback_strength:.3f}",
                }
            )

        # ==================================================
        # Scheduler
        # ==================================================

        if self.scheduler is not None:

            self.scheduler.step()

        epoch_time = (
            time.time() - start_time
        )

        samples_per_second = (
            processed_samples / epoch_time
            if epoch_time > 0
            else 0.0
        )

        return {
            "loss":
                train_loss.avg,

            "focal_loss":
                focal_meter.avg,

            "metric_loss":
                metric_meter.avg,

            "accuracy":
                train_acc.avg,

            "learning_rate":
                self.learning_rate,

            "feedback_strength":
                feedback_strength,

            "epoch_time":
                epoch_time,

            "samples_per_second":
                samples_per_second,
        }

    # ======================================================
    # Validation
    # ======================================================

    @torch.inference_mode()
    def validate(
        self,
        val_loader,
    ):

        self.model.eval()

        results = self.evaluator.evaluate(
            model=self.model,
            loader=val_loader,
            loss_fn=self.loss_fn,
            device=self.device,
            split="val",
        )

        return results["metrics"]

    # ======================================================
    # Test
    # ======================================================

    @torch.inference_mode()
    def test(
        self,
        test_loader,
    ):

        self.model.eval()

        results = self.evaluator.evaluate(
            model=self.model,
            loader=test_loader,
            loss_fn=self.loss_fn,
            device=self.device,
            split="test",
        )

        return results["metrics"]

    # ======================================================
    # Save History
    # ======================================================

    def save_history(
        self,
        train_metrics,
        val_metrics,
    ):

        row = {
            "epoch":
                self.current_epoch + 1,

            "train_loss":
                train_metrics["loss"],

            "val_loss":
                val_metrics["loss"],

            "train_accuracy":
                train_metrics["accuracy"],

            "val_accuracy":
                val_metrics.get(
                    "accuracy",
                    0.0,
                ),

            "focal_loss":
                train_metrics["focal_loss"],

            "metric_loss":
                train_metrics["metric_loss"],

            "learning_rate":
                train_metrics["learning_rate"],

            "feedback_strength":
                train_metrics[
                    "feedback_strength"
                ],

            "epoch_time":
                train_metrics["epoch_time"],

            "samples_per_second":
                train_metrics[
                    "samples_per_second"
                ],
        }

        self.history.append(
            row
        )

        pd.DataFrame(
            [row]
        ).to_csv(
            self.history_file,
            mode="a",
            header=False,
            index=False,
        )

    # ======================================================
    # Epoch Logging
    # ======================================================

    def log_epoch(
        self,
        train_metrics,
        val_metrics,
    ):

        self.log(
            f"Epoch "
            f"[{self.current_epoch + 1}] | "
            f"Train Loss: "
            f"{train_metrics['loss']:.4f} | "
            f"Val Loss: "
            f"{val_metrics['loss']:.4f} | "
            f"Train Acc: "
            f"{train_metrics['accuracy']:.4f} | "
            f"Val Acc: "
            f"{val_metrics.get('accuracy', 0.0):.4f} | "
            f"Focal: "
            f"{train_metrics['focal_loss']:.4f} | "
            f"Metric: "
            f"{train_metrics['metric_loss']:.4f} | "
            f"LR: "
            f"{train_metrics['learning_rate']:.2e} | "
            f"FB: "
            f"{train_metrics['feedback_strength']:.3f} | "
            f"Time: "
            f"{train_metrics['epoch_time']:.2f}s"
        )

    # ======================================================
    # Validation Metrics
    # ======================================================

    def save_validation_metrics(
        self,
        metrics,
    ):

        path = (
            self.metrics_dir
            / "val_metrics.jsonl"
        )

        record = {
            "epoch":
                self.current_epoch + 1,
            **metrics,
        }

        with path.open(
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    record,
                    default=str,
                )
            )

            f.write("\n")

    # ======================================================
    # Test Metrics
    # ======================================================

    def save_test_metrics(
        self,
        metrics,
    ):

        path = (
            self.metrics_dir
            / "test_metrics.json"
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metrics,
                f,
                indent=4,
                default=str,
            )

    # ======================================================
    # Train
    # ======================================================

    def train(
        self,
        train_loader,
        val_loader,
    ):

        epochs = int(
            self.cfg.training.epochs
        )

        self.log("=" * 80)
        self.log("Starting CARL training")
        self.log(
            f"Dataset   : "
            f"{self.cfg.dataset.name}"
        )
        self.log(
            f"Backbone  : "
            f"{self.cfg.backbone.name}"
        )
        self.log(
            f"Epochs    : {epochs}"
        )
        self.log(
            f"Device    : {self.device}"
        )
        self.log(
            f"AMP       : {self.use_amp}"
        )
        self.log("=" * 80)

        monitor = str(
            self.cfg.metrics.primary
        )

        mode = str(
            self.cfg.checkpoint.mode
        ).lower()

        for self.current_epoch in range(
            epochs
        ):

            train_metrics = (
                self.train_one_epoch(
                    train_loader
                )
            )

            val_metrics = (
                self.validate(
                    val_loader
                )
            )

            self.save_history(
                train_metrics,
                val_metrics,
            )

            self.save_validation_metrics(
                val_metrics
            )

            self.log_epoch(
                train_metrics,
                val_metrics,
            )

            if monitor not in val_metrics:

                raise KeyError(
                    f"Validation metrics "
                    f"do not contain "
                    f"'{monitor}'. "
                    f"Available: "
                    f"{list(val_metrics.keys())}"
                )

            metric = float(
                val_metrics[monitor]
            )

            # ------------------------------------------
            # Save last
            # ------------------------------------------

            if bool(
                self.cfg.checkpoint.get(
                    "save_last",
                    False,
                )
            ):

                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=self.current_epoch,
                    score=metric,
                    checkpoint_dir=self.checkpoint_dir,
                )

            # ------------------------------------------
            # Best model
            # ------------------------------------------

            if mode == "max":

                improved = (
                    metric > self.best_score
                )

            elif mode == "min":

                improved = (
                    metric < self.best_score
                )

            else:

                raise ValueError(
                    f"Unknown checkpoint mode: "
                    f"{mode}"
                )

            if improved:

                self.best_score = metric
                self.best_epoch = (
                    self.current_epoch
                )
                self.patience_counter = 0

                if bool(
                    self.cfg.checkpoint.get(
                        "save_best",
                        True,
                    )
                ):

                    save_best_model(
                        model=self.model,
                        epoch=self.current_epoch,
                        score=self.best_score,
                        checkpoint_dir=self.checkpoint_dir,
                    )

                self.log(
                    f"Best model updated | "
                    f"Epoch "
                    f"{self.best_epoch + 1} | "
                    f"{monitor}="
                    f"{self.best_score:.4f}"
                )

            else:

                self.patience_counter += 1

            # ------------------------------------------
            # Early stopping
            # ------------------------------------------

            patience = self.cfg.training.get(
                "patience",
                None,
            )

            if (
                patience is not None
                and self.patience_counter >= int(
                    patience
                )
            ):

                self.log(
                    "Early stopping triggered."
                )

                break

        self.log("=" * 80)
        self.log(
            f"CARL training completed | "
            f"Best {monitor}: "
            f"{self.best_score:.4f} | "
            f"Epoch: "
            f"{self.best_epoch + 1}"
        )
        self.log("=" * 80)

        return {
            "best_score":
                self.best_score,

            "best_epoch":
                self.best_epoch,
        }

    # ======================================================
    # Final Evaluation
    # ======================================================

    @torch.inference_mode()
    def evaluate(
        self,
        test_loader,
        checkpoint_path=None,
    ):

        if checkpoint_path is None:

            checkpoint_path = (
                self.checkpoint_dir
                / "best_model.pth"
            )

        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{checkpoint_path}"
            )

        load_checkpoint(
            path=checkpoint_path,
            model=self.model,
            map_location=self.device,
        )

        self.model.eval()

        test_metrics = self.test(
            test_loader
        )

        self.save_test_metrics(
            test_metrics
        )

        self.log("=" * 80)
        self.log("Final CARL Test Results")
        self.log("=" * 80)

        for key, value in test_metrics.items():

            if isinstance(
                value,
                (int, float),
            ):

                self.log(
                    f"{key}: "
                    f"{value:.4f}"
                )

        self.log("=" * 80)

        return test_metrics


# ==========================================================
# Sanity Check
# ==========================================================

if __name__ == "__main__":

    from omegaconf import OmegaConf

    try:
        from src.utils import get_device
    except ModuleNotFoundError:
        from utils import get_device

    print("=" * 70)
    print("CARL TRAINER SANITY CHECK")
    print("=" * 70)

    cfg = OmegaConf.load(
        "/home/prasad/CARL/config.yaml"
    )

    device = get_device(cfg)

    print(
        f"Dataset : "
        f"{cfg.dataset.name}"
    )

    print(
        f"Backbone: "
        f"{cfg.backbone.name}"
    )

    print(
        f"Device  : "
        f"{device}"
    )

    print()

    # ------------------------------------------------------
    # Structural checks
    # ------------------------------------------------------

    required_sections = [
        "dataset",
        "backbone",
        "carl",
        "loss",
        "optimizer",
        "scheduler",
        "training",
        "metrics",
        "checkpoint",
        "output",
    ]

    for section in required_sections:

        assert section in cfg, (
            f"Missing config section: "
            f"{section}"
        )

    print(
        "Configuration sections : OK"
    )

    assert "name" in cfg.backbone
    assert "embedding_dim" in cfg.carl
    assert "metric_weight" in cfg.loss

    print(
        "CARL configuration       : OK"
    )

    # ------------------------------------------------------
    # Feedback schedule
    # ------------------------------------------------------

    print(
        "Feedback schedule        : "
        "checking"
    )

    # Lightweight object construction
    trainer = object.__new__(
        Trainer
    )

    trainer.cfg = cfg
    trainer.current_epoch = 0

    feedback_before = (
        Trainer.compute_feedback_strength(
            trainer
        )
    )

    trainer.current_epoch = int(
        cfg.carl.feedback_start_epoch
    ) - 1

    feedback_at_start = (
        Trainer.compute_feedback_strength(
            trainer
        )
    )

    print(
        f"Feedback before start    : "
        f"{feedback_before:.3f}"
    )

    print(
        f"Feedback at start        : "
        f"{feedback_at_start:.3f}"
    )

    print(
        "Feedback schedule        : OK"
    )

    print()

    print(
        "CARL trainer sanity check "
        "completed."
    )

    print("=" * 70)