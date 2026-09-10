"""
=============================================================
CARL Ablation Study
=============================================================

Runs controlled ablation experiments for the CARL framework.

Experiments
-----------
1. Baseline
2. CA-AS3
3. CAFA
4. CA-AS3 + CAFA
5. Full CARL

For every experiment this script:

    - creates an independent configuration
    - builds dataloaders
    - builds the corresponding model
    - builds the appropriate loss
    - trains the model
    - evaluates the best model on the test set
    - saves the experiment configuration
    - appends final metrics to CSV
    - generates an overall summary

The only difference between experiments is the enabled
architectural component(s).

=============================================================
"""

import copy
import csv
import json
import time
from pathlib import Path

import torch
import torch.optim as optim
from omegaconf import OmegaConf

from src.data import build_dataloaders, compute_class_weights
from src.losses import Losses
from src.model import build_model
from src.trainer import Trainer
from src.utils import set_seed


# ============================================================
# Paths
# ============================================================

CONFIG_FILE = Path("config.yaml")

OUTPUT_DIR = Path("outputs")

ABLATION_DIR = (
    OUTPUT_DIR / "experiments" / "ablation"
)

RESULTS_CSV = (
    OUTPUT_DIR / "ablation_results.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR / "ablation_summary.json"
)

SUMMARY_CSV = (
    OUTPUT_DIR / "ablation_summary.csv"
)


# ============================================================
# Ablation Experiments
# ============================================================

ABLATIONS = [

    {
        "name": "baseline",
        "description": "ResNet50 baseline",

        "ca_as3": False,
        "cafa": False,
        "carl": False,
    },

    {
        "name": "ca_as3",
        "description": "ResNet50 + CA-AS3",

        "ca_as3": True,
        "cafa": False,
        "carl": False,
    },

    {
        "name": "cafa",
        "description": "ResNet50 + CAFA",

        "ca_as3": False,
        "cafa": True,
        "carl": False,
    },

    {
        "name": "ca_as3_cafa",
        "description": "ResNet50 + CA-AS3 + CAFA",

        "ca_as3": True,
        "cafa": True,
        "carl": False,
    },

    {
        "name": "carl",
        "description": "Full CARL",

        "ca_as3": True,
        "cafa": True,
        "carl": True,
    },
]


# ============================================================
# Utility
# ============================================================

def safe_float(value):
    """
    Convert a metric value into a CSV/JSON-safe float.
    """

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ============================================================
# Configuration
# ============================================================

def load_config():
    """
    Load the main CARL configuration.
    """

    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_FILE}"
        )

    cfg = OmegaConf.load(CONFIG_FILE)

    cfg["_config_path"] = str(CONFIG_FILE)

    return cfg


def prepare_config(
    base_cfg,
    experiment,
):
    """
    Create an independent configuration for one
    ablation experiment.
    """

    cfg = copy.deepcopy(base_cfg)

    # --------------------------------------------------------
    # Experiment name
    # --------------------------------------------------------

    cfg.experiment.name = (
        f"ablation_{experiment['name']}"
    )

    # --------------------------------------------------------
    # Enable / disable architectural components
    # --------------------------------------------------------

    cfg.ca_as3.enabled = (
        experiment["ca_as3"]
    )

    cfg.cafa.enabled = (
        experiment["cafa"]
    )

    cfg.carl.enabled = (
        experiment["carl"]
    )

    # --------------------------------------------------------
    # CARL metric loss
    #
    # When CARL is disabled, metric learning must also
    # be disabled.
    # --------------------------------------------------------

    if experiment["carl"]:

        cfg.loss.metric_weight = (
            base_cfg.loss.metric_weight
        )

    else:

        cfg.loss.metric_weight = 0.0

    # --------------------------------------------------------
    # Experiment-specific output directory
    # --------------------------------------------------------

    cfg.output.save_dir = str(
        ABLATION_DIR / experiment["name"]
    )

    return cfg


# ============================================================
# Results File
# ============================================================

def initialize_results_file():
    """
    Create the ablation results CSV if it does not exist.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if RESULTS_CSV.exists():
        return

    with open(
        RESULTS_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([

            "Timestamp",

            "Experiment",

            "Description",

            "Dataset",

            "Backbone",

            "Epochs",

            "Batch Size",

            "CA_AS3",

            "CAFA",

            "CARL",

            "Metric Weight",

            "Accuracy",

            "Balanced Accuracy",

            "Precision Macro",

            "Recall Macro",

            "Macro F1",

            "Sensitivity",

            "Specificity",

            "ROC AUC",

            "PR AUC",

            "ECE",

            "Best Epoch",

            "Training Time (s)",

            "Success",

            "Error",

        ])


# ============================================================
# Build Optimizer
# ============================================================

def build_optimizer(
    cfg,
    model,
):
    """
    Build optimizer using the same parameter-group
    strategy as the working CARL train.py.
    """

    parameter_groups = []

    # --------------------------------------------------------
    # Backbone
    # --------------------------------------------------------

    if hasattr(model, "backbone"):

        parameter_groups.append({

            "params":
                model.backbone.parameters(),

            "lr":
                cfg.optimizer.backbone_lr,

        })

    # --------------------------------------------------------
    # Classifier
    # --------------------------------------------------------

    if hasattr(model, "classifier_head"):

        parameter_groups.append({

            "params":
                model.classifier_head.parameters(),

            "lr":
                cfg.optimizer.head_lr,

        })

    # --------------------------------------------------------
    # CARL projection head
    # --------------------------------------------------------

    projection_head = getattr(
        model,
        "projection_head",
        None,
    )

    if projection_head is not None:

        parameter_groups.append({

            "params":
                projection_head.parameters(),

            "lr":
                cfg.optimizer.head_lr,

        })

    # --------------------------------------------------------
    # Optional bias / auxiliary heads
    #
    # Kept for compatibility with future model versions.
    # --------------------------------------------------------

    bias_heads = getattr(
        model,
        "bias_heads",
        None,
    )

    if bias_heads is not None:

        for head in bias_heads.values():

            parameter_groups.append({

                "params":
                    head.parameters(),

                "lr":
                    cfg.optimizer.head_lr,

            })

    if not parameter_groups:

        raise RuntimeError(
            "No trainable parameter groups found."
        )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer_name = (
        cfg.optimizer.name.lower()
    )

    if optimizer_name == "adamw":

        optimizer = optim.AdamW(

            parameter_groups,

            weight_decay=(
                cfg.optimizer.weight_decay
            ),

            betas=tuple(
                cfg.optimizer.betas
            ),

            eps=(
                cfg.optimizer.eps
            ),
        )

    elif optimizer_name == "adam":

        optimizer = optim.Adam(

            parameter_groups,

            weight_decay=(
                cfg.optimizer.weight_decay
            ),
        )

    else:

        raise ValueError(
            f"Unknown optimizer: "
            f"{cfg.optimizer.name}"
        )

    return optimizer


# ============================================================
# Build Scheduler
# ============================================================

def build_scheduler(
    cfg,
    optimizer,
):
    """
    Build learning-rate scheduler.
    """

    scheduler_name = (
        cfg.scheduler.name.lower()
    )

    if scheduler_name == "cosine":

        scheduler = (
            optim.lr_scheduler.CosineAnnealingLR(

                optimizer,

                T_max=cfg.training.epochs,

                eta_min=(
                    cfg.scheduler.min_lr
                ),
            )
        )

    elif scheduler_name == "none":

        scheduler = None

    else:

        raise ValueError(
            f"Unknown scheduler: "
            f"{cfg.scheduler.name}"
        )

    return scheduler


# ============================================================
# Build Trainer
# ============================================================

def build_trainer(cfg):
    """
    Build the complete CARL training pipeline.
    """

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(
        cfg.seed,

        deterministic=(
            cfg.reproducibility.deterministic
        ),
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if cfg.device == "auto":

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:

        device = torch.device(
            cfg.device
        )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    print("\nBuilding dataloaders...")
    print("-" * 70)

    (
        train_loader,
        val_loader,
        test_loader,
    ) = build_dataloaders(cfg)

    print(
        f"Train batches : "
        f"{len(train_loader)}"
    )

    print(
        f"Val batches   : "
        f"{len(val_loader)}"
    )

    print(
        f"Test batches  : "
        f"{len(test_loader)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nBuilding model...")
    print("-" * 70)

    model = build_model(
        cfg
    ).to(device)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    print("\nComputing class weights...")
    print("-" * 70)

    class_weights = (
        compute_class_weights(
            train_loader.dataset
        )
    )

    class_weights = (
        class_weights.to(device)
    )

    print(
        f"Class weights: "
        f"{class_weights.detach().cpu().tolist()}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    print("\nBuilding loss...")
    print("-" * 70)

    loss_fn = Losses(
        cfg,
        class_weights=class_weights,
    )

    print(
        f"Classification : "
        f"{cfg.loss.classification}"
    )

    print(
        f"Metric         : "
        f"{cfg.loss.metric}"
    )

    print(
        f"Focal weight      : "
        f"{cfg.loss.focal_weight}"
    )

    print(
        f"Metric weight  : "
        f"{cfg.loss.metric_weight}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    print("\nBuilding optimizer...")
    print("-" * 70)

    optimizer = build_optimizer(
        cfg,
        model,
    )

    for idx, group in enumerate(
        optimizer.param_groups
    ):

        parameter_count = sum(
            p.numel()
            for p in group["params"]
        )

        print(
            f"Parameter group {idx}: "
            f"lr={group['lr']:.2e}, "
            f"parameters={parameter_count:,}"
        )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    print("\nBuilding scheduler...")
    print("-" * 70)

    scheduler = build_scheduler(
        cfg,
        optimizer,
    )

    print(
        f"Scheduler : "
        f"{cfg.scheduler.name}"
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    print("\nBuilding trainer...")
    print("-" * 70)

    trainer = Trainer(

        cfg=cfg,

        model=model,

        loss_fn=loss_fn,

        optimizer=optimizer,

        scheduler=scheduler,

        device=device,
    )

    print(
        "Trainer initialized successfully."
    )

    return (
        trainer,
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# Append Results
# ============================================================

def append_results(
    cfg,
    experiment,
    trainer,
    metrics,
    elapsed_time,
):
    """
    Append one completed experiment to the results CSV.
    """

    with open(
        RESULTS_CSV,
        "a",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([

            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            experiment["name"],

            experiment["description"],

            cfg.dataset.name,

            cfg.backbone.name,

            cfg.training.epochs,

            cfg.dataset.batch_size,

            experiment["ca_as3"],

            experiment["cafa"],

            experiment["carl"],

            cfg.loss.metric_weight,

            safe_float(
                metrics.get("accuracy")
            ),

            safe_float(
                metrics.get(
                    "balanced_accuracy"
                )
            ),

            safe_float(
                metrics.get(
                    "precision_macro"
                )
            ),

            safe_float(
                metrics.get(
                    "recall_macro"
                )
            ),

            safe_float(
                metrics.get("macro_f1")
            ),

            safe_float(
                metrics.get("sensitivity")
            ),

            safe_float(
                metrics.get("specificity")
            ),

            safe_float(
                metrics.get("roc_auc")
            ),

            safe_float(
                metrics.get("pr_auc")
            ),

            safe_float(
                metrics.get(
                    "expected_calibration_error"
                )
            ),

            (
                trainer.best_epoch + 1
                if trainer.best_epoch is not None
                else ""
            ),

            round(
                elapsed_time,
                2,
            ),

            True,

            "",

        ])


# ============================================================
# Run One Experiment
# ============================================================

def run_experiment(
    base_cfg,
    experiment,
):
    """
    Run one ablation experiment.
    """

    cfg = prepare_config(
        base_cfg,
        experiment,
    )

    exp_dir = Path(
        cfg.output.save_dir
    )

    exp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save experiment configuration
    # --------------------------------------------------------

    OmegaConf.save(
        cfg,
        exp_dir / "config.yaml",
    )

    # --------------------------------------------------------
    # Print experiment information
    # --------------------------------------------------------

    print("\n")
    print("=" * 90)
    print(
        f"EXPERIMENT : "
        f"{experiment['name']}"
    )
    print(
        f"Description: "
        f"{experiment['description']}"
    )
    print("=" * 90)

    print(
        f"CA-AS3 : "
        f"{experiment['ca_as3']}"
    )

    print(
        f"CAFA   : "
        f"{experiment['cafa']}"
    )

    print(
        f"CARL   : "
        f"{experiment['carl']}"
    )

    print(
        f"Metric weight : "
        f"{cfg.loss.metric_weight}"
    )

    start_time = time.time()

    try:

        # ----------------------------------------------------
        # Build pipeline
        # ----------------------------------------------------

        (
            trainer,
            train_loader,
            val_loader,
            test_loader,
        ) = build_trainer(cfg)

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        print("\nStarting training...")
        print("=" * 70)

        trainer.train(
            train_loader,
            val_loader,
        )

        # ----------------------------------------------------
        # Final test evaluation
        # ----------------------------------------------------

        print("\nRunning final test evaluation...")
        print("=" * 70)

        metrics = trainer.evaluate(
            test_loader,
        )

        elapsed_time = (
            time.time() - start_time
        )

        # ----------------------------------------------------
        # Save results
        # ----------------------------------------------------

        append_results(
            cfg,
            experiment,
            trainer,
            metrics,
            elapsed_time,
        )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print("\nExperiment completed successfully.")
        print("-" * 70)

        print(
            f"Accuracy          : "
            f"{metrics.get('accuracy', 0):.4f}"
        )

        print(
            f"Balanced Accuracy : "
            f"{metrics.get('balanced_accuracy', 0):.4f}"
        )

        print(
            f"Macro F1          : "
            f"{metrics.get('macro_f1', 0):.4f}"
        )

        print(
            f"ROC AUC           : "
            f"{metrics.get('roc_auc', 0):.4f}"
        )

        print(
            f"Training Time     : "
            f"{elapsed_time / 60:.2f} min"
        )

        return {

            "success": True,

            "experiment":
                experiment["name"],

            "description":
                experiment["description"],

            "components": {

                "ca_as3":
                    experiment["ca_as3"],

                "cafa":
                    experiment["cafa"],

                "carl":
                    experiment["carl"],
            },

            "metrics":
                metrics,

            "training_time":
                elapsed_time,

            "best_epoch":
                (
                    trainer.best_epoch + 1
                    if trainer.best_epoch is not None
                    else None
                ),
        }

    except KeyboardInterrupt:

        elapsed_time = (
            time.time() - start_time
        )

        print(
            "\nExperiment interrupted."
        )

        return {

            "success": False,

            "experiment":
                experiment["name"],

            "description":
                experiment["description"],

            "training_time":
                elapsed_time,

            "error":
                "KeyboardInterrupt",
        }

    except Exception as e:

        elapsed_time = (
            time.time() - start_time
        )

        print(
            "\nExperiment FAILED."
        )

        print(
            f"Error: {e}"
        )

        return {

            "success": False,

            "experiment":
                experiment["name"],

            "description":
                experiment["description"],

            "training_time":
                elapsed_time,

            "error":
                str(e),
        }


# ============================================================
# Run All Ablations
# ============================================================

def run_ablation():
    """
    Run every experiment in ABLATIONS.
    """

    initialize_results_file()

    base_cfg = load_config()

    summary = []

    total_start = time.time()

    for index, experiment in enumerate(
        ABLATIONS,
        start=1,
    ):

        print("\n")
        print("#" * 90)

        print(
            f"ABLATION "
            f"{index}/{len(ABLATIONS)}"
        )

        print("#" * 90)

        result = run_experiment(
            base_cfg,
            experiment,
        )

        summary.append(
            result
        )

    total_time = (
        time.time() - total_start
    )

    return (
        summary,
        total_time,
    )


# ============================================================
# Save Summary
# ============================================================

def save_summary(
    summary,
):
    """
    Save overall ablation results to JSON and CSV.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    with open(
        SUMMARY_JSON,
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            default=str,
        )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    with open(
        SUMMARY_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([

            "Experiment",

            "Success",

            "CA_AS3",

            "CAFA",

            "CARL",

            "Accuracy",

            "Balanced Accuracy",

            "Precision Macro",

            "Recall Macro",

            "Macro F1",

            "Sensitivity",

            "Specificity",

            "ROC AUC",

            "PR AUC",

            "ECE",

            "Best Epoch",

            "Training Time (s)",

            "Error",
        ])

        for result in summary:

            components = result.get(
                "components",
                {},
            )

            if result["success"]:

                metrics = (
                    result["metrics"]
                )

                writer.writerow([

                    result["experiment"],

                    True,

                    components.get(
                        "ca_as3"
                    ),

                    components.get(
                        "cafa"
                    ),

                    components.get(
                        "carl"
                    ),

                    safe_float(
                        metrics.get(
                            "accuracy"
                        )
                    ),

                    safe_float(
                        metrics.get(
                            "balanced_accuracy"
                        )
                    ),

                    safe_float(
                        metrics.get(
                            "precision_macro"
                        )
                    ),

                    safe_float(
                        metrics.get(
                            "recall_macro"
                        )
                    ),

                    safe_float(
                        metrics.get(
                            "macro_f1"
                        )
                    ),

                    safe_float(
                        metrics.get(
                            "sensitivity"
                        )
                    ),

                    safe_float(
                        metrics.get(
                            "specificity"
                        )
                    ),

                    safe_float(
                        metrics.get(
                            "roc_auc"
                        )
                    ),

                    safe_float(
                        metrics.get(
                            "pr_auc"
                        )
                    ),

                    safe_float(
                        metrics.get(
                            "expected_calibration_error"
                        )
                    ),

                    result.get(
                        "best_epoch"
                    ),

                    round(
                        result[
                            "training_time"
                        ],
                        2,
                    ),

                    "",
                ])

            else:

                writer.writerow([

                    result["experiment"],

                    False,

                    components.get(
                        "ca_as3",
                        "",
                    ),

                    components.get(
                        "cafa",
                        "",
                    ),

                    components.get(
                        "carl",
                        "",
                    ),

                    "",

                    "",

                    "",

                    "",

                    "",

                    "",

                    "",

                    "",

                    "",

                    "",

                    "",

                    round(
                        result[
                            "training_time"
                        ],
                        2,
                    ),

                    result.get(
                        "error",
                        "",
                    ),
                ])


# ============================================================
# Print Summary
# ============================================================

def print_summary(
    summary,
    total_time,
):
    """
    Print the final ablation comparison.
    """

    print("\n")
    print("=" * 100)
    print("FINAL CARL ABLATION SUMMARY")
    print("=" * 100)

    successful = [

        result

        for result in summary

        if result["success"]
    ]

    if not successful:

        print(
            "No successful experiments."
        )

        return

    # --------------------------------------------------------
    # Sort by balanced accuracy
    # --------------------------------------------------------

    successful = sorted(

        successful,

        key=lambda result:
            result["metrics"].get(
                "balanced_accuracy",
                0.0,
            ),

        reverse=True,
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()

    print(

        f"{'Experiment':<18}"

        f"{'CA-AS3':>9}"

        f"{'CAFA':>9}"

        f"{'CARL':>9}"

        f"{'Accuracy':>12}"

        f"{'BACC':>12}"

        f"{'F1':>12}"

        f"{'ROC-AUC':>12}"

    )

    print("-" * 100)

    # --------------------------------------------------------
    # Rows
    # --------------------------------------------------------

    for result in successful:

        metrics = (
            result["metrics"]
        )

        components = (
            result["components"]
        )

        print(

            f"{result['experiment']:<18}"

            f"{str(components['ca_as3']):>9}"

            f"{str(components['cafa']):>9}"

            f"{str(components['carl']):>9}"

            f"{metrics.get('accuracy', 0):>12.4f}"

            f"{metrics.get('balanced_accuracy', 0):>12.4f}"

            f"{metrics.get('macro_f1', 0):>12.4f}"

            f"{metrics.get('roc_auc', 0):>12.4f}"

        )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    best = successful[0]

    print("\n")
    print("=" * 100)
    print("BEST ABLATION")
    print("=" * 100)

    print(
        f"Experiment          : "
        f"{best['experiment']}"
    )

    print(
        f"Description         : "
        f"{best['description']}"
    )

    print(
        f"Balanced Accuracy   : "
        f"{best['metrics'].get('balanced_accuracy', 0):.4f}"
    )

    print(
        f"Macro F1            : "
        f"{best['metrics'].get('macro_f1', 0):.4f}"
    )

    print(
        f"ROC AUC             : "
        f"{best['metrics'].get('roc_auc', 0):.4f}"
    )

    print(
        f"Best Epoch          : "
        f"{best.get('best_epoch')}"
    )

    print(
        f"Training Time       : "
        f"{best['training_time'] / 60:.2f} min"
    )

    print(
        f"Total Runtime       : "
        f"{total_time / 60:.2f} min"
    )

    print("=" * 100)


# ============================================================
# Main
# ============================================================

def main():
    """
    Entry point.
    """

    print()

    print("=" * 100)
    print("CARL ABLATION STUDY")
    print("=" * 100)

    print(
        f"Dataset    : "
        f"Loading from {CONFIG_FILE}"
    )

    print(
        f"Experiments: "
        f"{len(ABLATIONS)}"
    )

    print("=" * 100)

    summary, total_time = (
        run_ablation()
    )

    save_summary(
        summary
    )

    print_summary(
        summary,
        total_time,
    )

    print()

    print("=" * 100)
    print("All ablation experiments finished.")
    print("=" * 100)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()