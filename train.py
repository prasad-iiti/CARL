import torch
import torch.optim as optim
from omegaconf import OmegaConf

from src.data import (
    build_dataloaders,
    compute_class_weights,
)
from src.losses import Losses
from src.model import build_model
from src.trainer import Trainer
from src.utils import set_seed


# ==========================================================
# Device
# ==========================================================

def resolve_device(cfg):
    """Resolve configured device."""

    device_name = str(cfg.device).lower()

    if device_name == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available."
        )

    return device


# ==========================================================
# Optimizer
# ==========================================================

def build_optimizer(cfg, model):
    """
    Build CARL optimizer with separate backbone/head
    learning rates.
    """

    backbone_parameters = []
    head_parameters = []

    # ------------------------------------------------------
    # Backbone
    # ------------------------------------------------------

    if hasattr(model, "backbone"):
        backbone_parameters.extend(
            parameter
            for parameter in model.backbone.parameters()
            if parameter.requires_grad
        )

    # ------------------------------------------------------
    # Everything outside backbone
    # ------------------------------------------------------

    backbone_ids = {
        id(parameter)
        for parameter in backbone_parameters
    }

    for parameter in model.parameters():
        if (
            parameter.requires_grad
            and id(parameter) not in backbone_ids
        ):
            head_parameters.append(parameter)

    parameter_groups = []

    if backbone_parameters:
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": float(cfg.optimizer.backbone_lr),
            }
        )

    if head_parameters:
        parameter_groups.append(
            {
                "params": head_parameters,
                "lr": float(cfg.optimizer.head_lr),
            }
        )

    if not parameter_groups:
        raise RuntimeError(
            "No trainable parameters found for optimizer."
        )

    optimizer_name = str(
        cfg.optimizer.name
    ).lower()

    if optimizer_name == "adamw":

        optimizer = optim.AdamW(
            parameter_groups,
            weight_decay=float(
                cfg.optimizer.weight_decay
            ),
            betas=tuple(cfg.optimizer.betas),
            eps=float(cfg.optimizer.eps),
        )

    elif optimizer_name == "adam":

        optimizer = optim.Adam(
            parameter_groups,
            weight_decay=float(
                cfg.optimizer.weight_decay
            ),
            betas=tuple(cfg.optimizer.betas),
            eps=float(cfg.optimizer.eps),
        )

    else:
        raise ValueError(
            f"Unknown optimizer: {cfg.optimizer.name}"
        )

    return optimizer


# ==========================================================
# Scheduler
# ==========================================================

def build_scheduler(cfg, optimizer):
    """
    Build learning-rate scheduler.
    """

    scheduler_name = str(
        cfg.scheduler.name
    ).lower()

    if scheduler_name == "cosine":

        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.training.epochs),
            eta_min=float(cfg.scheduler.min_lr),
        )

    if scheduler_name == "none":
        return None

    raise ValueError(
        f"Unknown scheduler: {cfg.scheduler.name}"
    )


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)
    print("CARL TRAINING")
    print("=" * 70)

    # ------------------------------------------------------
    # Configuration
    # ------------------------------------------------------

    cfg = OmegaConf.load("config.yaml")

    cfg["_config_path"] = "config.yaml"

    print("\nConfiguration")
    print("-" * 70)
    print(
        OmegaConf.to_yaml(cfg)
    )

    # ------------------------------------------------------
    # Device
    # ------------------------------------------------------

    device = resolve_device(cfg)

    print(f"Using device: {device}")
    print(f"Dataset    : {cfg.dataset.name}")
    print(f"Experiment : {cfg.experiment.name}")
    print(f"Run ID     : {cfg.experiment.run_id}")

    # ------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------

    set_seed(cfg.seed, deterministic=cfg.reproducibility.deterministic,)

    # ------------------------------------------------------
    # Data
    # ------------------------------------------------------

    print("\nBuilding dataloaders...")
    print("-" * 70)

    (
        train_loader,
        val_loader,
        test_loader,
    ) = build_dataloaders(cfg)

    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")
    print(f"Test batches  : {len(test_loader)}")

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    print("\nBuilding CARL model...")
    print("-" * 70)

    model = build_model(cfg).to(device)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    # ------------------------------------------------------
    # Class weights
    # ------------------------------------------------------

    print("\nComputing class weights...")
    print("-" * 70)

    class_weights = compute_class_weights(
        train_loader.dataset
    )

    class_weights = class_weights.to(
        device
    )

    print(
        "Class weights:",
        class_weights.detach().cpu().tolist(),
    )

    # ------------------------------------------------------
    # Loss
    # ------------------------------------------------------

    print("\nBuilding CARL loss...")
    print("-" * 70)

    loss_fn = Losses(cfg, class_weights=class_weights,)
    
    print("Classification : " f"{cfg.loss.classification}")

    print("Metric         : " f"{cfg.loss.metric}")

    print("Focal weight      : " f"{cfg.loss.focal_weight}")

    print("Metric weight  : " f"{cfg.loss.metric_weight}")

    # ------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------

    print("\nBuilding optimizer...")
    print("-" * 70)

    optimizer = build_optimizer(cfg, model,)

    for index, group in enumerate(
        optimizer.param_groups
    ):
        print(
            f"Parameter group {index}: "
            f"lr={group['lr']:.2e}, "
            f"parameters={len(group['params'])}"
        )

    # ------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------

    print("\nBuilding scheduler...")
    print("-" * 70)

    scheduler = build_scheduler(cfg, optimizer,)

    if scheduler is not None:
        print(
            f"Scheduler : {cfg.scheduler.name}"
        )
    else:
        print(
            "Scheduler : disabled"
        )

    # ------------------------------------------------------
    # Trainer
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # Training
    # ------------------------------------------------------

    # print("\nStarting training...")
    # print("=" * 70)

    try:

        training_result = trainer.train(
            train_loader,
            val_loader,
        )

    except KeyboardInterrupt:

        print(
            "\nTraining interrupted by user."
        )

        return

    # ------------------------------------------------------
    # Training summary
    # ------------------------------------------------------

    # print("\nTraining completed.")
    # print("-" * 70)

    print(
        f"Best {cfg.metrics.primary}: "
        f"{training_result['best_score']:.6f}"
    )

    print(
        f"Best epoch: "
        f"{training_result['best_epoch'] + 1}"
    )

    # ------------------------------------------------------
    # Final test evaluation
    # ------------------------------------------------------

    # print("\nRunning final test evaluation...")
    # print("=" * 70)

    test_metrics = trainer.evaluate(
        test_loader,
    )

    # print("\nFinal test results")
    # print("-" * 70)

    # for key, value in test_metrics.items():

    #     if isinstance(
    #         value,
    #         (int, float),
    #     ):

    #         print(
    #             f"{key:25s}: "
    #             f"{value:.6f}"
    #         )

    # print("=" * 70)
    # print("CARL training completed.")
    # print("=" * 70)


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":
    main()
