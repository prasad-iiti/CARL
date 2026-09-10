import argparse

import torch
from omegaconf import OmegaConf

from src.data import build_dataloaders, compute_class_weights
from src.losses import Losses
from src.model import build_model
from src.trainer import Trainer
from src.utils import set_seed


from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ADF-Skin model."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint file.",
    )

    return parser.parse_args()


def main():
    
    # =====================================================
    # Parse Arguments
    # =====================================================
    args = parse_args()

    # =====================================================
    # Configuration
    # =====================================================
    cfg = OmegaConf.load("config.yaml")
    cfg["_config_path"] = "configs/config.yaml"

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Run ID     : {cfg.experiment.run_id}")

    # =====================================================
    # Reproducibility
    # =====================================================
    set_seed(cfg["seed"])

    # =====================================================
    # Device
    # =====================================================
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    # =====================================================
    # Data
    # =====================================================
    train_loader, _, test_loader = build_dataloaders(cfg)

    # =====================================================
    # Model
    # =====================================================
    model = build_model(cfg).to(device)

    # =====================================================
    # Loss
    # =====================================================
    class_weights = compute_class_weights(train_loader.dataset)
    loss_fn = Losses(cfg, class_weights=class_weights.to(device))

    # =====================================================
    # Optimizer
    # =====================================================
    optimizer = None

    # =====================================================
    # Scheduler
    # =====================================================
    scheduler = None

    print(f"Device     : {device}")
    print(f"Dataset    : {cfg.dataset.name}")
    print(f"Backbone   : {cfg.backbone}")

    # =====================================================
    # Trainer
    # =====================================================
    trainer = Trainer(
        cfg=cfg,
        model=model,
        loss_fn=loss_fn,
        device=device,
    )

    # =====================================================
    # Evaluate
    # =====================================================
    trainer.evaluate(
        test_loader=test_loader,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()