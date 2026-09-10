import torch
from omegaconf import OmegaConf
from thop import profile

from src.model import build_model, count_parameters


# ============================================================
# Configuration
# ============================================================

CONFIG_PATH = "/home/prasad/CARL/config.yaml"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("CARL FLOPs ANALYSIS")
print("=" * 70)

print(f"Device: {device}")


# ============================================================
# Load configuration
# ============================================================

cfg = OmegaConf.load(CONFIG_PATH)

print(f"Dataset : {cfg.dataset.name}")
print(f"Backbone: {cfg.backbone.name}")

image_size = int(cfg.dataset.image_size)

print(
    f"Input   : 3 x {image_size} x {image_size}"
)


# ============================================================
# Build model
# ============================================================

model = build_model(cfg)

model = model.to(device)
model.eval()


# ============================================================
# Parameters
# ============================================================

total_params, trainable_params = count_parameters(
    model
)

print()
print(
    f"Total parameters     : "
    f"{total_params:,}"
)

print(
    f"Trainable parameters : "
    f"{trainable_params:,}"
)

print(
    f"Parameters           : "
    f"{total_params / 1e6:.3f} M"
)


# ============================================================
# Dummy input
# ============================================================

dummy_input = torch.randn(
    1,
    3,
    image_size,
    image_size,
    device=device,
)


# ============================================================
# FLOPs / MACs using THOP
# ============================================================

print()
print("Calculating FLOPs using THOP...")

with torch.no_grad():

    macs, thop_params = profile(
        model,
        inputs=(dummy_input, 0),
        verbose=False,
    )


# ============================================================
# Convert MACs to FLOPs
# ============================================================

# THOP reports MACs.
# Conventionally:
#
#     1 MAC = 2 FLOPs
#
# Therefore:
#
#     FLOPs = 2 x MACs

total_flops = 2 * macs


# ============================================================
# Results
# ============================================================

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

print(
    f"Total MACs  : {macs:,.0f}"
)

print(
    f"GMACs       : {macs / 1e9:.4f}"
)

print(
    f"Total FLOPs : {total_flops:,.0f}"
)

print(
    f"GFLOPs      : {total_flops / 1e9:.4f}"
)

print(
    f"THOP Params : {thop_params:,}"
)

print("=" * 70)