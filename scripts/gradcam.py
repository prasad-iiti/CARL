# ============================================================
# CARL - Grad-CAM++ Visualization
# ============================================================
#
# Generates Grad-CAM++ visualizations for the CARL model.
#
# Targets:
#   1. EfficientNet-B0 backbone: features[7]
#   2. CAFA refinement:           cafa.refinement[0]
#
# NOTE:
#   The representation_projection[0] layer is intentionally
#   NOT used as a visualization target because it is a 1x1
#   projection layer and provides weak spatial localization.
#
# IMPORTANT:
#   Evaluation is performed with epoch=0 to exactly match the
#   current Evaluator implementation.
#
# Usage:
#
#   python gradcam.py \
#       --checkpoint /path/to/best_model.pth
#
# Optional:
#
#   python gradcam.py \
#       --checkpoint /path/to/best_model.pth \
#       --num-correct 20 \
#       --num-misclassified 20
#
#   python gradcam.py \
#       --checkpoint /path/to/best_model.pth \
#       --num-correct 10 \
#       --num-misclassified 10 \
#       --target predicted
#
# ============================================================

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.data import build_dataloaders
from src.model import build_model
from src.utils import load_checkpoint, set_seed


# ============================================================
# Argument Parser
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM++ visualizations for CARL."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to best_model.pth",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/gradcam",
        help="Directory where Grad-CAM++ results are saved.",
    )

    parser.add_argument(
        "--num-correct",
        type=int,
        default=20,
        help="Number of correctly classified images to visualize.",
    )

    parser.add_argument(
        "--num-misclassified",
        type=int,
        default=20,
        help="Number of misclassified images to visualize.",
    )

    parser.add_argument(
        "--target",
        type=str,
        choices=["predicted", "true"],
        default="predicted",
        help=(
            "Class whose logit is used for Grad-CAM++. "
            "'predicted' is recommended for classification "
            "explanation."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for selecting images.",
    )

    return parser.parse_args()


# ============================================================
# Grad-CAM++
# ============================================================

class GradCAMPlusPlus:
    """
    Generic Grad-CAM++ implementation.

    Grad-CAM++ uses higher-order gradient information to
    estimate spatially sensitive channel weights.

    Target layer should produce:
        [B, C, H, W]
    """

    def __init__(
        self,
        model,
        target_layer,
    ):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # ----------------------------------------------------
        # Register ONLY a forward hook.
        #
        # The gradient is captured directly from the output
        # tensor using Tensor.register_hook().
        # This is more reliable for composite modules such
        # as CA-AS3 and CAFA.
        # ----------------------------------------------------

        self.forward_handle = (
            self.target_layer.register_forward_hook(
                self._forward_hook
            )
        )

    # --------------------------------------------------------
    # Forward hook
    # --------------------------------------------------------

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ):
        """
        Capture the spatial tensor produced by the target layer.

        CA-AS3 returns a dictionary, so extract the appropriate
        spatial feature tensor from the dictionary.
        """

        # --------------------------------------------------------
        # Case 1: Tensor
        # --------------------------------------------------------

        if torch.is_tensor(output):
            target_output = output

        # --------------------------------------------------------
        # Case 2: Tuple / list
        # --------------------------------------------------------

        elif isinstance(output, (tuple, list)):

            target_output = None

            for item in output:
                if torch.is_tensor(item) and item.ndim == 4:
                    target_output = item
                    break

            if target_output is None:
                raise RuntimeError(
                    "Grad-CAM++ could not find a 4-D spatial tensor "
                    "inside the target layer tuple/list."
                )

        # --------------------------------------------------------
        # Case 3: Dictionary
        # --------------------------------------------------------

        elif isinstance(output, dict):

            target_output = None

            # Preferred keys for CA-AS3
            preferred_keys = [
                "output",
                "fused",
                "features",
                "feature",
                "x",
                "out",
            ]

            for key in preferred_keys:

                if key in output:

                    value = output[key]

                    if (
                        torch.is_tensor(value)
                        and value.ndim == 4
                    ):
                        target_output = value
                        break

            # ----------------------------------------------------
            # If preferred key not found, search all dictionary
            # values for a spatial feature tensor.
            # ----------------------------------------------------

            if target_output is None:

                for key, value in output.items():

                    if (
                        torch.is_tensor(value)
                        and value.ndim == 4
                    ):
                        target_output = value

                        print(
                            f"[Grad-CAM++] Using CA-AS3 "
                            f"dictionary key: '{key}'"
                        )

                        break

            if target_output is None:

                raise RuntimeError(
                    "Grad-CAM++ could not find a 4-D spatial "
                    "tensor inside the target layer dictionary. "
                    f"Available keys: {list(output.keys())}"
                )

        else:

            raise RuntimeError(
                "Grad-CAM++ target layer returned unsupported "
                f"type: {type(output)}"
            )

        # --------------------------------------------------------
        # Validate target tensor
        # --------------------------------------------------------

        if not torch.is_tensor(target_output):

            raise RuntimeError(
                "Grad-CAM++ target output is not a tensor."
            )

        if target_output.ndim != 4:

            raise RuntimeError(
                "Grad-CAM++ target layer must provide a "
                "[B, C, H, W] tensor. "
                f"Received shape: {tuple(target_output.shape)}"
            )

        # --------------------------------------------------------
        # Store activation
        # --------------------------------------------------------

        self.activations = target_output

        # --------------------------------------------------------
        # Capture gradient directly from target tensor
        # --------------------------------------------------------

        if target_output.requires_grad:

            target_output.register_hook(
                self._gradient_hook
            )

        else:

            raise RuntimeError(
                "Grad-CAM++ target output does not require "
                "gradients."
            )

    # --------------------------------------------------------
    # Tensor gradient hook
    # --------------------------------------------------------

    def _gradient_hook(
        self,
        grad,
    ):
        """
        Capture gradient with respect to target-layer output.
        """

        self.gradients = grad

    # --------------------------------------------------------
    # Generate Grad-CAM++
    # --------------------------------------------------------

    def generate(
        self,
        input_tensor,
        target_class,
        epoch=0,
    ):
        """
        Generate Grad-CAM++ visualization for the specified
        target class.

        Args:
            input_tensor:
                Input image tensor [B, C, H, W]

            target_class:
                Target class index.
                Can be:
                    int
                    or
                    Tensor [B]

            epoch:
                Training epoch passed to the model.
                For inference/evaluation this should normally
                be 0.

        Returns:
            cam:
                Normalized CAM as numpy array [H, W]
        """

        # ----------------------------------------------------
        # Clear previous state
        # ----------------------------------------------------

        self.activations = None
        self.gradients = None

        # ----------------------------------------------------
        # Prepare input
        # ----------------------------------------------------

        input_tensor = (
            input_tensor
            .clone()
            .detach()
            .requires_grad_(True)
        )

        # ----------------------------------------------------
        # Clear model gradients BEFORE forward/backward
        # ----------------------------------------------------

        self.model.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        output = self.model(
            input_tensor,
            epoch=epoch,
        )

        # ----------------------------------------------------
        # Handle model output
        # ----------------------------------------------------

        if isinstance(output, tuple):
            output = output[0]

        elif isinstance(output, dict):
            if "logits" not in output:
                raise RuntimeError(
                    "Model output dictionary does not contain 'logits'."
                )

            output = output["logits"]

        # Verify logits
        if not torch.is_tensor(output):
            raise RuntimeError(
                "Model did not return a logits tensor. "
                f"Received type: {type(output)}"
            )

        # ----------------------------------------------------
        # Select target class score
        # ----------------------------------------------------

        if isinstance(target_class, int):

            target_score = (
                output[:, target_class].sum()
            )

        elif torch.is_tensor(target_class):

            # Ensure correct device
            target_class = target_class.to(
                output.device
            )

            # Ensure integer type
            target_class = target_class.long()

            # Handle scalar tensor
            if target_class.ndim == 0:
                target_class = target_class.unsqueeze(0)

            target_score = (
                output.gather(
                    1,
                    target_class.view(-1, 1),
                )
                .sum()
            )

        else:

            raise TypeError(
                "target_class must be an int "
                "or torch.Tensor."
            )

        # ----------------------------------------------------
        # Backward pass
        # ----------------------------------------------------

        target_score.backward(
            retain_graph=False
        )

        # ----------------------------------------------------
        # Verify activations
        # ----------------------------------------------------

        if self.activations is None:

            raise RuntimeError(
                "Grad-CAM++ failed: target layer "
                "activations were not captured."
            )

        # ----------------------------------------------------
        # Verify gradients
        # ----------------------------------------------------

        if self.gradients is None:

            raise RuntimeError(
                "Grad-CAM++ failed: target layer "
                "gradients were not captured."
            )

        # ----------------------------------------------------
        # Get activation and gradient tensors
        # ----------------------------------------------------

        activations = self.activations
        gradients = self.gradients

        # ----------------------------------------------------
        # Validate dimensions
        # ----------------------------------------------------

        if activations.ndim != 4:

            raise RuntimeError(
                "Grad-CAM++ target layer must produce "
                "[B, C, H, W] activations. "
                f"Received shape: {tuple(activations.shape)}"
            )

        if gradients.ndim != 4:

            raise RuntimeError(
                "Grad-CAM++ gradients must have shape "
                "[B, C, H, W]. "
                f"Received shape: {tuple(gradients.shape)}"
            )

        # ----------------------------------------------------
        # Ensure activation/gradient shapes match
        # ----------------------------------------------------

        if activations.shape != gradients.shape:

            raise RuntimeError(
                "Grad-CAM++ activation/gradient shape mismatch: "
                f"activations={tuple(activations.shape)}, "
                f"gradients={tuple(gradients.shape)}"
            )

        # ----------------------------------------------------
        # Standard practical Grad-CAM++ formulation
        # ----------------------------------------------------

        # First higher-order term
        gradients_2 = gradients.pow(2)

        # Second higher-order term
        gradients_3 = gradients.pow(3)

        # ----------------------------------------------------
        # Sum:
        #
        # A * gradient^3
        #
        # over spatial dimensions
        # ----------------------------------------------------

        activation_gradients_3 = (
            activations * gradients_3
        ).sum(
            dim=(2, 3),
            keepdim=True,
        )

        # ----------------------------------------------------
        # Alpha denominator
        # ----------------------------------------------------

        alpha_denom = (
            2.0 * gradients_2
            + activation_gradients_3
        )

        # Numerical stability
        eps = 1e-7

        # ----------------------------------------------------
        # Alpha coefficients
        # ----------------------------------------------------

        alphas = (
            gradients_2
            / (alpha_denom + eps)
        )

        # ----------------------------------------------------
        # Positive gradients
        # ----------------------------------------------------

        positive_gradients = F.relu(
            gradients
        )

        # ----------------------------------------------------
        # Channel importance weights
        # ----------------------------------------------------

        weights = (
            alphas
            * positive_gradients
        ).sum(
            dim=(2, 3),
            keepdim=True,
        )

        # ----------------------------------------------------
        # Weighted combination of activation maps
        # ----------------------------------------------------

        cam = (
            weights
            * activations
        ).sum(
            dim=1
        )

        # ----------------------------------------------------
        # Keep only positive evidence
        # ----------------------------------------------------

        cam = F.relu(cam)

        # ----------------------------------------------------
        # Remove batch dimension
        # ----------------------------------------------------

        cam = cam[0]

        # ----------------------------------------------------
        # Normalize CAM to [0, 1]
        # ----------------------------------------------------

        cam_min = cam.min()
        cam_max = cam.max()

        if (
            cam_max - cam_min
        ).abs() > eps:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            # No meaningful activation
            cam = torch.zeros_like(
                cam
            )

        # ----------------------------------------------------
        # Return CPU numpy array
        # ----------------------------------------------------

        return (
            cam
            .detach()
            .cpu()
            .numpy()
        )

    # --------------------------------------------------------
    # Remove hooks
    # --------------------------------------------------------

    def remove_hooks(self):

        if self.forward_handle is not None:

            self.forward_handle.remove()

            self.forward_handle = None


# ============================================================
# Image Utilities
# ============================================================

def denormalize_image(
    tensor,
    mean,
    std,
):
    """
    Convert normalized tensor back to uint8 RGB image.

    NOTE:
        This image remains at the model input resolution.
    """

    image = (
        tensor
        .detach()
        .cpu()
        .numpy()
    )

    image = np.transpose(
        image,
        (1, 2, 0),
    )

    mean = np.asarray(
        mean,
        dtype=np.float32,
    )

    std = np.asarray(
        std,
        dtype=np.float32,
    )

    image = (
        image * std
    ) + mean

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    image = (
        image * 255.0
    ).round().astype(
        np.uint8
    )

    return image


def load_original_image(
    dataset,
    index,
    mean,
    std,
):
    """
    Recover the RGB visualization image from the tensor
    returned by SkinDataset.

    The dataset applies Resize(224,224), ToTensor and
    ImageNet normalization for the test set.
    """

    sample = dataset[index]

    image_tensor = sample["image"]

    if not torch.is_tensor(image_tensor):
        raise RuntimeError(
            "Dataset did not return a tensor image."
        )

    return denormalize_image(
        image_tensor,
        mean=mean,
        std=std,
    )


def resize_cam(
    cam,
    height,
    width,
):
    """
    Resize CAM to target image resolution.
    """

    return cv2.resize(
        cam,
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )


def smooth_cam(
    cam,
):
    """
    Apply light Gaussian smoothing to reduce pixel-level
    artifacts while retaining lesion localization.
    """

    cam = cv2.GaussianBlur(
        cam,
        ksize=(0, 0),
        sigmaX=3.0,
        sigmaY=3.0,
    )

    cam_min = cam.min()
    cam_max = cam.max()

    if cam_max - cam_min > 1e-12:

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min
        )

    return np.clip(
        cam,
        0.0,
        1.0,
    ).astype(
        np.float32
    )


def create_heatmap(
    cam,
):
    """
    Convert normalized CAM [0,1] to RGB heatmap.
    """

    cam_uint8 = (
        np.clip(
            cam,
            0.0,
            1.0,
        )
        * 255.0
    ).astype(
        np.uint8
    )

    heatmap = cv2.applyColorMap(
        cam_uint8,
        cv2.COLORMAP_JET,
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB,
    )

    return heatmap


def create_overlay(
    image,
    heatmap,
    alpha=0.40,
):
    """
    Blend original RGB image with heatmap.

    The original image is deliberately given slightly
    greater weight to preserve lesion morphology.
    """

    image_float = (
        image.astype(
            np.float32
        )
    )

    heatmap_float = (
        heatmap.astype(
            np.float32
        )
    )

    overlay = (
        (1.0 - alpha)
        *
        image_float
        +
        alpha
        *
        heatmap_float
    )

    return np.clip(
        overlay,
        0,
        255,
    ).astype(
        np.uint8
    )


# ============================================================
# Save Image
# ============================================================

def save_rgb_image(
    path,
    image,
):
    """
    Save RGB image using OpenCV.
    """

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )

    success = cv2.imwrite(
        str(path),
        image_bgr,
    )

    if not success:

        raise IOError(
            f"Failed to save image:\n{path}"
        )


# ============================================================
# Class Names
# ============================================================

def get_class_names(
    dataset,
):
    """
    Return class names indexed according to the dataset's
    deterministic label mapping.
    """

    if dataset.task == "multiclass":

        return [
            str(
                dataset.index_to_label[idx]
            )
            for idx in range(
                dataset.num_classes
            )
        ]

    return [
        "class_0",
        "class_1",
    ]


# ============================================================
# Target Layers
# ============================================================

def get_target_layers(model):
    layers = {}

    # 1. CA-AS3 output
    if getattr(model, "ca_as3", None) is not None:
        layers["ca_as3"] = model.ca_as3

    # 2. CAFA output
    if getattr(model, "cafa", None) is not None:
        layers["cafa"] = model.cafa

    # 3. CARL representation
    if getattr(model, "representation_projection", None) is not None:
        layers["carl"] = model.representation_projection

    return layers


# ============================================================
# Checkpoint Loading
# ============================================================

def load_model_checkpoint(
    model,
    checkpoint_path,
    device,
):
    """
    Load CARL checkpoint using the project's existing
    checkpoint utility.
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    print()
    print(
        "=" * 70
    )

    print(
        "Loading checkpoint"
    )

    print(
        f"Path: {checkpoint_path}"
    )

    model, _, _, epoch, score = (
        load_checkpoint(
            path=str(
                checkpoint_path
            ),
            model=model,
            optimizer=None,
            scheduler=None,
            map_location=device,
        )
    )

    print(
        f"Checkpoint epoch : {epoch}"
    )

    print(
        f"Checkpoint score : {score}"
    )

    print(
        "=" * 70
    )

    return model


# ============================================================
# Select Samples
# ============================================================

def select_samples(
    records,
    labels,
    predictions,
    num_correct,
    num_misclassified,
    seed,
):
    """
    Select samples deterministically.

    Correct samples are taken from the beginning of the
    correctly classified set. Misclassified samples are
    taken from the beginning of the misclassified set.
    """

    correct_indices = [
        idx
        for idx in range(len(labels))
        if int(labels[idx]) == int(predictions[idx])
    ]

    wrong_indices = [
        idx
        for idx in range(len(labels))
        if int(labels[idx]) != int(predictions[idx])
    ]

    selected_correct = correct_indices[:num_correct]
    selected_wrong = wrong_indices[:num_misclassified]

    return (
        selected_correct,
        selected_wrong,
    )


# ============================================================
# Run Predictions
# ============================================================

@torch.no_grad()
def collect_predictions(
    model,
    loader,
    device,
):
    """
    Run the test set exactly as the existing Evaluator does.

    IMPORTANT:
        epoch=0

    Returns
    -------
    labels
    predictions
    probabilities
    sample_records
    """

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_records = []

    dataset = loader.dataset

    for batch in loader:

        images = batch[
            "image"
        ].to(
            device,
            non_blocking=True,
        )

        labels = batch[
            "label"
        ].to(
            device,
            non_blocking=True,
        )

        outputs = model(
            images,
            epoch=0,
        )

        logits = outputs[
            "logits"
        ]

        probabilities = F.softmax(
            logits,
            dim=1,
        )

        predictions = (
            probabilities.argmax(
                dim=1
            )
        )

        all_labels.append(
            labels.cpu()
        )

        all_predictions.append(
            predictions.cpu()
        )

        all_probabilities.append(
            probabilities.cpu()
        )

    labels = torch.cat(
        all_labels,
        dim=0,
    ).numpy()

    predictions = torch.cat(
        all_predictions,
        dim=0,
    ).numpy()

    probabilities = torch.cat(
        all_probabilities,
        dim=0,
    ).numpy()

    # Test loader uses shuffle=False.
    for record in dataset.records:

        all_records.append(
            record
        )

    if len(all_records) != len(labels):

        raise RuntimeError(
            "Number of dataset records does not match "
            "number of predictions.\n"
            f"Records: {len(all_records)}\n"
            f"Predictions: {len(labels)}"
        )

    return (
        labels,
        predictions,
        probabilities,
        all_records,
    )


# ============================================================
# Generate Single Image CAM
# ============================================================

def generate_single_sample_cams(
    model,
    target_layers,
    image_tensor,
    target_class,
):
    """
    Generate Grad-CAM++ from all target layers.
    """

    cams = {}

    for name, layer in (
        target_layers.items()
    ):

        gradcam = GradCAMPlusPlus(
            model=model,
            target_layer=layer,
        )

        try:

            cam = gradcam.generate(
                input_tensor=image_tensor,
                target_class=target_class,
                epoch=0,
            )

            cams[name] = cam

        finally:

            gradcam.remove_hooks()

    return cams


# ============================================================
# Save CAM Outputs
# ============================================================

def save_sample_outputs(
    output_dir,
    category,
    sample_index,
    image_name,
    original_image,
    cams,
    true_class_name,
    predicted_class_name,
    confidence,
    target_class_name,
):
    """
    Save all visualization images for one sample.
    """

    image_stem = Path(
        str(image_name)
    ).stem

    safe_name = "".join(
        character
        if (
            character.isalnum()
            or character in (
                "-",
                "_",
            )
        )
        else "_"
        for character in image_stem
    )

    sample_dir = (
        Path(output_dir)
        /
        category
        /
        f"{sample_index:04d}_{safe_name}"
    )

    sample_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    save_rgb_image(
        sample_dir / "original.jpg",
        original_image,
    )

    # --------------------------------------------------------
    # CAM outputs
    # --------------------------------------------------------

    height, width = (
        original_image.shape[:2]
    )

    for name, cam in (
        cams.items()
    ):

        resized_cam = resize_cam(
            cam,
            height,
            width,
        )

        # Light smoothing improves visual readability.
        resized_cam = smooth_cam(
            resized_cam
        )

        heatmap = create_heatmap(
            resized_cam
        )

        overlay = create_overlay(
            original_image,
            heatmap,
            alpha=0.40,
        )

        save_rgb_image(
            sample_dir
            /
            f"{name}_gradcampp_heatmap.jpg",
            heatmap,
        )

        save_rgb_image(
            sample_dir
            /
            f"{name}_gradcampp_overlay.jpg",
            overlay,
        )

    # --------------------------------------------------------
    # Prediction information
    # --------------------------------------------------------

    info_path = (
        sample_dir
        /
        "prediction.txt"
    )

    with info_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            f"Image: {image_name}\n"
        )

        file.write(
            f"True class: "
            f"{true_class_name}\n"
        )

        file.write(
            f"Predicted class: "
            f"{predicted_class_name}\n"
        )

        file.write(
            f"Confidence: "
            f"{confidence:.6f}\n"
        )

        file.write(
            f"Grad-CAM++ target: "
            f"{target_class_name}\n"
        )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # ========================================================
    # Configuration
    # ========================================================

    cfg = OmegaConf.load(
        "config.yaml"
    )

    print()
    print(
        "=" * 70
    )

    print(
        "CARL GRAD-CAM++"
    )

    print(
        "=" * 70
    )

    print(
        f"Checkpoint : "
        f"{args.checkpoint}"
    )

    print(
        f"Dataset    : "
        f"{cfg.dataset.name}"
    )

    print(
        f"Backbone   : "
        f"{cfg.backbone.name}"
    )

    print(
        f"Target     : "
        f"{args.target}"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # Seed
    # ========================================================

    set_seed(
        int(args.seed)
    )

    # ========================================================
    # Device
    # ========================================================

    if cfg.device == "auto":

        if torch.cuda.is_available():

            device = torch.device(
                "cuda"
            )

        else:

            device = torch.device(
                "cpu"
            )

    else:

        device = torch.device(
            str(cfg.device)
        )

    print(
        f"Device: {device}"
    )

    # ========================================================
    # Data
    # ========================================================

    print()
    print(
        "Building DataLoaders..."
    )

    (
        _,
        _,
        test_loader,
    ) = build_dataloaders(
        cfg
    )

    test_dataset = (
        test_loader.dataset
    )

    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )

    # ========================================================
    # Model
    # ========================================================

    print()
    print(
        "Building CARL model..."
    )

    model = build_model(
        cfg
    ).to(
        device
    )

    # ========================================================
    # Checkpoint
    # ========================================================

    model = load_model_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    model.eval()

    # ========================================================
    # Class names
    # ========================================================

    class_names = get_class_names(
        test_dataset
    )

    print()
    print(
        "Class mapping:"
    )

    for idx, name in enumerate(
        class_names
    ):

        print(
            f"  [{idx}] {name}"
        )

    # ========================================================
    # Target layers
    # ========================================================

    target_layers = get_target_layers(
        model
    )

    print()
    print(
        "Grad-CAM++ target layers:"
    )

    for name, layer in (
        target_layers.items()
    ):

        print(
            f"  {name:10s}: "
            f"{layer.__class__.__name__}"
        )

    # ========================================================
    # Collect predictions
    # ========================================================

    print()
    print(
        "Running test-set inference..."
    )

    (
        labels,
        predictions,
        probabilities,
        records,
    ) = collect_predictions(
        model=model,
        loader=test_loader,
        device=device,
    )

    # ========================================================
    # Summary
    # ========================================================

    correct_mask = (
        labels
        ==
        predictions
    )

    num_correct = int(
        correct_mask.sum()
    )

    num_wrong = int(
        (~correct_mask).sum()
    )

    accuracy = (
        num_correct
        /
        len(labels)
    )

    print()
    print(
        "=" * 70
    )

    print(
        "TEST PREDICTION SUMMARY"
    )

    print(
        f"Total samples    : "
        f"{len(labels)}"
    )

    print(
        f"Correct          : "
        f"{num_correct}"
    )

    print(
        f"Misclassified    : "
        f"{num_wrong}"
    )

    print(
        f"Accuracy         : "
        f"{accuracy:.6f}"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # Select samples
    # ========================================================

    (
        selected_correct,
        selected_wrong,
    ) = select_samples(
        records=records,
        labels=labels,
        predictions=predictions,
        num_correct=args.num_correct,
        num_misclassified=args.num_misclassified,
        seed=args.seed,
    )

    print()
    print(
        f"Selected correct samples       : "
        f"{len(selected_correct)}"
    )

    print(
        f"Selected misclassified samples : "
        f"{len(selected_wrong)}"
    )

    # ========================================================
    # Output
    # ========================================================

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        /
        "gradcam_summary.csv"
    )

    csv_file = csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    )

    csv_writer = csv.writer(
        csv_file
    )

    csv_writer.writerow(
        [
            "index",
            "image_name",
            "patient_id",
            "true_label",
            "true_class",
            "predicted_label",
            "predicted_class",
            "confidence",
            "correct",
            "target_for_gradcam",
            "target_class",
            "category",
        ]
    )

    # ========================================================
    # Combine selected samples
    # ========================================================

    all_selected = []

    for index in selected_correct:

        all_selected.append(
            (
                index,
                "correct",
            )
        )

    for index in selected_wrong:

        all_selected.append(
            (
                index,
                "misclassified",
            )
        )

    print()
    print(
        "=" * 70
    )

    print(
        "Generating Grad-CAM++ visualizations..."
    )

    print(
        "=" * 70
    )

    # ========================================================
    # Process samples
    # ========================================================

    mean = list(
        cfg.dataset.normalization.mean
    )

    std = list(
        cfg.dataset.normalization.std
    )

    for counter, (
        dataset_index,
        category,
    ) in enumerate(
        all_selected,
        start=1,
    ):

        record = records[
            dataset_index
        ]

        # ----------------------------------------------------
        # Get transformed image for model
        # ----------------------------------------------------

        sample = test_dataset[
            dataset_index
        ]

        image_tensor = sample[
            "image"
        ]

        if not torch.is_tensor(
            image_tensor
        ):

            raise RuntimeError(
                "Dataset did not return "
                "a tensor image."
            )

        input_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(
                device,
                non_blocking=True,
            )
        )

        # ----------------------------------------------------
        # Load ORIGINAL image from disk
        # ----------------------------------------------------

        original_image = denormalize_image(
            image_tensor,
            mean=mean,
            std=std,
        )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        true_label = int(
            labels[
                dataset_index
            ]
        )

        predicted_label = int(
            predictions[
                dataset_index
            ]
        )

        confidence = float(
            probabilities[
                dataset_index
            ].max()
        )

        # ----------------------------------------------------
        # Grad-CAM target
        # ----------------------------------------------------

        if args.target == "predicted":

            target_class = (
                predicted_label
            )

        else:

            target_class = (
                true_label
            )

        target_class_name = (
            class_names[
                target_class
            ]
        )

        # ----------------------------------------------------
        # Generate CAMs
        # ----------------------------------------------------

        try:

            cams = (
                generate_single_sample_cams(
                    model=model,
                    target_layers=target_layers,
                    image_tensor=input_tensor,
                    target_class=target_class,
                )
            )

        except Exception as exc:

            print()
            print(
                f"ERROR processing "
                f"{record[test_dataset.image_column]}:"
            )

            print(
                repr(exc)
            )

            raise

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_sample_outputs(
            output_dir=output_dir,
            category=category,
            sample_index=counter,
            image_name=record[
                test_dataset.image_column
            ],
            original_image=original_image,
            cams=cams,
            true_class_name=class_names[
                true_label
            ],
            predicted_class_name=class_names[
                predicted_label
            ],
            confidence=confidence,
            target_class_name=target_class_name,
        )

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        csv_writer.writerow(
            [
                dataset_index,

                record[
                    test_dataset.image_column
                ],

                record[
                    test_dataset.patient_column
                ],

                true_label,

                class_names[
                    true_label
                ],

                predicted_label,

                class_names[
                    predicted_label
                ],

                confidence,

                int(
                    true_label
                    ==
                    predicted_label
                ),

                target_class,

                target_class_name,

                category,
            ]
        )

        # ----------------------------------------------------
        # Console
        # ----------------------------------------------------

        print(
            f"[{counter:03d}/"
            f"{len(all_selected):03d}] "
            f"{category:14s} | "
            f"{record[test_dataset.image_column]} | "
            f"True: "
            f"{class_names[true_label]} | "
            f"Pred: "
            f"{class_names[predicted_label]} | "
            f"Conf: "
            f"{confidence:.4f}"
        )

    # ========================================================
    # Close CSV
    # ========================================================

    csv_file.close()

    # ========================================================
    # Final output
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "Grad-CAM++ generation completed."
    )

    print(
        f"Output directory:"
    )

    print(
        f"  {output_dir.resolve()}"
    )

    print()
    print(
        "Summary CSV:"
    )

    print(
        f"  {csv_path.resolve()}"
    )

    print()
    print(
        "Generated CAM types:"
    )

    print(
        "  backbone -> EfficientNet-B0 features[7]"
    )

    if "cafa" in target_layers:

        print(
            "  cafa     -> CAFA refinement[0]"
        )

    print()
    print(
        "Recommended primary visualization:"
    )

    print(
        "  CAFA Grad-CAM++"
    )

    print()
    print(
        "Note:"
    )

    print(
        "  representation_projection[0] is intentionally "
        "excluded because it is a 1x1 projection layer "
        "and is not an ideal spatial explanation target."
    )

    print(
        "=" * 70
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()