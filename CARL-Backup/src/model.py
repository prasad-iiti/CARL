"""
=========================================================
CARL
Contrastive Adaptive Representation Learning for
Fine-Grained Skin Lesion Classification

Components
----------
1. CNN Backbone
   - ResNet-50
   - EfficientNet-B0

2. CA-AS3
   Cross-Attention Adaptive Scale Selection and
   Spatial-Scale aggregation

3. CAFA
   Channel and Spatial Feature Attention

4. CARL
   Contrastive Adaptive Representation Learning
   projection and feedback module

5. Disease Classification Head

Author: Prasad Kanhegaonkar
=========================================================
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights,
)


# ============================================================
# Utility Functions
# ============================================================

def build_activation(name: str) -> nn.Module:
    """Build activation function from configuration."""

    name = name.lower()

    if name == "relu":
        return nn.ReLU(inplace=True)

    if name == "gelu":
        return nn.GELU()

    if name in {"silu", "swish"}:
        return nn.SiLU(inplace=True)

    if name == "leakyrelu":
        return nn.LeakyReLU(0.01, inplace=True)

    raise ValueError(
        f"Unsupported activation: {name}"
    )


def build_norm(
    name: str,
    channels: int,
) -> nn.Module:
    """Build normalization layer."""

    name = name.lower()

    if name == "batchnorm":
        return nn.BatchNorm1d(channels)

    if name == "layernorm":
        return nn.LayerNorm(channels)

    if name == "identity":
        return nn.Identity()

    raise ValueError(
        f"Unsupported normalization: {name}"
    )


# ============================================================
# ResNet-50 Backbone
# ============================================================

class ResNet50Backbone(nn.Module):
    """
    ResNet-50 feature extractor.

    Exposes:
        layer1
        layer2
        layer3
        layer4
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        weights = (
            ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        model = resnet50(weights=weights)

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.feature_dims = {
            "layer1": 256,
            "layer2": 512,
            "layer3": 1024,
            "layer4": 2048,
        }

        self.output_dim = 2048

        if freeze_backbone:
            self.freeze()

    def freeze(self):
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        return {
            "layer1": layer1,
            "layer2": layer2,
            "layer3": layer3,
            "layer4": layer4,
        }


# ============================================================
# EfficientNet-B0 Backbone
# ============================================================

class EfficientNetB0Backbone(nn.Module):
    """
    EfficientNet-B0 feature extractor.

    Internally maps EfficientNet feature stages to the
    generic CARL feature levels:

        layer2
        layer3
        layer4

    This keeps CA-AS3 independent of the actual backbone.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )

        model = efficientnet_b0(weights=weights)

        self.features = model.features

        # EfficientNet-B0 feature dimensions.
        #
        # We expose three useful spatial levels:
        #
        # layer2 -> 40 channels
        # layer3 -> 112 channels
        # layer4 -> 320 channels

        self.feature_dims = {
            "layer2": 40,
            "layer3": 112,
            "layer4": 320,
        }

        self.output_dim = 1280

        if freeze_backbone:
            self.freeze()

    def freeze(self):
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        features = {}

        for index, block in enumerate(self.features):

            x = block(x)

            if index == 3:
                features["layer2"] = x

            elif index == 5:
                features["layer3"] = x

            elif index == 7:
                features["layer4"] = x

            elif index == 8:
                features["final"] = x

        return features


# ============================================================
# Backbone Builder
# ============================================================

def build_backbone(cfg):
    """
    Build backbone selected by config.yaml.

    Supported:
        resnet50
        efficientnet_b0
    """

    name = str(cfg.backbone.name).lower()

    pretrained = bool(cfg.backbone.pretrained)
    freeze_backbone = bool(
        cfg.backbone.freeze_backbone
    )

    if name == "resnet50":

        backbone = ResNet50Backbone(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    elif name in {
        "efficientnet_b0",
        "efficientnet-b0",
    }:

        backbone = EfficientNetB0Backbone(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    else:

        raise ValueError(
            f"Unsupported backbone: {name}. "
            f"Supported backbones: "
            f"resnet50, efficientnet_b0"
        )

    return backbone


# ============================================================
# CA-AS3
# Cross-Attention Adaptive Scale Selection
# ============================================================

class CAAS3(nn.Module):
    """
    CA-AS3 module.

    Takes multiple backbone feature levels with different
    spatial resolutions and channel dimensions.

    Each feature level is:

        1. projected to a common hidden dimension
        2. spatially pooled into a descriptor
        3. assigned an adaptive scale attention weight
        4. fused into a common representation

    The original spatial feature maps are also retained and
    projected for downstream CAFA processing.
    """

    def __init__(
        self,
        feature_dims: Dict[str, int],
        cfg,
    ):
        super().__init__()

        self.feature_levels = list(
            cfg.ca_as3.feature_levels
        )

        self.hidden_dim = int(
            cfg.ca_as3.hidden_dim
        )

        self.attention_heads = int(
            cfg.ca_as3.attention_heads
        )

        self.dropout = float(
            cfg.ca_as3.dropout
        )

        self.pool_average = bool(
            cfg.ca_as3.pooling.average
        )

        self.pool_maximum = bool(
            cfg.ca_as3.pooling.maximum
        )

        self.pool_variance = bool(
            cfg.ca_as3.pooling.variance
        )

        # ----------------------------------------------------
        # Per-level projections
        # ----------------------------------------------------

        self.projections = nn.ModuleDict()

        for level in self.feature_levels:

            if level not in feature_dims:

                raise ValueError(
                    f"Feature level '{level}' is not "
                    f"available in the selected backbone."
                )

            self.projections[level] = nn.Sequential(
                nn.Conv2d(
                    feature_dims[level],
                    self.hidden_dim,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    self.hidden_dim
                ),
                nn.ReLU(inplace=True),
            )

        # ----------------------------------------------------
        # Scale attention
        # ----------------------------------------------------

        self.scale_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.attention_heads,
            dropout=self.dropout,
            batch_first=True,
        )

        self.scale_norm = nn.LayerNorm(
            self.hidden_dim
        )

        # ----------------------------------------------------
        # Adaptive scale scoring
        # ----------------------------------------------------

        self.scale_score = nn.Sequential(
            nn.Linear(
                self.hidden_dim,
                self.hidden_dim // 2,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(
                self.hidden_dim // 2,
                1,
            ),
        )

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        self.fusion = nn.Sequential(
            nn.Conv2d(
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                self.hidden_dim
            ),
            nn.ReLU(inplace=True),
        )

    def _pool_descriptor(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Produce a descriptor from a feature map using the
        configured pooling operations.
        """

        descriptors = []

        if self.pool_average:

            descriptors.append(
                F.adaptive_avg_pool2d(
                    x,
                    1,
                ).flatten(1)
            )

        if self.pool_maximum:

            descriptors.append(
                F.adaptive_max_pool2d(
                    x,
                    1,
                ).flatten(1)
            )

        if self.pool_variance:

            mean = x.mean(
                dim=(2, 3),
                keepdim=True,
            )

            variance = (
                (x - mean) ** 2
            ).mean(
                dim=(2, 3)
            )

            descriptors.append(
                variance
            )

        if not descriptors:

            raise ValueError(
                "CA-AS3 requires at least one "
                "pooling operation."
            )

        descriptor = torch.stack(
            descriptors,
            dim=0,
        ).mean(dim=0)

        return descriptor

    def forward(
        self,
        feature_maps: Dict[str, torch.Tensor],
    ) -> Tuple[
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Returns
        -------
        projected_features:
            Dictionary of projected feature maps.

        fused_feature:
            Adaptive multi-scale fused feature map.

        scale_weights:
            Adaptive scale-selection weights.
        """

        projected = {}
        descriptors = []

        for level in self.feature_levels:

            x = feature_maps[level]

            x = self.projections[level](x)

            projected[level] = x

            descriptor = self._pool_descriptor(x)

            descriptors.append(
                descriptor
            )

        # [B, number_of_levels, hidden_dim]

        tokens = torch.stack(
            descriptors,
            dim=1,
        )

        # Self cross-attention among scales.

        attended, _ = self.scale_attention(
            tokens,
            tokens,
            tokens,
            need_weights=False,
        )

        tokens = self.scale_norm(
            tokens + attended
        )

        # Adaptive scale scores.

        scores = self.scale_score(
            tokens
        ).squeeze(-1)

        scale_weights = F.softmax(
            scores,
            dim=1,
        )

        # ----------------------------------------------------
        # Align all feature maps to highest spatial resolution
        # among selected levels.
        # ----------------------------------------------------

        target_level = self.feature_levels[0]

        target_h = projected[
            target_level
        ].shape[-2]

        target_w = projected[
            target_level
        ].shape[-1]

        fused = None

        for index, level in enumerate(
            self.feature_levels
        ):

            x = projected[level]

            if (
                x.shape[-2] != target_h
                or x.shape[-1] != target_w
            ):

                x = F.interpolate(
                    x,
                    size=(
                        target_h,
                        target_w,
                    ),
                    mode="bilinear",
                    align_corners=False,
                )

            weight = scale_weights[
                :, index
            ].view(
                -1,
                1,
                1,
                1,
            )

            weighted = x * weight

            if fused is None:
                fused = weighted
            else:
                fused = fused + weighted

        fused = self.fusion(fused)

        return (
            projected,
            fused,
            scale_weights,
        )


# ============================================================
# CAFA
# Channel and Spatial Feature Attention
# ============================================================

class ChannelAttention(nn.Module):
    """
    Channel attention component of CAFA.
    """

    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
    ):
        super().__init__()

        hidden = max(
            channels // reduction_ratio,
            1,
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                channels,
                kernel_size=1,
                bias=False,
            ),
        )

        self.activation = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        avg_attention = self.mlp(
            self.avg_pool(x)
        )

        max_attention = self.mlp(
            self.max_pool(x)
        )

        attention = (
            avg_attention
            + max_attention
        )

        return self.activation(
            attention
        )


class SpatialAttention(nn.Module):
    """
    Spatial attention component of CAFA.
    """

    def __init__(
        self,
        kernel_size: int = 7,
    ):
        super().__init__()

        if kernel_size not in {
            3,
            7,
        }:

            raise ValueError(
                "Spatial attention kernel_size "
                "must be 3 or 7."
            )

        padding = (
            kernel_size - 1
        ) // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.activation = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        avg_map = x.mean(
            dim=1,
            keepdim=True,
        )

        max_map = x.max(
            dim=1,
            keepdim=True,
        ).values

        combined = torch.cat(
            [
                avg_map,
                max_map,
            ],
            dim=1,
        )

        return self.activation(
            self.conv(combined)
        )


class CAFA(nn.Module):
    """
    Channel and Spatial Feature Attention.

    Sequentially applies:

        channel attention
              ↓
        spatial attention
              ↓
        residual refinement
    """

    def __init__(
        self,
        channels: int,
        cfg,
    ):
        super().__init__()

        self.channel_enabled = bool(
            cfg.cafa.channel_attention
        )

        self.spatial_enabled = bool(
            cfg.cafa.spatial_attention
        )

        if self.channel_enabled:

            self.channel_attention = (
                ChannelAttention(
                    channels=channels,
                    reduction_ratio=int(
                        cfg.cafa.reduction_ratio
                    ),
                )
            )

        else:

            self.channel_attention = None

        if self.spatial_enabled:

            self.spatial_attention = (
                SpatialAttention(
                    kernel_size=int(
                        cfg.cafa.kernel_size
                    )
                )
            )

        else:

            self.spatial_attention = None

        self.refinement = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                channels
            ),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        residual = x

        if self.channel_attention is not None:

            channel_weights = (
                self.channel_attention(x)
            )

            x = x * channel_weights

        if self.spatial_attention is not None:

            spatial_weights = (
                self.spatial_attention(x)
            )

            x = x * spatial_weights

        x = self.refinement(x)

        x = x + residual

        return x


# ============================================================
# CARL Projection Head
# ============================================================

class CARLProjectionHead(nn.Module):
    """
    CARL projection head.

    Produces normalized embeddings for supervised
    contrastive learning.

    Optional feedback can be used to adapt the
    representation using the contrastive embedding.
    """

    def __init__(
        self,
        feature_dim: int,
        cfg,
    ):
        super().__init__()

        self.embedding_dim = int(
            cfg.carl.embedding_dim
        )

        self.hidden_dim = int(
            cfg.carl.proj_hidden_dim
        )

        self.num_layers = int(
            cfg.carl.num_layers
        )

        self.dropout = float(
            cfg.carl.proj_dropout
        )

        self.activation_name = str(
            cfg.carl.proj_activation
        )

        self.normalization = str(
            cfg.carl.normalization
        )

        self.feedback_strength = float(
            cfg.carl.feedback_strength
        )

        self.training_feedback = bool(
            cfg.carl.training_feedback
        )

        layers = []

        current_dim = feature_dim

        for layer_index in range(
            self.num_layers - 1
        ):

            layers.append(
                nn.Linear(
                    current_dim,
                    self.hidden_dim,
                )
            )

            layers.append(
                build_norm(
                    self.normalization,
                    self.hidden_dim,
                )
            )

            layers.append(
                build_activation(
                    self.activation_name
                )
            )

            layers.append(
                nn.Dropout(
                    self.dropout
                )
            )

            current_dim = self.hidden_dim

        layers.append(
            nn.Linear(
                current_dim,
                self.embedding_dim,
            )
        )

        self.projector = nn.Sequential(
            *layers
        )

        # ----------------------------------------------------
        # Feedback transformation
        # ----------------------------------------------------

        self.feedback = nn.Sequential(
            nn.Linear(
                self.embedding_dim,
                feature_dim,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: torch.Tensor,
        epoch: int = 0,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        embedding = self.projector(
            features
        )

        embedding = F.normalize(
            embedding,
            dim=1,
        )

        use_feedback = (
            self.training
            and self.training_feedback
            and epoch >= 0
        )

        if use_feedback:

            feedback_gate = self.feedback(
                embedding
            )

            adapted_features = (
                features
                * (
                    1.0
                    + self.feedback_strength
                    * feedback_gate
                )
            )

        else:

            adapted_features = features

        return (
            embedding,
            adapted_features,
        )


# ============================================================
# Classifier Head
# ============================================================

class ClassifierHead(nn.Module):
    """
    Multiclass disease classification head.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
    ):
        super().__init__()

        self.classifier = nn.Linear(
            in_features,
            num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.classifier(x)


# ============================================================
# CARL Model
# ============================================================

class CARLModel(nn.Module):
    """
    Complete CARL model.

    Pipeline
    --------
    Image
      ↓
    Backbone
      ↓
    Multi-scale features
      ↓
    CA-AS3
      ↓
    CAFA
      ↓
    Global representation
      ↓
    CARL projection
      ↓
    Classifier
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # ----------------------------------------------------
        # Backbone
        # ----------------------------------------------------

        self.backbone = build_backbone(
            cfg
        )

        self.backbone_name = str(
            cfg.backbone.name
        ).lower()

        self.feature_dim = (
            self.backbone.output_dim
        )

        # ----------------------------------------------------
        # CA-AS3
        # ----------------------------------------------------

        self.ca_as3 = None

        if bool(
            cfg.ca_as3.enabled
        ):

            self.ca_as3 = CAAS3(
                feature_dims=self.backbone.feature_dims,
                cfg=cfg,
            )

            ca_as3_dim = int(
                cfg.ca_as3.hidden_dim
            )

        else:

            ca_as3_dim = self.feature_dim

        # ----------------------------------------------------
        # CAFA
        # ----------------------------------------------------

        self.cafa = None

        if bool(
            cfg.cafa.enabled
        ):

            self.cafa = CAFA(
                channels=ca_as3_dim,
                cfg=cfg,
            )

        # ----------------------------------------------------
        # Representation projection
        # ----------------------------------------------------

        self.representation_projection = (
            nn.Sequential(
                nn.Conv2d(
                    ca_as3_dim,
                    self.feature_dim,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    self.feature_dim
                ),
                nn.ReLU(inplace=True),
            )
        )

        # ----------------------------------------------------
        # Global pooling
        # ----------------------------------------------------

        self.global_pool = (
            nn.AdaptiveAvgPool2d(1)
        )

        # ----------------------------------------------------
        # CARL
        # ----------------------------------------------------

        self.carl = None

        if bool(
            cfg.carl.enabled
        ):

            self.carl = CARLProjectionHead(
                feature_dim=self.feature_dim,
                cfg=cfg,
            )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        num_classes = int(
            cfg.dataset.registry[
                cfg.dataset.name
            ].num_classes
        )

        self.num_classes = num_classes

        self.classifier = ClassifierHead(
            in_features=self.feature_dim,
            num_classes=num_classes,
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x: torch.Tensor,
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x:
            Input image tensor [B,3,H,W]

        epoch:
            Current training epoch.

        Returns
        -------
        Dictionary containing:
            logits
            features
            embedding
            scale_weights
        """

        # ----------------------------------------------------
        # Backbone
        # ----------------------------------------------------

        feature_maps = self.backbone(x)

        # ----------------------------------------------------
        # CA-AS3
        # ----------------------------------------------------

        scale_weights = None

        if self.ca_as3 is not None:

            (
                projected_features,
                fused_features,
                scale_weights,
            ) = self.ca_as3(
                feature_maps
            )

        else:

            # Use final backbone feature map.

            if "final" in feature_maps:

                fused_features = (
                    feature_maps["final"]
                )

            else:

                fused_features = (
                    feature_maps["layer4"]
                )

            projected_features = {
                "final": fused_features
            }

        # ----------------------------------------------------
        # CAFA
        # ----------------------------------------------------

        if self.cafa is not None:

            refined_features = self.cafa(
                fused_features
            )

        else:

            refined_features = (
                fused_features
            )

        # ----------------------------------------------------
        # Map to backbone representation dimension
        # ----------------------------------------------------

        refined_features = (
            self.representation_projection(
                refined_features
            )
        )

        # ----------------------------------------------------
        # Global average pooling
        # ----------------------------------------------------

        features = self.global_pool(
            refined_features
        )

        features = torch.flatten(
            features,
            1,
        )

        # ----------------------------------------------------
        # CARL projection
        # ----------------------------------------------------

        embedding = None

        if self.carl is not None:

            (
                embedding,
                adapted_features,
            ) = self.carl(
                features,
                epoch=epoch,
            )

            # During feedback training, use the
            # adapted representation for classification.

            if (
                self.training
                and bool(
                    self.cfg.carl.training_feedback
                )
                and epoch
                >= int(
                    self.cfg.carl.feedback_start_epoch
                )
            ):

                classifier_features = (
                    adapted_features
                )

            else:

                classifier_features = (
                    features
                )

        else:

            classifier_features = features

        # ----------------------------------------------------
        # Disease classification
        # ----------------------------------------------------

        logits = self.classifier(
            classifier_features
        )

        # ----------------------------------------------------
        # Output dictionary
        # ----------------------------------------------------

        outputs = {
            "logits": logits,
            "features": classifier_features,
            "embedding": embedding,
            "backbone_features": feature_maps,
            "ca_as3_features": projected_features,
            "refined_features": refined_features,
        }

        if scale_weights is not None:

            outputs[
                "scale_weights"
            ] = scale_weights

        return outputs


# ============================================================
# Public Model Builder
# ============================================================

def build_model(cfg) -> CARLModel:
    """
    Public model construction API.
    """

    return CARLModel(cfg)


# ============================================================
# Parameter Groups
# ============================================================

def get_parameter_groups(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
):
    """
    Create parameter groups for AdamW.

    Backbone:
        backbone_lr

    CARL / CA-AS3 / CAFA / classifier:
        head_lr

    This matches the current config:

        backbone_lr = 3e-5
        head_lr     = 3e-4
    """

    backbone_parameters = []
    head_parameters = []

    backbone_parameter_ids = set()

    for parameter in model.backbone.parameters():

        if parameter.requires_grad:

            backbone_parameters.append(
                parameter
            )

            backbone_parameter_ids.add(
                id(parameter)
            )

    for parameter in model.parameters():

        if not parameter.requires_grad:
            continue

        if id(parameter) not in (
            backbone_parameter_ids
        ):

            head_parameters.append(
                parameter
            )

    return [
        {
            "params": backbone_parameters,
            "lr": backbone_lr,
        },
        {
            "params": head_parameters,
            "lr": head_lr,
        },
    ]


# ============================================================
# Model Summary
# ============================================================

def count_parameters(
    model: nn.Module,
) -> Tuple[int, int]:
    """
    Return total and trainable parameter counts.
    """

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# ============================================================
# Sanity Check
# ============================================================

if __name__ == "__main__":

    from omegaconf import OmegaConf

    print("=" * 70)
    print("CARL MODEL SANITY CHECK")
    print("=" * 70)

    cfg = OmegaConf.load(
        "/home/prasad/CARL/config.yaml"
    )

    print(
        f"Dataset  : {cfg.dataset.name}"
    )

    print(
        f"Backbone : {cfg.backbone.name}"
    )

    print()

    model = build_model(cfg)

    total, trainable = count_parameters(
        model
    )

    print(
        f"Total parameters     : "
        f"{total:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable:,}"
    )

    print()

    # --------------------------------------------------------
    # Dummy input
    # --------------------------------------------------------

    image_size = int(
        cfg.dataset.image_size
    )

    x = torch.randn(
        2,
        3,
        image_size,
        image_size,
    )

    model.eval()

    with torch.no_grad():

        outputs = model(
            x,
            epoch=0,
        )

    print("Outputs")
    print("-" * 70)

    for key, value in outputs.items():

        if isinstance(value, dict):

            print(
                f"{key}:"
            )

            for name, tensor in value.items():

                if torch.is_tensor(tensor):

                    print(
                        f"  {name:<15} "
                        f"{tuple(tensor.shape)}"
                    )

        elif torch.is_tensor(value):

            print(
                f"{key:<25} "
                f"{tuple(value.shape)}"
            )

    print()

    print(
        "Model sanity check completed."
    )

    print("=" * 70)