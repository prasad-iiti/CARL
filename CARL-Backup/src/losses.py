import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# Focal Loss
# =========================================================

class FocalLoss(nn.Module):
    """
    Class-weighted Focal Loss.

    L = -alpha_y * (1 - p_t)^gamma * log(p_t)

    alpha_y:
        Class weight corresponding to the target class.

    gamma:
        Focusing parameter.
    """
        
    def __init__(self, alpha=None, gamma=5.0, reduction="mean"):
        super().__init__()

        self.gamma = float(gamma)
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits, targets):

        # Log probabilities
        log_probs = F.log_softmax(logits, dim=1)

        # Log probability of the true class
        log_pt = log_probs.gather(
            1,
            targets.unsqueeze(1)
        ).squeeze(1)

        pt = log_pt.exp()

        # Focal modulation
        focal_factor = (1.0 - pt).pow(self.gamma)

        # Focal loss
        loss = -focal_factor * log_pt

        # Optional class weighting
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        # reduction
        if self.reduction == "mean":
            return loss.mean()

        elif self.reduction == "sum":
            return loss.sum()

        else:
            return loss


# =========================================================
# Supervised Contrastive Loss
# =========================================================

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Samples belonging to the same class are treated as
    positive pairs, while samples from different classes
    are treated as negatives.

    The input features are normalized internally.
    """

    def __init__(self, temperature=0.1):
        super().__init__()

        if temperature <= 0:
            raise ValueError(
                "temperature must be > 0"
            )

        self.temperature = float(
            temperature
        )

    def forward(self, features, labels):

        if features.ndim != 2:
            raise ValueError(
                "features must have shape "
                "[batch_size, embedding_dim]"
            )

        device = features.device

        labels = labels.to(device).view(-1)

        # -------------------------------------------------
        # Normalize embeddings
        # -------------------------------------------------

        features = F.normalize(
            features,
            p=2,
            dim=1,
        )

        batch_size = features.size(0)

        if labels.size(0) != batch_size:
            raise ValueError(
                "Number of labels must match "
                "number of feature vectors."
            )

        # -------------------------------------------------
        # Pairwise similarity
        # -------------------------------------------------

        logits = torch.matmul(
            features,
            features.T,
        )

        logits = logits / self.temperature

        # -------------------------------------------------
        # Remove self-similarity
        # -------------------------------------------------

        self_mask = torch.eye(
            batch_size,
            dtype=torch.bool,
            device=device,
        )

        logits = logits.masked_fill(
            self_mask,
            torch.finfo(logits.dtype).min,
        )

        # -------------------------------------------------
        # Positive-pair mask
        # -------------------------------------------------

        positive_mask = (
            labels.unsqueeze(0)
            == labels.unsqueeze(1)
        ).float()

        positive_mask = positive_mask.masked_fill(
            self_mask,
            0.0,
        )

        # -------------------------------------------------
        # Log probability
        # -------------------------------------------------

        log_prob = F.log_softmax(
            logits,
            dim=1,
        )

        # -------------------------------------------------
        # Number of positive samples
        # -------------------------------------------------

        positive_count = positive_mask.sum(
            dim=1
        )

        valid = positive_count > 0

        # If no positive pair exists in the batch,
        # return a differentiable zero loss.
        if not valid.any():

            return torch.zeros(
                (),
                device=device,
                dtype=features.dtype,
                requires_grad=True,
            )

        # -------------------------------------------------
        # Mean positive log probability
        # -------------------------------------------------

        mean_log_prob_pos = (
            positive_mask * log_prob
        ).sum(dim=1) / (
            positive_count + 1e-8
        )

        loss = -mean_log_prob_pos[valid].mean()

        return loss


# =========================================================
# CARL Combined Loss
# =========================================================

class Losses(nn.Module):
    """
    CARL training objective.

    L =
        lambda_ce     * Classification Loss
      + lambda_metric * Supervised Contrastive Loss

    Classification:
        Weighted Cross Entropy

    Metric:
        Supervised Contrastive Loss
    """

    def __init__(
        self,
        cfg,
        class_weights=None,
    ):
        super().__init__()

        self.cfg = cfg

        loss_cfg = cfg.loss

        # -------------------------------------------------
        # Loss weights
        # -------------------------------------------------

        self.classification_weight = float(
            loss_cfg.get(
                "classification_weight",
                1.0,
            )
        )

        self.metric_weight = float(
            loss_cfg.get(
                "metric_weight",
                0.01,
            )
        )

        # self.ce_weight = float(
        #     loss_cfg.get(
        #         "ce_weight",
        #         1.0,
        #     )
        # )

        # self.metric_weight = float(
        #     loss_cfg.get(
        #         "metric_weight",
        #         0.05,
        #     )
        # )

        # -------------------------------------------------
        # Classification loss
        # -------------------------------------------------

        classification_name = str(
            loss_cfg.get(
                "classification",
                "focal",
            )
        ).lower()

        if classification_name in [
            "focal",
            "focal_loss",
        ]:

            gamma = float(
                loss_cfg.get(
                    "focal_gamma",
                    2.0,
                )
            )

            use_class_weights = bool(
                loss_cfg.get(
                    "focal_use_class_weights",
                    False,
                )
            )

            alpha = None

            if use_class_weights and class_weights is not None:
                alpha = class_weights

            self.classification_loss = FocalLoss(
                alpha=alpha,
                gamma=gamma,
                reduction="mean",
            )

        else:

            raise ValueError(
                f"Unsupported classification loss: "
                f"{classification_name}"
            )

        # classification_name = str(
        #     loss_cfg.get(
        #         "classification",
        #         "cross_entropy",
        #     )
        # ).lower()

        # if classification_name in [
        #     "cross_entropy",
        #     "weighted_cross_entropy",
        # ]:

        #     if (
        #         classification_name
        #         == "weighted_cross_entropy"
        #         and class_weights is not None
        #     ):

        #         self.register_buffer(
        #             "class_weights",
        #             class_weights.float(),
        #         )

        #         self.classification_loss = (
        #             nn.CrossEntropyLoss(
        #                 weight=self.class_weights
        #             )
        #         )

        #     else:

        #         self.classification_loss = (
        #             nn.CrossEntropyLoss()
        #         )

        # else:

        #     raise ValueError(
        #         f"Unsupported classification loss: "
        #         f"{classification_name}"
        #     )

        carl_cfg = cfg.get("carl", {})

        self.carl_enabled = bool(
            carl_cfg.get("enabled", False)
        )

        # -------------------------------------------------
        # Metric loss
        # -------------------------------------------------

        metric_name = str(
            loss_cfg.get(
                "metric",
                "supervised_contrastive",
            )
        ).lower()

        if metric_name in [
            "supervised_contrastive",
            "supcon",
        ]:

            temperature = float(
                cfg.carl.get(
                    "temperature",
                    0.1,
                )
            )

            self.metric_loss = (
                SupervisedContrastiveLoss(
                    temperature=temperature
                )
            )

        else:

            raise ValueError(
                f"Unsupported metric loss: "
                f"{metric_name}"
            )

    # =====================================================
    # Forward
    # =====================================================

    def forward(
        self,
        outputs,
        labels,
        metadata=None,
    ):
        """
        Parameters
        ----------
        outputs : dict
            Output dictionary returned by CARLModel.

        labels : Tensor
            Ground-truth class labels.

        metadata : optional
            Metadata dictionary. Currently not required
            by the CARL objective.

        Returns
        -------
        dict
            loss
            ce_loss
            metric_loss
        """

        # -------------------------------------------------
        # Device
        # -------------------------------------------------

        labels = labels.long()

        # -------------------------------------------------
        # Classification
        # -------------------------------------------------

        logits = outputs["logits"]

        focal_loss = self.classification_loss(
            logits,
            labels,
        )

        # ce_loss = self.classification_loss(
        #     logits,
        #     labels,
        # )


        # -------------------------------------------------
        # Supervised Contrastive Loss
        # -------------------------------------------------

        # if self.metric_weight > 0.0:
        if self.carl_enabled and self.metric_weight > 0.0:

            embedding = outputs.get(
                "embedding",
                None,
            )

            if embedding is None:

                raise KeyError(
                    "CARL metric loss is enabled "
                    "(metric_weight > 0), but model output "
                    "does not contain 'embedding'."
                )

            metric_loss = self.metric_loss(
                embedding,
                labels,
            )

        else:

            # CARL disabled for this ablation
            metric_loss = torch.zeros(
                (),
                device=logits.device,
                # dtype=ce_loss.dtype,
                dtype=focal_loss.dtype,
            )

        # # -------------------------------------------------
        # # Supervised Contrastive Loss
        # # -------------------------------------------------

        # embedding = outputs.get(
        #     "embedding",
        #     None,
        # )

        # if embedding is None:

        #     raise KeyError(
        #         "CARL model output does not contain "
        #         "'embedding'."
        #     )

        # metric_loss = self.metric_loss(
        #     embedding,
        #     labels,
        # )

        # -------------------------------------------------
        # Total loss
        # -------------------------------------------------

        total_loss = (
            self.classification_weight * focal_loss
            + self.metric_weight * metric_loss
        )

        # total_loss = (
        #     self.ce_weight * ce_loss
        #     + self.metric_weight * metric_loss
        # )

        # -------------------------------------------------
        # Return losses
        # -------------------------------------------------

        return {
            "loss": total_loss,
            "focal_loss": focal_loss.detach(),
            "metric_loss": metric_loss.detach(),
        }

        # return {
        #     "loss": total_loss,
        #     "ce_loss": ce_loss.detach(),
        #     "metric_loss": metric_loss.detach(),
        # }


# =========================================================
# Sanity Check
# =========================================================

if __name__ == "__main__":

    from omegaconf import OmegaConf

    print("=" * 70)
    print("CARL LOSS SANITY CHECK")
    print("=" * 70)

    cfg = OmegaConf.load(
        "/home/prasad/CARL/config.yaml"
    )

    num_classes = int(
        cfg.dataset.registry[
            cfg.dataset.name
        ].num_classes
    )

    embedding_dim = int(
        cfg.carl.embedding_dim
    )

    batch_size = 8

    # -----------------------------------------------------
    # Dummy outputs
    # -----------------------------------------------------

    outputs = {

        "logits": torch.randn(
            batch_size,
            num_classes,
            requires_grad=True,
        ),

        "embedding": torch.randn(
            batch_size,
            embedding_dim,
            requires_grad=True,
        ),
    }

    labels = torch.tensor(
        [
            0,
            0,
            1,
            1,
            2,
            2,
            3,
            3,
        ],
        dtype=torch.long,
    )

    # -----------------------------------------------------
    # Class weights
    # -----------------------------------------------------

    class_weights = torch.ones(
        num_classes
    )

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    criterion = Losses(
        cfg,
        class_weights=class_weights,
    )

    losses = criterion(
        outputs,
        labels,
    )

    print(
        f"Dataset       : "
        f"{cfg.dataset.name}"
    )

    print(
        f"Num classes   : "
        f"{num_classes}"
    )

    print(
        f"Embedding dim : "
        f"{embedding_dim}"
    )

    print()

    for name, value in losses.items():

        print(
            f"{name:<15}: "
            f"{value.item():.6f}"
        )

    print()

    # -----------------------------------------------------
    # Backward test
    # -----------------------------------------------------

    losses["loss"].backward()

    print(
        "Backward pass : OK"
    )

    print()
    print(
        "CARL loss sanity check completed."
    )

    print("=" * 70)