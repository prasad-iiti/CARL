# ============================================================
# CARL - Unified Evaluation Module
# ============================================================

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from src.metrics import (
        classification_summary,
        fairness_summary,
        uncertainty_summary,
    )
except ModuleNotFoundError:
    from metrics import (
        classification_summary,
        fairness_summary,
        uncertainty_summary,
    )


# ============================================================
# Evaluator
# ============================================================

class Evaluator:
    """
    Unified evaluation module for CARL.

    Responsibilities
    ----------------
    • Run inference
    • Compute evaluation losses
    • Collect predictions
    • Collect probabilities
    • Collect CARL embeddings
    • Collect metadata
    • Compute classification metrics
    • Compute fairness metrics
    • Compute uncertainty metrics
    • Save evaluation artifacts
    """

    def __init__(
        self,
        cfg,
        output_dir=None,
    ):
        self.cfg = cfg

        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else None
        )

        if self.output_dir is not None:

            (self.output_dir / "metrics").mkdir(
                parents=True,
                exist_ok=True,
            )

            (self.output_dir / "predictions").mkdir(
                parents=True,
                exist_ok=True,
            )

            if cfg.output.get(
                "save_embeddings",
                False,
            ):
                (
                    self.output_dir / "embeddings"
                ).mkdir(
                    parents=True,
                    exist_ok=True,
                )

    # ========================================================
    # JSON utilities
    # ========================================================

    @staticmethod
    def _json_serializer(obj):
        """
        Convert NumPy/PyTorch-compatible objects
        into JSON serializable objects.
        """

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if torch.is_tensor(obj):
            return obj.detach().cpu().tolist()

        if hasattr(obj, "tolist"):
            return obj.tolist()

        raise TypeError(
            f"Object of type {type(obj)} "
            f"is not JSON serializable."
        )

    # ========================================================

    @staticmethod
    def convert_keys(obj):
        """
        Recursively convert NumPy keys/values
        into JSON-compatible Python objects.
        """

        if isinstance(obj, dict):

            converted = {}

            for key, value in obj.items():

                if isinstance(key, np.integer):
                    key = int(key)

                elif isinstance(key, np.floating):
                    key = float(key)

                else:
                    key = str(key)

                converted[key] = (
                    Evaluator.convert_keys(value)
                )

            return converted

        if isinstance(obj, list):

            return [
                Evaluator.convert_keys(value)
                for value in obj
            ]

        if isinstance(obj, tuple):

            return [
                Evaluator.convert_keys(value)
                for value in obj
            ]

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if torch.is_tensor(obj):
            return obj.detach().cpu().tolist()

        return obj

    # ========================================================
    # Metadata utilities
    # ========================================================

    @staticmethod
    def _move_metadata_to_device(
        metadata,
        device,
    ):
        """
        Move tensor metadata to the selected device.

        Non-tensor metadata is preserved.
        """

        if metadata is None:
            return {}

        if not isinstance(metadata, dict):
            return {}

        moved = {}

        for key, value in metadata.items():

            if torch.is_tensor(value):

                moved[key] = value.to(
                    device,
                    non_blocking=True,
                )

            else:

                moved[key] = value

        return moved

    # ========================================================

    @staticmethod
    def _collect_metadata(
        storage,
        metadata,
    ):
        """
        Collect metadata on CPU for later concatenation.
        """

        if not metadata:
            return storage

        for key, value in metadata.items():

            if key not in storage:
                storage[key] = []

            if torch.is_tensor(value):

                storage[key].append(
                    value.detach().cpu()
                )

            elif isinstance(value, np.ndarray):

                storage[key].append(value)

            else:

                storage[key].append(value)

        return storage

    # ========================================================

    @staticmethod
    def _finalize_metadata(metadata_storage):
        """
        Convert accumulated batch-wise metadata into
        sample-wise NumPy arrays.

        Supports:
            - torch tensors
            - NumPy arrays
            - Python lists
            - tuples
            - scalar values

        The final output contains one metadata entry per sample.
        """

        metadata = {}

        for key, values in metadata_storage.items():

            if not values:
                continue

            first = values[0]

            # --------------------------------------------------
            # Torch tensors
            # --------------------------------------------------
            if torch.is_tensor(first):

                metadata[key] = torch.cat(
                    [
                        value.detach().cpu()
                        for value in values
                    ],
                    dim=0,
                ).numpy()

            # --------------------------------------------------
            # NumPy arrays
            # --------------------------------------------------
            elif isinstance(first, np.ndarray):

                metadata[key] = np.concatenate(
                    values,
                    axis=0,
                )

            # --------------------------------------------------
            # Python lists / tuples
            # --------------------------------------------------
            elif isinstance(first, (list, tuple)):

                flattened = []

                for value in values:
                    flattened.extend(value)

                metadata[key] = np.asarray(
                    flattened,
                    dtype=object,
                )

            # --------------------------------------------------
            # Scalar / other values
            # --------------------------------------------------
            else:

                metadata[key] = np.asarray(
                    values,
                    dtype=object,
                )

        return metadata

    # ========================================================
    # Output extraction
    # ========================================================

    @staticmethod
    def _extract_embedding(outputs):
        """
        Extract CARL embedding from model outputs.

        Expected CARL output:
            outputs["embedding"]

        Returns None if unavailable.
        """

        embedding = outputs.get(
            "embedding",
            None,
        )

        if embedding is None:
            return None

        if not torch.is_tensor(embedding):
            raise TypeError(
                "Model output 'embedding' "
                "must be a torch.Tensor."
            )

        return embedding

    # ========================================================

    @staticmethod
    def _extract_logits(outputs):
        """
        Extract classification logits.
        """

        logits = outputs.get(
            "logits",
            None,
        )

        if logits is None:
            raise KeyError(
                "Model did not return 'logits'."
            )

        if not torch.is_tensor(logits):
            raise TypeError(
                "Model output 'logits' "
                "must be a torch.Tensor."
            )

        return logits

    # ========================================================
    # Evaluation
    # ========================================================

    @torch.no_grad()
    def evaluate(
        self,
        model,
        loader,
        loss_fn,
        device,
        split="val",
    ):
        """
        Evaluate CARL on validation/test data.

        Parameters
        ----------
        model:
            CARL model.

        loader:
            DataLoader.

        loss_fn:
            CARL loss module.

        device:
            torch.device.

        split:
            Dataset split name.

        Returns
        -------
        dict
            Metrics, labels, predictions,
            probabilities, metadata and embeddings.
        """

        model.eval()

        total_samples = 0

        total_loss = 0.0
        total_ce = 0.0
        total_metric = 0.0

        all_probabilities = []
        all_predictions = []
        all_labels = []

        all_embeddings = []

        metadata_storage = {}

        # ----------------------------------------------------
        # Iterate over loader
        # ----------------------------------------------------

        for batch in loader:

            if "image" not in batch:
                raise KeyError(
                    "Evaluation batch does not contain "
                    "'image'."
                )

            if "label" not in batch:
                raise KeyError(
                    "Evaluation batch does not contain "
                    "'label'."
                )

            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            labels = batch["label"].to(
                device,
                non_blocking=True,
            )

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            batch_metadata = batch.get(
                "metadata",
                {},
            )

            metadata = (
                self._move_metadata_to_device(
                    batch_metadata,
                    device,
                )
            )

            metadata_storage = (
                self._collect_metadata(
                    metadata_storage,
                    batch_metadata,
                )
            )

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            outputs = model(
                images,
                epoch=0,
            )

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss_dict = loss_fn(
                outputs=outputs,
                labels=labels,
                metadata=metadata,
            )

            batch_size = labels.size(0)

            total_samples += batch_size

            total_loss += (
                float(
                    loss_dict["loss"].detach().item()
                )
                * batch_size
            )

            if "focal_loss" in loss_dict:

                total_ce += (
                    float(
                        loss_dict["focal_loss"]
                        .detach()
                        .item()
                    )
                    * batch_size
                )

            if "metric_loss" in loss_dict:

                total_metric += (
                    float(
                        loss_dict["metric_loss"]
                        .detach()
                        .item()
                    )
                    * batch_size
                )

            # ------------------------------------------------
            # Classification output
            # ------------------------------------------------

            logits = self._extract_logits(
                outputs
            )

            probabilities = F.softmax(
                logits,
                dim=1,
            )

            predictions = probabilities.argmax(
                dim=1
            )

            # ------------------------------------------------
            # Store predictions
            # ------------------------------------------------

            all_probabilities.append(
                probabilities.detach().cpu()
            )

            all_predictions.append(
                predictions.detach().cpu()
            )

            all_labels.append(
                labels.detach().cpu()
            )

            # ------------------------------------------------
            # CARL embedding
            # ------------------------------------------------

            embedding = self._extract_embedding(
                outputs
            )

            if embedding is not None:

                all_embeddings.append(
                    embedding.detach().cpu()
                )

        # ====================================================
        # Empty loader check
        # ====================================================

        if total_samples == 0:

            raise ValueError(
                "Evaluation loader is empty."
            )

        # ====================================================
        # Concatenate outputs
        # ====================================================

        probabilities = torch.cat(
            all_probabilities,
            dim=0,
        ).numpy()

        predictions = torch.cat(
            all_predictions,
            dim=0,
        ).numpy()

        labels = torch.cat(
            all_labels,
            dim=0,
        ).numpy()

        metadata = self._finalize_metadata(
            metadata_storage
        )

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        if all_embeddings:

            embeddings = torch.cat(
                all_embeddings,
                dim=0,
            ).numpy()

        else:

            embeddings = None

        # ====================================================
        # Metrics
        # ====================================================

        classification_metrics = (
            classification_summary(
                probabilities,
                labels,
            )
        )

        # ----------------------------------------------------
        # Fairness
        # ----------------------------------------------------

        fairness_metrics = {}

        if metadata:

            try:

                fairness_metrics = (
                    fairness_summary(
                        predictions,
                        labels,
                        metadata,
                    )
                )

            except (
                ValueError,
                KeyError,
                TypeError,
            ) as exc:

                # Fairness should not prevent
                # classification evaluation.

                fairness_metrics = {
                    "fairness_evaluation_error":
                        str(exc)
                }

        # ----------------------------------------------------
        # Uncertainty
        # ----------------------------------------------------

        uncertainty_metrics = (
            uncertainty_summary(
                probabilities,
                labels,
            )
        )

        # ====================================================
        # Aggregate losses
        # ====================================================

        results = {

            "loss":
                total_loss / total_samples,

            "focal_loss":
                total_ce / total_samples,

            "metric_loss":
                total_metric / total_samples,

            **classification_metrics,

            **fairness_metrics,

            **uncertainty_metrics,
        }

        results = self.convert_keys(
            results
        )

        # ====================================================
        # Save artifacts
        # ====================================================

        if self.output_dir is not None:

            self._save_metrics(
                results,
                split,
            )

            self._save_predictions(
                labels=labels,
                predictions=predictions,
                probabilities=probabilities,
                metadata=metadata,
                split=split,
            )

            if (
                embeddings is not None
                and self.cfg.output.get(
                    "save_embeddings",
                    False,
                )
            ):

                self._save_embeddings(
                    embeddings,
                    labels,
                    predictions,
                    split,
                )

        # ====================================================
        # Return
        # ====================================================

        return {

            "metrics": results,

            "labels": labels,

            "predictions": predictions,

            "probabilities":
                probabilities,

            "metadata":
                metadata,

            "embeddings":
                embeddings,
        }

    # ========================================================
    # Save metrics
    # ========================================================

    def _save_metrics(
        self,
        results,
        split,
    ):
        """
        Save evaluation metrics as JSON.
        """

        metrics_path = (
            self.output_dir
            / "metrics"
            / f"{split}_metrics.json"
        )

        with metrics_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                results,
                file,
                indent=4,
                default=self._json_serializer,
            )

    # ========================================================
    # Save predictions
    # ========================================================

    def _save_predictions(
        self,
        labels,
        predictions,
        probabilities,
        metadata,
        split,
    ):
        """
        Save predictions and metadata as CSV.
        """

        if not self.cfg.output.get(
            "save_predictions",
            True,
        ):
            return

        predictions_path = (
            self.output_dir
            / "predictions"
            / f"{split}_predictions.csv"
        )

        metadata_keys = list(
            metadata.keys()
        )

        with predictions_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            header = [
                "sample_id",
                "true_label",
                "predicted_label",
                "confidence",
            ]

            header.extend(
                metadata_keys
            )

            header.append(
                "probabilities"
            )

            writer.writerow(
                header
            )

            # ------------------------------------------------
            # Rows
            # ------------------------------------------------

            for idx in range(
                len(labels)
            ):

                confidence = float(
                    probabilities[idx].max()
                )

                row = [

                    idx,

                    int(labels[idx]),

                    int(predictions[idx]),

                    confidence,
                ]

                # --------------------------------------------
                # Metadata
                # --------------------------------------------

                for key in metadata_keys:

                    value = metadata[key][idx]

                    if isinstance(
                        value,
                        np.generic,
                    ):

                        value = value.item()

                    row.append(
                        value
                    )

                # --------------------------------------------
                # Probability vector
                # --------------------------------------------

                row.append(
                    probabilities[idx].tolist()
                )

                writer.writerow(
                    row
                )

    # ========================================================
    # Save embeddings
    # ========================================================

    def _save_embeddings(
        self,
        embeddings,
        labels,
        predictions,
        split,
    ):
        """
        Save CARL embeddings.

        Stores:
            embeddings
            labels
            predictions
        """

        embedding_path = (
            self.output_dir
            / "embeddings"
            / f"{split}_embeddings.npz"
        )

        np.savez_compressed(
            embedding_path,
            embeddings=embeddings,
            labels=labels,
            predictions=predictions,
        )

    # ========================================================
    # Convenience method
    # ========================================================

    def evaluate_and_save(
        self,
        model,
        loader,
        loss_fn,
        device,
        split="val",
    ):
        """
        Convenience wrapper around evaluate().
        """

        return self.evaluate(
            model=model,
            loader=loader,
            loss_fn=loss_fn,
            device=device,
            split=split,
        )


# ============================================================
# Sanity Check
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CARL EVALUATOR SANITY CHECK")
    print("=" * 70)

    from omegaconf import OmegaConf

    try:
        from src.utils import get_device
    except ModuleNotFoundError:
        from utils import get_device

    cfg = OmegaConf.load(
        "/home/prasad/CARL/config.yaml"
    )

    print(
        f"Dataset : "
        f"{cfg.dataset.name}"
    )

    print(
        f"Backbone: "
        f"{cfg.backbone.name}"
    )

    print()

    # --------------------------------------------------------
    # Create evaluator
    # --------------------------------------------------------

    evaluator = Evaluator(
        cfg=cfg,
        output_dir=None,
    )

    assert evaluator.cfg is cfg

    print(
        "Evaluator initialization : OK"
    )

    # --------------------------------------------------------
    # Test JSON conversion
    # --------------------------------------------------------

    test_object = {

        "integer":
            np.int64(10),

        "float":
            np.float64(0.5),

        "array":
            np.array(
                [1, 2, 3]
            ),

        "nested": {

            np.int64(1):
                np.float64(2.0)
        },
    }

    converted = (
        evaluator.convert_keys(
            test_object
        )
    )

    assert isinstance(
        converted["integer"],
        int,
    )

    assert isinstance(
        converted["float"],
        float,
    )

    print(
        "JSON utilities            : OK"
    )

    # --------------------------------------------------------
    # Test metadata utilities
    # --------------------------------------------------------

    metadata = {

        "sex":
            torch.tensor(
                [0, 1, 0, 1]
            ),

        "location":
            torch.tensor(
                [1, 2, 1, 3]
            ),
    }

    device = get_device(
        cfg
    )

    moved = (
        evaluator._move_metadata_to_device(
            metadata,
            device,
        )
    )

    assert torch.is_tensor(
        moved["sex"]
    )

    assert (
        moved["sex"].device.type
        == device.type
    )

    print(
        "Metadata device handling  : OK"
    )

    # --------------------------------------------------------
    # Test metadata collection
    # --------------------------------------------------------

    storage = {}

    storage = (
        evaluator._collect_metadata(
            storage,
            metadata,
        )
    )

    finalized = (
        evaluator._finalize_metadata(
            storage
        )
    )

    assert (
        "sex"
        in finalized
    )

    assert (
        finalized["sex"].shape
        == (4,)
    )

    print(
        "Metadata collection        : OK"
    )

    # --------------------------------------------------------
    # Test model output extraction
    # --------------------------------------------------------

    dummy_logits = torch.randn(
        4,
        int(
            cfg.dataset.registry[
                cfg.dataset.name
            ].num_classes
        ),
    )

    dummy_embedding = torch.randn(
        4,
        int(
            cfg.carl.embedding_dim
        ),
    )

    dummy_outputs = {

        "logits":
            dummy_logits,

        "embedding":
            dummy_embedding,
    }

    logits = (
        evaluator._extract_logits(
            dummy_outputs
        )
    )

    embedding = (
        evaluator._extract_embedding(
            dummy_outputs
        )
    )

    assert (
        logits.shape[0] == 4
    )

    assert (
        embedding.shape
        == (
            4,
            int(
                cfg.carl.embedding_dim
            ),
        )
    )

    print(
        "Model output extraction   : OK"
    )

    # --------------------------------------------------------
    # Test probability generation
    # --------------------------------------------------------

    probabilities = F.softmax(
        dummy_logits,
        dim=1,
    )

    assert (
        probabilities.shape
        == dummy_logits.shape
    )

    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(4),
        atol=1e-5,
    )

    print(
        "Probability computation   : OK"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    print(
        f"Device                    : "
        f"{device}"
    )

    print()

    print(
        "CARL evaluator sanity "
        "check completed."
    )

    print("=" * 70)