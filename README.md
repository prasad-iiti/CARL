# CARL

Official PyTorch implementation of **CARL (Contrastive Adaptive Representation Learning)** for robust fine-grained skin lesion classification.

CARL combines **Context-Aware Adaptive Semantic Scale Selection (CA-AS3)**, **Context-Aware Fine-Grained Attention (CAFA)**, and **contrastive representation learning with feedback refinement** to improve fine-grained lesion discrimination and representation robustness. Contrastive feedback refinement is applied **only during training** and introduces no additional inference-time computational overhead.

---

## Repository Structure

```text
CARL/
├── configs/
│   └── config.yaml
│
├── datasets/
│   ├── ham10000/
│   └── padufes20/
│
├── outputs/
│   ├── experiments/
│   └── results.csv 
│
├── src/
│   ├── data.py
│   ├── evaluator.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── trainer.py
│   └── utils.py
│
├── train.py
├── evaluate.py
├── ablation.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<username>/CARL.git
cd CARL
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

### Linux

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

Place the datasets in the corresponding directories:

```text
datasets/
├── ham10000/
└── padufes20/
```

Configure dataset paths and experimental settings in:

```text
configs/config.yaml
```

The datasets are **not distributed with this repository**. Users should obtain the datasets from their respective official sources and comply with their applicable terms of use.

---

## Supported Backbones

CARL currently supports the following backbone architectures:

* **EfficientNet-B0**
* **ResNet-50**

The backbone can be selected through the configuration file.

---

## Model Components

CARL consists of the following principal components:

### 1. Context-Aware Adaptive Semantic Scale Selection (CA-AS3)

CA-AS3 adaptively aggregates multi-level feature representations to capture discriminative information at different semantic scales.

The module operates on selected intermediate feature levels and uses:

* Average pooling
* Max pooling
* Variance pooling
* Learnable semantic-scale attention

### 2. Context-Aware Fine-Grained Attention (CAFA)

CAFA refines the aggregated representation using channel and spatial attention to emphasize diagnostically relevant feature responses.

### 3. Contrastive Representation Learning

CARL jointly optimizes classification and supervised contrastive objectives to improve the separability of fine-grained lesion representations.

### 4. Contrastive Feedback Refinement

The learned representation is refined using contrastive feedback during training. This refinement is used only during optimization and does not introduce additional inference-time processing.

---

## Training

Configure the desired dataset, backbone, optimization parameters, and experiment settings in:

```text
configs/config.yaml
```

Run training using:

```bash
python scripts/train.py
```

Each execution automatically creates a separate experiment directory under:

```text
outputs/experiments/
```

---

## Evaluation

Specify the trained checkpoint in:

```yaml
evaluation:
  checkpoint: outputs/experiments/.../checkpoints/best_model.pth
```

Then run:

```bash
python scripts/evaluate.py
```

The evaluation script reports the configured classification metrics and generates the corresponding evaluation artifacts.

---

## Ablation Studies

Run the ablation experiments using:

```bash
python scripts/ablation.py
```

The ablation configuration is controlled through the experiment settings defined in:

```text
configs/config.yaml
```

---

## Experiment Outputs

Each experiment creates a separate directory containing the configuration, checkpoints, logs, metrics, plots, and predictions.

```text
outputs/experiments/run_folder/
│
├── config.yaml
│
├── logs/
│   └── train.log
│
├── checkpoints/
│   ├── best_model.pth
│   ├── best_model_meta.json
│   ├── last_model.pth
│   └── last_model_meta.json
│
├── metrics/
│   ├── val_metrics.jsonl
│   └── test_metrics.json
│
├── plots/
│   ├── training/
│   │   ├── loss_curve_train_val.pdf
│   │   └── acc_curve_train_val.pdf
│   │
│   └── evaluation/
│       ├── roc_curve_micro_macro.pdf
│       ├── pr_curve_micro_macro.pdf
│       └── confusion_matrix_normalized.pdf
│
├── predictions/
│   ├── test_predictions.csv
│   ├── high_confidence_errors.csv
│   └── misclassified/
│       ├── sample_001.png
│       └── sample_045.png
│
├── artifacts/
│   ├── model_summary.txt
│   └── feature_stats.json
│
└── meta.json
```

---

## Configuration

All major experiment settings are controlled through:

```text
configs/config.yaml
```

These include:

* Dataset and dataset paths
* Number of classes
* Backbone architecture
* CA-AS3 configuration
* CAFA configuration
* Projection-head configuration
* Contrastive-learning parameters
* Loss-function weights
* Optimizer
* Learning-rate scheduler
* Batch size
* Number of training epochs
* Checkpointing
* Evaluation settings

---

## Reproducibility

The configuration used for each experiment is automatically saved inside its corresponding experiment directory:

```text
outputs/experiments/run_folder/config.yaml
```

This preserves the experimental configuration associated with each trained model and result.

---

## Hardware and Software

The experiments were conducted on a Dell Precision 7920 server equipped with:

```text
CPU: Intel Xeon Gold 6348 (112 cores)
RAM: 128 GB
OS: Ubuntu 24.04
```

GPU and CUDA requirements depend on the selected training configuration.

---

## Requirements

All required Python packages are listed in:

```text
requirements.txt
```

Install them with:

```bash
pip install -r requirements.txt
```

---

## Citation

If you use CARL or its implementation in your research, please cite the corresponding publication:

```text
Citation will be added after publication.
```

---

## License

This repository is released for **academic and research purposes**.
