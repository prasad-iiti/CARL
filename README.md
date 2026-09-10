# ADF-Skin

Official PyTorch implementation of **ADF-Skin**, a fairness-aware deep learning framework for skin lesion classification using adversarial debiasing and feature-guided contrastive learning.

---

## Repository Structure

```
ADF-Skin/
├── configs/
│   └── config.yaml
│
├── datasets/
│   ├── derm7pt/
│   ├── hiba/
│   └── padufes20/
│
├── outputs/
│   ├── experiments/
│   └── results.csv
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── ablation.py
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
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

Clone the repository.

```bash
git clone https://github.com/<username>/ADF-Skin.git
cd ADF-Skin
```

Create a virtual environment (recommended).

```bash
python -m venv .venv
source .venv/bin/activate        # Linux
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

Place the datasets inside the `datasets/` directory.

```
datasets/
├── derm7pt/
├── hiba/
└── padufes20/
```

Configure dataset paths and training parameters in

```
configs/config.yaml
```

---

## Training

```bash
python scripts/train.py
```

---

## Evaluation

Update the checkpoint path inside `configs/config.yaml`.

```yaml
evaluation:
  checkpoint: outputs/experiments/.../checkpoints/best_model.pth
```

Then run

```bash
python scripts/evaluate.py
```

---

## Ablation Studies

```bash
python scripts/ablation.py
```

---

## Experiment Outputs

Each execution automatically creates a new experiment directory inside

```
outputs/experiments/
```

with the following structure:

```
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
│   ├── test_metrics.json
│   ├── fairness_metrics.json        (optional)
│   └── uncertainty_metrics.json     (optional)
│
├── plots/
│   ├── training/
│   │   ├── loss_curve_train_val.pdf
│   │   └── acc_curve_train_val.pdf
│   │
│   ├── evaluation/
│   │   ├── roc_curve_micro_macro.pdf
│   │   ├── pr_curve_micro_macro.pdf
│   │   └── confusion_matrix_normalized.pdf
│   │
│   └── fairness/
│       ├── eo_gap_vs_epoch.pdf
│       └── subgroup_accuracy.pdf
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
│   ├── feature_stats.json
│   └── config_diff.yaml            (optional)
│
└── meta.json
```

---

## Configuration

All experiment settings are controlled through

```
configs/config.yaml
```

including

- Dataset selection
- Model architecture
- Loss function weights
- Optimizer
- Learning rate scheduler
- Training hyperparameters
- Evaluation checkpoint

---

## Requirements

The required Python packages are listed in

```
requirements.txt
```

Install them using

```bash
pip install -r requirements.txt
```

---

## Citation

If you use this repository in your research, please cite the corresponding publication.

```
Citation will be added after publication.
```

---

## License

This repository is released for academic and research purposes.