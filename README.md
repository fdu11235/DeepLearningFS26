# DeepLearningFS26 — LSTM Fraud Detection

Credit card fraud detection using an LSTM trained on per-card transaction sequences. Compares against the classic SVM/RF (with and without SMOTE) baselines from the original notebook.

## Layout

```
src/fraud/         # Python package (data, preprocessing, models/lstm, evaluation, utils)
scripts/           # CLI entrypoints (train_lstm.py, evaluate.py)
configs/           # YAML configs (lstm.yaml, lstm_smote.yaml, lstm_smoke.yaml)
notebooks/         # 01_eda.ipynb, 02_lstm_results.ipynb (thin, import from src)
notebooks/legacy/  # original fraud_detection_notebook.ipynb (classic-ML reference)
data/raw/          # fraudTrain.csv, fraudTest.csv (gitignored, see "Data")
data/processed/    # legacy_predictions.parquet, sequence caches (gitignored)
models/            # trained model artifacts per run (gitignored)
tests/             # pytest unit tests
```

## Install

### pip (recommended)

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# PyTorch with CUDA (skip if you only need CPU; default torch from PyPI is CPU)
pip install --index-url https://download.pytorch.org/whl/cu124 torch

pip install -r requirements.txt
pip install -e .
```

### conda

```bash
conda env create -f environment.yml
conda activate fraud-fs26
pip install -e .
```

Verify CUDA is picked up:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

## Data

Place `fraudTrain.csv` and `fraudTest.csv` (Kaggle "Credit Card Transactions Fraud Detection Dataset") under `data/raw/`. They are gitignored.

## Quick smoke test (~30 seconds)

Verify the pipeline end-to-end on a 2% subset for 1 epoch before launching full training:

```bash
python scripts/train_lstm.py --config configs/lstm_smoke.yaml
```

Discard the resulting model — its metrics aren't meaningful, this only checks that loading, preprocessing, sequence building, and training all work on your machine.

## Train

```bash
# Baseline LSTM (BCE with pos_weight, no oversampling)
python scripts/train_lstm.py --config configs/lstm.yaml

# LSTM + sequence-level SMOTE
python scripts/train_lstm.py --config configs/lstm_smote.yaml
```

Artifacts go to `models/lstm/` and `models/lstm_smote/`: `model.pt`, `preprocessor.pkl`, `threshold.json`, `metrics.json`, `training_log.csv`, `test_predictions.npz`. Training on a single GPU (RTX 4070 Ti class) takes well under an hour.

## Evaluate and compare against classic baselines

```bash
python scripts/evaluate.py
```

Prints a side-by-side metrics table for both LSTM configs and runs McNemar tests against the classic baselines if `data/processed/legacy_predictions.parquet` exists.

To produce that parquet: open `notebooks/legacy/fraud_detection_notebook.ipynb`, run the whole notebook, and run the final export cell. (The notebook's CSV paths were updated to read from `../../data/raw/` for its new location.)

## Tests

```bash
pytest tests/
```

## Reference baseline (classic ML, from legacy notebook)

| Model       | ROC-AUC | PR-AUC | P / R / F1 (fraud) |
|-------------|---------|--------|--------------------|
| SVM         | 0.9031  | 0.1014 | 0.18 / 0.38 / 0.24 |
| **RF**      | **0.9817** | **0.5085** | **0.59 / 0.41 / 0.48** |
| SVM + SMOTE | 0.8971  | 0.0857 | 0.03 / 0.68 / 0.06 |
| RF + SMOTE  | 0.8143  | 0.1014 | 0.20 / 0.10 / 0.14 |

The LSTM aims to beat the strongest baseline (RF, PR-AUC 0.5085) by exploiting per-card transaction sequences.
