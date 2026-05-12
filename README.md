# DeepLearningFS26 — LSTM Fraud Detection

Credit-card fraud detection using an LSTM trained on per-card transaction sequences. Three imbalance-handling variants are compared against the team's prior classical-ML baseline (Random Forest / SVM).

## Layout

```
src/fraud/         # Python package (data, preprocessing, models/lstm, evaluation, utils)
scripts/           # CLI entrypoints (train_lstm.py, evaluate.py, tune_lstm.py, generate_plots.py)
configs/           # YAML configs
notebooks/         # 01_eda.ipynb, 02_lstm_results.ipynb (thin, import from src)
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

## Smoke test (~30 seconds)

Verify the pipeline end-to-end on a 2% subset for 1 epoch before launching full training:

```bash
python scripts/train_lstm.py --config configs/lstm_smoke.yaml
```

Discard the resulting model — its metrics aren't meaningful, this only checks that loading, preprocessing, sequence building, and training all work on your machine.

## Train

```bash
# Plain BCE (paper's baseline — no explicit imbalance handling)
python scripts/train_lstm.py --config configs/lstm_no_imbalance.yaml

# Cost-sensitive BCE (pos_weight ≈ 172)
python scripts/train_lstm.py --config configs/lstm.yaml

# Sequence-level SMOTE
python scripts/train_lstm.py --config configs/lstm_smote.yaml
```

Artifacts go to `models/<config-name>/`: `model.pt`, `preprocessor.pkl`, `threshold.json`, `metrics.json`, `training_log.csv`, `test_predictions.npz`, training-curve and confusion-matrix PNGs.

## Evaluate

```bash
python scripts/evaluate.py
```

Prints a side-by-side metrics table for the LSTM runs against the classic baselines if `data/processed/legacy_predictions.parquet` exists. To produce that parquet, open `notebooks/02_lstm_results.ipynb` (or the legacy notebook) and run its export cell.

## Tests

```bash
pytest tests/
```
