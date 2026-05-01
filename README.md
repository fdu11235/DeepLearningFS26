# DeepLearningFS26 — LSTM Fraud Detection

Credit card fraud detection using an LSTM trained on per-card transaction sequences. Compares against the classic SVM/RF (with and without SMOTE) baselines from the original notebook.

## Layout

```
src/fraud/         # Python package (data, preprocessing, models/lstm, evaluation, utils)
scripts/           # CLI entrypoints (train_lstm.py, evaluate.py)
configs/           # YAML configs for each LSTM run (lstm.yaml, lstm_smote.yaml)
notebooks/         # 01_eda.ipynb, 02_lstm_results.ipynb (thin, import from src)
notebooks/legacy/  # original fraud_detection_notebook.ipynb (classic-ML reference)
data/raw/          # fraudTrain.csv, fraudTest.csv (gitignored)
data/processed/    # cached parquet, sequence tensors, legacy_predictions.parquet (gitignored)
models/            # trained model artifacts per run (gitignored)
tests/             # pytest unit tests
```

## Install

### pip (recommended)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For CUDA support, install PyTorch with the matching CUDA wheels:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### conda

```bash
conda env create -f environment.yml
conda activate fraud-fs26
pip install -e .
```

## Data

Place `fraudTrain.csv` and `fraudTest.csv` (Kaggle "Credit Card Transactions Fraud Detection Dataset") under `data/raw/`. They are gitignored.

## Train

```bash
# Baseline LSTM (BCE with pos_weight, no oversampling)
python scripts/train_lstm.py --config configs/lstm.yaml

# LSTM + sequence-level SMOTE
python scripts/train_lstm.py --config configs/lstm_smote.yaml
```

Artifacts are written to `models/lstm/` and `models/lstm_smote/` respectively (`model.pt`, `preprocessor.pkl`, `threshold.json`, `metrics.json`, `training_log.csv`).

## Evaluate and compare against classic baselines

```bash
python scripts/evaluate.py \
    --lstm models/lstm \
    --lstm-smote models/lstm_smote \
    --legacy-predictions data/processed/legacy_predictions.parquet
```

This prints a side-by-side metrics table and McNemar p-values. The four classic baselines (SVM, RF, SVM+SMOTE, RF+SMOTE) come from `notebooks/legacy/fraud_detection_notebook.ipynb` — re-run its final cell to (re)create `legacy_predictions.parquet`.

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
