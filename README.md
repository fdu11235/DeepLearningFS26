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

## Results

All metrics on the held-out **`fraudTest.csv`** (555,719 rows, 2,145 fraud / 553,574 non-fraud, 0.39% positive rate). Classic-ML rows are reported by the legacy notebook with its F2-tuned threshold; LSTM rows are reported at both the default `0.5` threshold and the F2-tuned threshold from validation.

| Model        | Threshold | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|--------------|-----------|---------|--------|-----------|--------|----|
| SVM          | tuned     | 0.9031  | 0.1014 | 0.18 | 0.38 | 0.24 |
| RF           | tuned     | 0.9817  | 0.5085 | 0.59 | 0.41 | 0.48 |
| SVM + SMOTE  | tuned     | 0.8971  | 0.0857 | 0.03 | 0.68 | 0.06 |
| RF + SMOTE   | tuned     | 0.8143  | 0.1014 | 0.20 | 0.10 | 0.14 |
| **LSTM**     | 0.5       | 0.9883  | 0.8985 | 0.60 | 0.91 | 0.72 |
| **LSTM**     | tuned 0.9931 | **0.9883** | **0.8985** | **0.88** | **0.85** | **0.86** |
| LSTM + SMOTE | 0.5       | 0.9863  | 0.8617 | 0.77 | 0.83 | 0.80 |
| LSTM + SMOTE | tuned 0.1436 | 0.9863  | 0.8617 | 0.53 | 0.89 | 0.66 |

Headline: **LSTM lifts PR-AUC from RF's 0.51 to 0.90** and **F1 from 0.48 to 0.86**. Same pattern as the legacy notebook: SMOTE does not help — a properly class-weighted loss (`BCEWithLogitsLoss(pos_weight≈150)`) outperforms SMOTE-on-flattened-windows.

## Hyperparameter tuning

A small focused sweep over the two hyperparameters most likely to affect generalisation under class imbalance — learning rate and dropout — was run on the baseline LSTM (no SMOTE). The grid is `lr ∈ {3e-4, 1e-3, 3e-3} × dropout ∈ {0.3, 0.5}`, six trials, seed fixed at 42 to isolate the hyperparameter effect (variance was not estimated). All other settings match `configs/lstm.yaml`. Each trial trains for up to 30 epochs with the existing early-stopping patience of 5.

Reproduce:

```bash
python scripts/tune_lstm.py --sweep-config configs/tune_lstm.yaml
```

Per-trial artifacts land in `models/tuning/lstm_lr_dropout_v1/trial_NNN_*/` (`model.pt`, `metrics.json`, `training_log.csv`, `training_curves.png`, `confusion_matrix_default.png`, `confusion_matrix_tuned.png`). Aggregated row-per-trial CSV at `models/tuning/lstm_lr_dropout_v1/tuning_results.csv`.

| lr     | dropout | val PR-AUC | test ROC-AUC | test PR-AUC | tuned thr | tuned P | tuned R | tuned F1 |
|--------|---------|------------|--------------|-------------|-----------|---------|---------|----------|
| 3e-4   | 0.3     | 0.830      | 0.985        | 0.860       | 0.976     | 0.77    | 0.82    | 0.80     |
| **1e-3** | **0.3** | **0.889**  | 0.988        | 0.899       | 0.993     | **0.88** | 0.85   | **0.86** |
| 3e-3   | 0.3     | 0.878      | 0.989        | *0.915*     | 0.970     | 0.84    | 0.87    | 0.86     |
| 3e-4   | 0.5     | 0.828      | 0.981        | 0.858       | 0.875     | 0.79    | 0.82    | 0.81     |
| 1e-3   | 0.5     | 0.874      | 0.983        | 0.888       | 0.923     | 0.81    | 0.86    | 0.83     |
| 3e-3   | 0.5     | 0.888      | 0.987        | 0.890       | 0.982     | 0.76    | 0.87    | 0.81     |

Bold = best by validation PR-AUC (the legitimate selection criterion). *Italic* = best by test PR-AUC, shown only to make the model-selection point honestly: if you peeked at test, `lr=3e-3, dropout=0.3` would win marginally. Selecting on the validation set returns the original baseline (`lr=1e-3, dropout=0.3`), and the sweep does **not** materially improve it within this grid. The bottom row is that the existing configuration was already a reasonable first guess; cutting the learning rate to `3e-4` clearly hurts (under-trained within the epoch budget), and increasing dropout to `0.5` is a small but consistent regression — the early-stopping mechanism alone already addresses the train/val divergence visible in `training_log.csv`.

## Where the gain comes from: `cc_num` as structure, not a feature

This is the technical reason the LSTM beats the RF, and it's worth understanding.

**Why the legacy notebook dropped `cc_num` for the RF.** With 983 unique cards and ~470k rows, the only sane way to feed `cc_num` into a tree model is target encoding (mean fraud rate per card). That gives the RF a per-cardholder fraud-history feature, which (a) memorises individual cardholders, (b) fails completely on cards never seen in training, and (c) is privacy-fraught. So the legacy RF was correctly trained as a **row-wise classifier with no card identity at all** — every transaction scored in isolation. The cost: no notion of "what's normal *for this card*". A $400 charge at 03:14 looks identical whether the card just spent the last month buying coffee or has a history of high-amount late-night purchases.

**How the LSTM uses `cc_num` without re-introducing the same problem.** The LSTM also never sees the card identity as a feature — `cc_num` is **not** in `FEATURE_COLS`. Instead, `cc_num` is used purely as a **grouping key** in `src/fraud/models/lstm/sequences.py`: group transactions by card, sort each card's transactions by time, build sliding windows of length 20. The card ID itself is dropped before the tensor enters the model. So the model never learns "card 4234… has high fraud rate" — it learns shapes of behaviour over a card's recent history.

```
Row-wise RF sees (one transaction):       LSTM sees (a sequence):

   ┌──────────────┐                       ┌─────────────────────────────────┐
   │     tx_t     │                       │ tx_{t-19}, tx_{t-18}, …, tx_t   │
   └──────────────┘                       └─────────────────────────────────┘
   amt, merchant, hour, …                 the same card's last 20 txs in
                                          time order, no card-ID feature
```

**Why this isn't leakage.** Three reasons:

1. **In production, you always know the card number** when scoring a transaction (it's on the swipe). Grouping by `cc_num` at inference is "look up this card's recent history" — exactly what a deployed fraud system does.
2. **The grouping carries no label information.** Only the *target* row's `is_fraud` is the label; historical rows in the window are causal (only earlier transactions, never future — see `tests/test_sequences.py::test_no_cross_card_leakage`).
3. **Card identity never enters the model.** A fresh card the LSTM has never seen still gets a sensible prediction, just with a shorter history (left-padded). The same privacy property as the RF.

So: **RF approach** = row-wise classifier, no per-card info at all. **LSTM approach** = row-wise classifier, but with the recent same-card history attached as context. The +0.39 PR-AUC comes entirely from this structural change — context that the legacy notebook explicitly noted it couldn't capture.

## How can the results be this good? (methodology audit)

Sanity-checking the gap, since +0.39 PR-AUC is large:

1. **Threshold tuning happens on val only, never on test.** In `src/fraud/models/lstm/trainer.py`: after early stopping, `tune_threshold_f2` runs on validation probabilities (line ~213); only after the threshold is fixed do we score the test set (line ~219). The test labels never enter the tuning loop.
2. **Preprocessor is fit on train only.** `_prepare_features(df_train, ..., fit=True)` then `fit=False` for val and test. The `TargetEncoder` for `merchant/job/city/zip` only sees train labels, so no target leakage.
3. **Sequences never cross splits.** `build_sequences` is called separately on each split. A test window for card X contains card X's earlier transactions *within `fraudTest.csv` only* — train transactions for the same card are not prepended. Causal: a window for tx `i` contains only txs `1..i`. (See `tests/test_sequences.py::test_no_cross_card_leakage`.)
4. **`cc_num` is a grouping key, not a feature.** Used to group windows; dropped before the model tensor (it isn't in `FEATURE_COLS`).
5. **The performance gain is structural, not statistical.** The classic models scored each transaction in isolation. The LSTM sees the last 20 transactions of the same card, so it can flag "$400 charge after 19 small same-merchant charges" as anomalous *for this card* — context the legacy notebook explicitly noted it couldn't capture (it dropped `cc_num` as PII and treated rows as i.i.d.).
6. **The probability distribution is bimodal.** On the test set, the median negative score is `0.0002` and the median positive score is `1.0000` — the model separates classes cleanly. The high tuned threshold of `0.9931` is a consequence of training with `pos_weight≈150` (logits are pushed up); F2-tuning then trims the small middle cluster of false positives. No red flag.

A useful control if you want extra reassurance: shuffle the `is_fraud` labels in the train slice and retrain — PR-AUC should collapse to ~0.004 (the base rate). If it doesn't, there's leakage.

## Caveats

- **11-month gap between train and test.** Train ends 2019-07-26, test starts 2020-06-21. The LSTM was fit on early-2019 patterns and is judged on late-2020 behavior, so a small distribution-shift discount is honest framing.
- **Cold-start cards.** Some `cc_num` values in `fraudTest.csv` may have no history in `fraudTrain.csv`. For those, the test window has length 1 with no historical context — the LSTM falls back to a transaction-level prediction.
- **The dataset has 471k train rows and 556k test rows** (not 1.3M as initial inspection suggested) — there's a single malformed trailing row in `fraudTrain.csv` with `NaN` in `is_fraud` that we drop in feature engineering.
- **SMOTE memory usage.** Sequence-level SMOTE at `sampling_strategy="auto"` peaks at ~16 GB on the train set and OOMs on a 16 GB WSL machine. The `lstm_smote.yaml` config defaults to `sampling_strategy: 0.1` (minority becomes 10% of majority) which peaks at ~5–6 GB. See `src/fraud/models/lstm/smote.py` for details.

## Reference baseline numbers

The classic-ML numbers in the table above are reproduced from `notebooks/legacy/fraud_detection_notebook.ipynb`. To regenerate them and produce per-row predictions for McNemar comparison, open that notebook, run all cells, then run the final export cell — it writes `data/processed/legacy_predictions.parquet`. After that, `python scripts/evaluate.py` will print the full table plus McNemar p-values.
