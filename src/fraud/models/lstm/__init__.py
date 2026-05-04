"""LSTM submodule. Import the specific module you need:

    from fraud.models.lstm.model import FraudLSTM
    from fraud.models.lstm.sequences import build_sequences
    from fraud.models.lstm.dataset import SequenceDataset
    from fraud.models.lstm.smote import sequence_smote   # requires imblearn
    from fraud.models.lstm.trainer import train_lstm     # requires sklearn + imblearn

Submodules are not eagerly imported here so the package can be partially
imported even when optional dependencies (e.g. imbalanced-learn) are absent.
"""
