"""CLI entrypoint: python scripts/train_lstm.py --config configs/lstm.yaml"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Allow running without `pip install -e .` by inserting src/ on the path.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fraud.models.lstm.trainer import TrainConfig, train_lstm  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train FraudLSTM")
    parser.add_argument("--config", required=True, help="path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = TrainConfig(**raw)
    train_lstm(cfg)


if __name__ == "__main__":
    main()
