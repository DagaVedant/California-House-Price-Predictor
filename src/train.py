"""
train.py — Local training script

Before running:
  1. Make sure housing.csv is in data/raw/
  2. Run: python src/train.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.utils import load_config, save_checkpoint, get_checkpoint_path
from src.dataset import prepare_dataset
from src.model import HousePriceModel
from src.evaluate import compute_metrics, print_metrics, plot_feature_importance, plot_predictions


def train():
    config = load_config("configs/config.yaml")

    print("Preparing dataset...")
    X_train, X_val, y_train, y_val = prepare_dataset(config)

    print("Training XGBoost model...")
    model = HousePriceModel(config)
    model.fit(X_train, y_train, X_val, y_val)
    print(f"Best iteration: {model.best_iteration()}")

    y_pred = model.predict(X_val)
    metrics = compute_metrics(y_val.values, y_pred)
    print_metrics(metrics)

    feature_names = config["features"]["numeric"] + config["features"]["categorical"]
    importance = model.get_feature_importance(feature_names)

    os.makedirs(config["paths"]["checkpoint_dir"], exist_ok=True)
    plot_feature_importance(importance)
    plot_predictions(y_val.values, y_pred)

    checkpoint_path = get_checkpoint_path(config)
    save_checkpoint(model, checkpoint_path)
    print("Training complete.")


if __name__ == "__main__":
    train()
