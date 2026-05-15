"""
train.py — Local training script (GTX 1650)

Before running:
  1. Set tree_method: gpu_hist in configs/config.yaml to use your GPU
  2. Make sure housing.csv is in data/raw/
  3. Run: python train.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import numpy as np
from src.utils import load_config, save_checkpoint, get_checkpoint_path
from src.dataset import prepare_dataset
from src.model import HousePriceModel
from src.evaluate import compute_metrics, print_metrics, log_to_wandb


def train():
    config = load_config("configs/config.yaml")

    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        config=config["model"]["params"]
    )

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

    log_to_wandb(metrics, importance, y_val.values, y_pred, config)

    checkpoint_path = get_checkpoint_path(config)
    save_checkpoint(model, checkpoint_path)

    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    train()
