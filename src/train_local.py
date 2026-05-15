"""
train_local.py — Training script for local machine (GTX 1650)

Before running:
  1. In configs/config.yaml, set tree_method: gpu_hist  (uses your GTX 1650)
     OR leave as hist to train on CPU instead
  2. Make sure housing.csv is in data/raw/housing.csv
  3. Run from the project root:
       python src/train_local.py

Output:
  - checkpoints/xgb_california.pkl   (saved model)
  - checkpoints/feature_importance.png
  - checkpoints/pred_vs_actual.png
  - WandB run logged to your project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import numpy as np

from src.utils import load_config, save_checkpoint, get_checkpoint_path
from src.dataset import prepare_dataset
from src.model import HousePriceModel
from src.evaluate import compute_metrics, print_metrics, log_to_wandb, plot_feature_importance, plot_predictions


def train():
    config = load_config("configs/config.yaml")

    # Init WandB — logs hyperparams, metrics, and charts
    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        config=config["model"]["params"]
    )

    print("=" * 50)
    print("  California House Price Predictor — Local")
    print("=" * 50)

    print("\n[1/4] Preparing dataset...")
    X_train, X_val, y_train, y_val = prepare_dataset(config)

    print("\n[2/4] Training XGBoost model...")
    tree_method = config["model"]["params"].get("tree_method", "hist")
    print(f"      tree_method = {tree_method}")
    model = HousePriceModel(config)
    model.fit(X_train, y_train, X_val, y_val)
    print(f"      Best iteration: {model.best_iteration()}")

    print("\n[3/4] Evaluating...")
    y_pred = model.predict(X_val)
    metrics = compute_metrics(y_val.values, y_pred)
    print_metrics(metrics)

    feature_names = config["features"]["numeric"] + config["features"]["categorical"]
    importance = model.get_feature_importance(feature_names)

    # Save charts locally + log to WandB
    plot_feature_importance(importance)
    plot_predictions(y_val.values, y_pred)
    log_to_wandb(metrics, importance, y_val.values, y_pred, config)

    print("\n[4/4] Saving checkpoint...")
    save_checkpoint(model, get_checkpoint_path(config))

    wandb.finish()
    print("\nDone! Transfer checkpoints/xgb_california.pkl to your i7 machine.")
    print("Upload to Google Drive or copy via USB/network.")


if __name__ == "__main__":
    train()
