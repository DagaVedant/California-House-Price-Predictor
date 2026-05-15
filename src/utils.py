import yaml
import pickle
import os
import numpy as np
from pathlib import Path


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def load_checkpoint(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {path}")
    return model


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two lat/lon points."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def get_checkpoint_path(config):
    return os.path.join(config["paths"]["checkpoint_dir"], config["paths"]["checkpoint_name"])
