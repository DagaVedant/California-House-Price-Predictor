import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.utils import load_config, haversine_distance


def load_raw_data(config):
    df = pd.read_csv(config["paths"]["raw_data"])
    print(f"Loaded {len(df)} rows from {config['paths']['raw_data']}")
    return df


def clean_data(df, config):
    strategy = config["preprocessing"]["fill_missing_strategy"]
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            fill_val = df[col].median() if strategy == "median" else df[col].mean()
            df[col] = df[col].fillna(fill_val)
    return df


def engineer_features(df, config):
    cities = config["cities"]

    df["rooms_per_household"]      = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"]        = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]

    df["dist_to_sf"] = haversine_distance(
        df["latitude"], df["longitude"],
        cities["san_francisco"]["lat"], cities["san_francisco"]["lon"]
    )
    df["dist_to_la"] = haversine_distance(
        df["latitude"], df["longitude"],
        cities["los_angeles"]["lat"], cities["los_angeles"]["lon"]
    )
    df["dist_to_san_diego"] = haversine_distance(
        df["latitude"], df["longitude"],
        cities["san_diego"]["lat"], cities["san_diego"]["lon"]
    )
    df["dist_to_sacramento"] = haversine_distance(
        df["latitude"], df["longitude"],
        cities["sacramento"]["lat"], cities["sacramento"]["lon"]
    )

    le = LabelEncoder()
    df["ocean_proximity"] = le.fit_transform(df["ocean_proximity"].astype(str))

    return df


def get_feature_matrix(df, config):
    features = config["features"]["numeric"] + config["features"]["categorical"]
    target   = config["features"]["target"]
    return df[features], df[target]


def split_data(X, y, config):
    return train_test_split(
        X, y,
        test_size=config["preprocessing"]["test_size"],
        random_state=config["preprocessing"]["random_state"]
    )


def prepare_dataset(config):
    df = load_raw_data(config)
    df = clean_data(df, config)
    df = engineer_features(df, config)

    os.makedirs(os.path.dirname(config["paths"]["processed_data"]), exist_ok=True)
    df.to_csv(config["paths"]["processed_data"], index=False)
    print(f"Processed data saved to {config['paths']['processed_data']}")

    X, y = get_feature_matrix(df, config)

    # FIX: Log-transform the target so the model can generalise beyond the
    # $500,001 Kaggle cap and predict luxury prices accurately.
    # We store log(price) during training and exponentiate at inference.
    y_log = np.log1p(y)

    X_train, X_val, y_train, y_val = split_data(X, y_log, config)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    return X_train, X_val, y_train, y_val
