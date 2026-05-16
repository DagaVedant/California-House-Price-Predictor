"""
server.py — Flask inference server

Run from the project root:
    python app/server.py

Requires a .env file in the project root with:
    GOOGLE_MAPS_API_KEY=your_key_here
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from src.utils import load_config, load_checkpoint, get_checkpoint_path, haversine_distance

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

app = Flask(__name__, static_folder=".")
CORS(app)

config = load_config("configs/config.yaml")
model  = load_checkpoint(get_checkpoint_path(config))
cities = config["cities"]

# Ocean proximity encoding matches LabelEncoder fit order from training:
# <1H OCEAN=0, INLAND=1, ISLAND=2, NEAR BAY=3, NEAR OCEAN=4
OCEAN_PROXIMITY_MAP = {
    "<1H OCEAN": 0,
    "INLAND":    1,
    "ISLAND":    2,
    "NEAR BAY":  3,
    "NEAR OCEAN": 4,
}

def classify_ocean_proximity(lat, lon):
    """Rough classification based on coordinates — good enough for inference."""
    dist_coast = min(
        haversine_distance(lat, lon, cities["san_francisco"]["lat"], cities["san_francisco"]["lon"]),
        haversine_distance(lat, lon, cities["los_angeles"]["lat"],   cities["los_angeles"]["lon"]),
        haversine_distance(lat, lon, cities["san_diego"]["lat"],     cities["san_diego"]["lon"]),
    )
    # SF Bay area
    if 37.2 < lat < 38.5 and -123.0 < lon < -121.5:
        return OCEAN_PROXIMITY_MAP["NEAR BAY"]
    if dist_coast < 15:
        return OCEAN_PROXIMITY_MAP["NEAR OCEAN"]
    if dist_coast < 80:
        return OCEAN_PROXIMITY_MAP["<1H OCEAN"]
    return OCEAN_PROXIMITY_MAP["INLAND"]


def build_feature_row(lat, lon, housing_median_age, total_rooms,
                      total_bedrooms, population, households, median_income):
    """Build the exact feature vector the model was trained on."""
    rooms_per_household      = total_rooms / max(households, 1)
    bedrooms_per_room        = total_bedrooms / max(total_rooms, 1)
    population_per_household = population / max(households, 1)

    dist_sf  = haversine_distance(lat, lon, cities["san_francisco"]["lat"], cities["san_francisco"]["lon"])
    dist_la  = haversine_distance(lat, lon, cities["los_angeles"]["lat"],   cities["los_angeles"]["lon"])
    dist_sd  = haversine_distance(lat, lon, cities["san_diego"]["lat"],     cities["san_diego"]["lon"])
    dist_sac = haversine_distance(lat, lon, cities["sacramento"]["lat"],    cities["sacramento"]["lon"])
    ocean    = classify_ocean_proximity(lat, lon)

    return {
        "longitude":               lon,
        "latitude":                lat,
        "housing_median_age":      housing_median_age,
        "total_rooms":             total_rooms,
        "total_bedrooms":          total_bedrooms,
        "population":              population,
        "households":              households,
        "median_income":           median_income,
        "rooms_per_household":     rooms_per_household,
        "bedrooms_per_room":       bedrooms_per_room,
        "population_per_household": population_per_household,
        "dist_to_sf":              dist_sf,
        "dist_to_la":              dist_la,
        "dist_to_san_diego":       dist_sd,
        "dist_to_sacramento":      dist_sac,
        "ocean_proximity":         ocean,
    }


def predict_price(row_dict):
    """Run inference and exponentiate from log space back to real dollars."""
    features = config["features"]["numeric"] + config["features"]["categorical"]
    df       = pd.DataFrame([row_dict])[features]
    log_pred = float(model.predict(df)[0])
    return np.expm1(log_pred)   # inverse of log1p used during training


@app.route("/")
def serve_index():
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    if not api_key:
        return "<h2>Error: GOOGLE_MAPS_API_KEY not set in .env</h2>", 500
    with open(os.path.join(os.path.dirname(__file__), "index.html"), "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("__GOOGLE_MAPS_API_KEY__", api_key)
    return html


@app.route("/predict/location", methods=["POST"])
def predict_location():
    data = request.json
    lat, lon = float(data["lat"]), float(data["lon"])

    # For heatmap clicks, use area-average defaults (representative of a typical block)
    row   = build_feature_row(
        lat=lat, lon=lon,
        housing_median_age=25,
        total_rooms=2500,
        total_bedrooms=500,
        population=1200,
        households=450,
        median_income=4.5,
    )
    price = predict_price(row)
    return jsonify({"predicted_price": round(max(0, price), 2)})


@app.route("/predict/address", methods=["POST"])
def predict_address():
    data = request.json
    lat  = float(data["lat"])
    lon  = float(data["lon"])

    bedrooms   = int(data.get("bedrooms", 3))
    sqft       = float(data.get("sqft", 1500))
    house_age  = float(data.get("house_age", 20))
    income     = float(data.get("median_income", 4.5))

    # Derive realistic neighbourhood-scale features from house inputs
    # avg room size ~200 sqft; total_rooms includes all rooms not just bedrooms
    total_rooms    = sqft / 200 * 5          # estimate ~5 rooms per house
    total_bedrooms = bedrooms
    # Typical California block: ~450 households, 2.7 persons/household
    households     = 450
    population     = int(households * 2.7)

    row   = build_feature_row(
        lat=lat, lon=lon,
        housing_median_age=house_age,
        total_rooms=total_rooms,
        total_bedrooms=total_bedrooms,
        population=population,
        households=households,
        median_income=income,
    )
    price = max(0, predict_price(row))

    return jsonify({
        "predicted_price":  round(price, 2),
        "confidence_low":   round(price * 0.82, 2),
        "confidence_high":  round(price * 1.18, 2),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(
        host=config["server"]["host"],
        port=config["server"]["port"],
        debug=config["server"]["debug"],
    )
