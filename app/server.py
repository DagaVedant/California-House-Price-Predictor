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
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from src.utils import load_config, load_checkpoint, get_checkpoint_path, haversine_distance

# Load .env from project root (one level up from app/)
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

app = Flask(__name__, static_folder=".")
CORS(app)

config = load_config("configs/config.yaml")
model = load_checkpoint(get_checkpoint_path(config))
cities = config["cities"]


def build_feature_row(lat, lon, housing_median_age=20, total_rooms=1500,
                      total_bedrooms=300, population=800, households=300,
                      median_income=4.0, ocean_proximity=0):
    rooms_per_household = total_rooms / households
    bedrooms_per_room = total_bedrooms / total_rooms
    population_per_household = population / households

    dist_sf  = haversine_distance(lat, lon, cities["san_francisco"]["lat"], cities["san_francisco"]["lon"])
    dist_la  = haversine_distance(lat, lon, cities["los_angeles"]["lat"],   cities["los_angeles"]["lon"])
    dist_sd  = haversine_distance(lat, lon, cities["san_diego"]["lat"],     cities["san_diego"]["lon"])
    dist_sac = haversine_distance(lat, lon, cities["sacramento"]["lat"],    cities["sacramento"]["lon"])

    return {
        "longitude": lon, "latitude": lat,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms, "total_bedrooms": total_bedrooms,
        "population": population, "households": households,
        "median_income": median_income,
        "rooms_per_household": rooms_per_household,
        "bedrooms_per_room": bedrooms_per_room,
        "population_per_household": population_per_household,
        "dist_to_sf": dist_sf, "dist_to_la": dist_la,
        "dist_to_san_diego": dist_sd, "dist_to_sacramento": dist_sac,
        "ocean_proximity": ocean_proximity
    }


# Serves index.html at http://localhost:5000  (key injected here, never stored in HTML)
@app.route("/")
def serve_index():
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    if not api_key:
        return "<h2>Error: GOOGLE_MAPS_API_KEY not set in .env</h2>", 500

    with open(os.path.join(os.path.dirname(__file__), "index.html"), "r") as f:
        html = f.read()

    # Inject the key into the Maps script tag at runtime
    html = html.replace("__GOOGLE_MAPS_API_KEY__", api_key)
    return html


@app.route("/predict/location", methods=["POST"])
def predict_location():
    data = request.json
    lat, lon = float(data["lat"]), float(data["lon"])
    row = build_feature_row(lat, lon)
    features = config["features"]["numeric"] + config["features"]["categorical"]
    price = float(model.predict(pd.DataFrame([row])[features])[0])
    return jsonify({"predicted_price": round(max(0, price), 2)})


@app.route("/predict/address", methods=["POST"])
def predict_address():
    data = request.json
    lat             = float(data["lat"])
    lon             = float(data["lon"])
    housing_median_age = float(data.get("house_age", 20))
    bedrooms        = int(data.get("bedrooms", 3))
    sqft            = float(data.get("sqft", 1500))
    median_income   = float(data.get("median_income", 4.0))

    total_rooms    = sqft / 150
    total_bedrooms = bedrooms
    households     = 300
    population     = households * 2.5

    row = build_feature_row(lat, lon,
        housing_median_age=housing_median_age,
        total_rooms=total_rooms, total_bedrooms=total_bedrooms,
        population=population, households=households,
        median_income=median_income)

    features = config["features"]["numeric"] + config["features"]["categorical"]
    price = float(model.predict(pd.DataFrame([row])[features])[0])
    price = max(0, price)

    return jsonify({
        "predicted_price": round(price, 2),
        "confidence_low":  round(price * 0.85, 2),
        "confidence_high": round(price * 1.15, 2)
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(
        host=config["server"]["host"],
        port=config["server"]["port"],
        debug=config["server"]["debug"]
    )
