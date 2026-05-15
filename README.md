# California House Price Predictor

Predict median house prices anywhere in California using XGBoost. Two modes: click a heatmap to see area prices, or enter an address with house details to get a full price prediction with confidence range.

## Features

* **Heatmap Mode** — click anywhere on a California map to get a predicted median price for that area, visualized as a color overlay
* **Address Lookup Mode** — enter an address and house details (bedrooms, sqft, year built) for a pin-dropped prediction with a ±15% confidence range
* **XGBoost model** trained on ~20,000 California housing records with engineered features (distance to cities, room ratios, and more)
* **WandB tracking** — feature importance chart, predicted vs actual scatter plot, and RMSE logged per run
* **Fully config-driven** — all hyperparameters and paths live in `configs/config.yaml`, nothing hardcoded

## Tech Stack

| Layer              | What                          |
|--------------------|-------------------------------|
| ML Model           | XGBoost Regressor             |
| Dataset            | California Housing (Kaggle)   |
| Backend            | Flask + Flask-CORS            |
| Frontend           | HTML / CSS / JS               |
| Map                | Google Maps JavaScript API    |
| Experiment Tracking| Weights & Biases (WandB)      |
| Config             | YAML                          |

## Project Structure

```
california-house-predictor/
│
├── data/
│   ├── raw/                    ← put housing.csv here
│   └── processed/              ← auto-generated after first run
│
├── src/
│   ├── model.py                ← XGBoost model wrapper
│   ├── train_local.py          ← training script for GTX 1650
│   ├── train.py                ← base training pipeline + WandB logging
│   ├── evaluate.py             ← metrics, feature importance, WandB charts
│   ├── dataset.py              ← data loading, cleaning, feature engineering
│   └── utils.py                ← config loader, checkpointing, haversine distance
│
├── configs/
│   └── config.yaml             ← all hyperparameters, paths, settings
│
├── notebooks/
│   └── train_colab.ipynb       ← self-contained Colab training notebook
│
├── checkpoints/                ← saved model goes here (gitignored)
│
├── app/
│   ├── server.py               ← Flask inference server
│   └── index.html              ← map + prediction web app
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

### 1. Clone the repo and install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/California-House-Price-Predictor.git
cd California-House-Price-Predictor
pip install -r requirements.txt
```

### 2. Get the dataset

1. Go to [kaggle.com/datasets/camnugent/california-housing-prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
2. Download `housing.csv`
3. Place it at `data/raw/housing.csv`

### 3. Set up Google Maps API

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a new project
3. Enable these three APIs:
   * Maps JavaScript API
   * Geocoding API
   * Places API
4. Go to **Credentials → Create API Key**
5. Copy your key and paste it into `app/index.html`, replacing `YOUR_GOOGLE_MAPS_API_KEY`

### 4. Log in to WandB

```bash
wandb login
```
Paste your API key from [wandb.ai/settings](https://wandb.ai/settings) when prompted.

## Training

### Option A — Local (GTX 1650)

Open `configs/config.yaml` and set:
```yaml
tree_method: gpu_hist
```
Then run:
```bash
python src/train_local.py
```

### Option B — Google Colab (free GPU/CPU)

1. Open `notebooks/train_colab.ipynb` in Colab
2. Run all cells top to bottom
3. Upload `housing.csv` when prompted
4. Download `xgb_california.pkl` at the end
5. Place it in your local `checkpoints/` folder

## Running the App

```bash
python app/server.py
```

Then open `app/index.html` in your browser. The Flask server must be running for predictions to work.

## Transferring the Model (1650 → i7)

1. Upload `checkpoints/xgb_california.pkl` to Google Drive
2. Download it on your i7 machine
3. Place it at `checkpoints/xgb_california.pkl`
4. Run `python app/server.py` — no GPU needed, just loads the saved model

---

Author: Vedant Daga
