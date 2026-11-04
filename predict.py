
# predict.py
# ML Zoomcamp 2026 Midterm Project
# Flask API for 4G/5G signal level prediction

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_signal_model.bin"

FEATURES = [
    "Latitude",
    "Longitude",
    "Speed",
    "Operatorname",
    "NetworkTech",
    "Mobility",
    "Altitude",
]

NUMERIC_FEATURES = [
    "Latitude",
    "Longitude",
    "Speed",
    "Altitude",
]

CATEGORICAL_FEATURES = [
    "Operatorname",
    "NetworkTech",
    "Mobility",
]

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Request body must be a JSON object"
        }), 400

    missing = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing:
        return jsonify({
            "error": "Missing required features",
            "missing": missing
        }), 400

    try:
        record = {feature: data[feature] for feature in FEATURES}

        for feature in NUMERIC_FEATURES:
            value = record[feature]

            if isinstance(value, bool):
                raise ValueError(f"{feature} must be numeric")

            record[feature] = float(value)

            if not np.isfinite(record[feature]):
                raise ValueError(f"{feature} must be finite")

        for feature in CATEGORICAL_FEATURES:
            if not isinstance(record[feature], str):
                raise ValueError(f"{feature} must be a string")

        if record["NetworkTech"] not in ("4G", "5G"):
            raise ValueError("NetworkTech must be 4G or 5G")

        if record["Operatorname"] not in (
            "Operator A", "Operator B", "Operator C"
        ):
            raise ValueError("Invalid Operatorname")

        if record["Mobility"] not in (
            "BRT", "Shuttle", "Walk"
        ):
            raise ValueError("Invalid Mobility")

        X = pd.DataFrame([record], columns=FEATURES)

        prediction = float(model.predict(X)[0])

        return jsonify({
            "predicted_signal_level_dbm": round(prediction, 4)
        })

    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    from waitress import serve

    print("Starting signal prediction API...")
    serve(app, host="0.0.0.0", port=9696)
