from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from collections import deque
import math
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

reg   = joblib.load(os.path.join(BASE_DIR, "fan_regressor_v4.pkl"))
stats = joblib.load(os.path.join(BASE_DIR, "z_stats_v4.pkl"))
calib = joblib.load(os.path.join(BASE_DIR, "calibration_v4.pkl"))

BUFFER = deque(maxlen=20)

BASE_FEATURES = ["cpu_usage", "cpu_temp", "gpu_temp", "power", "cpu_freq"]


def compute_score(z_abs):
    score = math.log1p(z_abs) * 40
    return min(score, 100)


def get_health(score):
    if score < 35:
        return "Normal"
    elif score < 70:
        return "Degrading"
    else:
        return "Critical"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        if not data:
            return jsonify({"error": "No JSON body received"}), 400

        missing = [f for f in BASE_FEATURES + ["fan1"] if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        df = pd.DataFrame([data])

        df["predicted_fan"] = reg.predict(df[BASE_FEATURES])
        df["fan_error"] = df["fan1"] - df["predicted_fan"]

        load = data.get("load", "MED")
        matched = stats[stats["load"] == load]
        if matched.empty:
            return jsonify({"error": f"Unknown load value: {load}"}), 400

        row = matched.iloc[0]
        z = (df["fan_error"][0] - row["mu"]) / row["sigma"]
        z = np.clip(z, -5, 5)
        z_abs = abs(z)

        score = compute_score(z_abs)
        health_status = get_health(score)

        return jsonify({
            "predicted_fan": float(df["predicted_fan"][0]),
            "fan_error":     float(df["fan_error"][0]),
            "z_score":       float(z),
            "anomaly_score": float(score),
            "health":        health_status
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
