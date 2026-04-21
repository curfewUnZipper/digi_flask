from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from collections import deque

app = Flask(__name__)
CORS(app)

reg = joblib.load("fan_regressor_v4.pkl")
stats = joblib.load("z_stats_v4.pkl")
calib = joblib.load("calibration_v4.pkl")

BUFFER = deque(maxlen=20)

BASE_FEATURES = ["cpu_usage","cpu_temp","gpu_temp","power","cpu_freq"]

import math

def compute_score(z_abs):
    # log-based scaling (stable)
    score = math.log1p(z_abs) * 40
    return min(score, 100)

def get_health(score):
    if score < 35:
        return "Normal"
    elif score < 70:
        return "Degrading"
    else:
        return "Critical"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    # digital twin
    df["predicted_fan"] = reg.predict(df[BASE_FEATURES])

    df["fan_error"] = df["fan1"] - df["predicted_fan"]

    # load stats
    load = data.get("load", "MED")  # default fallback
    row = stats[stats["load"] == load].iloc[0]

    z = (df["fan_error"][0] - row["mu"]) / row["sigma"]
    z = np.clip(z, -5, 5)
    z_abs = abs(z)

    score = compute_score(z_abs)
    # score = (z_abs / 5) * 100
    health = get_health(score)

    return jsonify({
        "predicted_fan": float(df["predicted_fan"][0]),
        "fan_error": float(df["fan_error"][0]),
        "z_score": float(z),
        "anomaly_score": float(score),
        "health": health
    })


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)