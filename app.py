from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from collections import deque
from datetime import datetime
import os
import math

app = Flask(__name__)
CORS(app)

reg = joblib.load("fan_regressor_v4.pkl")
stats = joblib.load("z_stats_v4.pkl")
calib = joblib.load("calibration_v4.pkl")

BUFFER = deque(maxlen=20)

BASE_FEATURES = ["cpu_usage","cpu_temp","gpu_temp","power","cpu_freq"]

CSV_FILE = "logs.csv"


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


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # ✅ device handling
    device_id = data.get("device_id", "frontend")

    # ✅ timestamp
    timestamp = datetime.utcnow().isoformat()

    df = pd.DataFrame([data])

    # digital twin
    df["predicted_fan"] = reg.predict(df[BASE_FEATURES])

    df["fan_error"] = df["fan1"] - df["predicted_fan"]

    # load stats
    load = data.get("load", "MED")
    row = stats[stats["load"] == load].iloc[0]

    z = (df["fan_error"][0] - row["mu"]) / row["sigma"]
    z = np.clip(z, -5, 5)
    z_abs = abs(z)

    score = compute_score(z_abs)
    health = get_health(score)

    # =========================
    # 📦 SAVE TO CSV
    # =========================
    log_entry = {
        "timestamp": timestamp,
        "device_id": device_id,
        **data,  # all incoming data
        "predicted_fan": float(df["predicted_fan"][0]),
        "fan_error": float(df["fan_error"][0]),
        "z_score": float(z),
        "anomaly_score": float(score),
        "health": health
    }

    log_df = pd.DataFrame([log_entry])

    # append safely
    if not os.path.exists(CSV_FILE):
        log_df.to_csv(CSV_FILE, index=False)
    else:
        log_df.to_csv(CSV_FILE, mode='a', header=False, index=False)

    # =========================

    return jsonify({
        "predicted_fan": float(df["predicted_fan"][0]),
        "fan_error": float(df["fan_error"][0]),
        "z_score": float(z),
        "anomaly_score": float(score),
        "health": health
    })


@app.route("/predict_series", methods=["POST"])
def predict_series():
    try:
        data = request.json  # expects list of dicts

        if not isinstance(data, list):
            return jsonify({"error": "Expected a list of data points"}), 400

        df = pd.DataFrame(data)

        # check required columns
        for col in BASE_FEATURES + ["fan1"]:
            if col not in df.columns:
                return jsonify({"error": f"Missing column: {col}"}), 400

        # predictions
        df["predicted_fan"] = reg.predict(df[BASE_FEATURES])
        df["fan_error"] = df["fan1"] - df["predicted_fan"]

        results = []

        for i in range(len(df)):
            load = df.iloc[i].get("load", "MED")

            row = stats[stats["load"] == load].iloc[0]

            z = (df.iloc[i]["fan_error"] - row["mu"]) / row["sigma"]
            z = np.clip(z, -5, 5)
            z_abs = abs(z)

            score = compute_score(z_abs)
            health = get_health(score)

            results.append({
                "predicted_fan": float(df.iloc[i]["predicted_fan"]),
                "fan_error": float(df.iloc[i]["fan_error"]),
                "z_score": float(z),
                "anomaly_score": float(score),
                "health": health
            })

        return jsonify({
            "count": len(results),
            "results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)})
    
    
@app.route("/logs", methods=["GET"])
def get_logs():
    try:
        # query params
        offset = int(request.args.get("offset", 0))
        limit = int(request.args.get("limit", 10))

        df = pd.read_csv("logs.csv")

        total_rows = len(df)

        # slice data
        sliced = df.iloc[offset:offset+limit]

        return jsonify({
            "data": sliced.to_dict(orient="records"),
            "next_offset": offset + len(sliced),
            "total_rows": total_rows
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True) 