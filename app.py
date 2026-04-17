# =========================
# 📦 IMPORTS
# =========================
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# =========================
# LOAD MODEL + SCALER
# =========================
model = load_model("lstm_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

WINDOW = 50
TOTAL_LIFE = 8 * 365 * 24 * 60 * 60  # 8 years in seconds

features = [
    "fan1","fan2",
    "cpu_temp","gpu_temp","nvidia_temp",
    "cpu_usage",
    "current","power"
]


# =========================
# 🔁 CORE: RUL SERIES
# =========================
def predict_rul_series(df):

    df = df.copy()
    df[features] = scaler.transform(df[features])

    rul_list = []

    for i in range(len(df)):

        if i < WINDOW:
            rul_list.append(None)
            continue

        window = df[features].values[i-WINDOW:i]
        window = np.expand_dims(window, axis=0)

        pred = model.predict(window, verbose=0)[0][0]

        # convert to real units
        rul_seconds = pred * TOTAL_LIFE
        rul_hours = rul_seconds / 3600
        rul_years = rul_seconds / (365 * 24 * 3600)

        rul_list.append({
            "normalized": float(pred),
            "seconds": float(rul_seconds),
            "hours": float(rul_hours),
            "years": float(rul_years)
        })

    return rul_list


# =========================
# 📡 API
# =========================
@app.route("/predict_series", methods=["POST"])
def predict_series():

    data = request.json
    df = pd.DataFrame(data)

    rul_series = predict_rul_series(df)

    return jsonify({
        "rul_series": rul_series
    })


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)