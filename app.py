from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import onnxruntime as ort

app = Flask(__name__)

# =========================
# MODEL
# =========================
class ONNXModelWrapper:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, x):
        return self.session.run(None, {self.input_name: np.array(x).astype(np.float32)})[0]

model = ONNXModelWrapper("lstm_model.onnx")
scaler = joblib.load("scaler.pkl")

WINDOW = 50
TOTAL_LIFE = 8 * 365 * 24 * 60 * 60

features = [
    "fan1","fan2",
    "cpu_temp","gpu_temp","nvidia_temp",
    "cpu_usage",
    "current","power"
]

# =========================
# CORE
# =========================
def predict_rul_series(df):

    df = df.copy().fillna(0)
    df[features] = scaler.transform(df[features])

    rul_list = []

    for i in range(len(df)):

        if i < WINDOW:
            rul_list.append(None)
            continue

        window = df[features].values[i-WINDOW:i]
        window = np.expand_dims(window, axis=0)

        pred = model.predict(window)[0][0]

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
# ROUTE
# =========================
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame(data)

    rul_series = predict_rul_series(df)

    return jsonify({"rul_series": rul_series})

# IMPORTANT: export app
handler = app
