from flask import Flask, request, jsonify
from flask_cors import CORS

from collections import deque
from datetime import datetime

from supabase import create_client
from dotenv import load_dotenv

import numpy as np
import joblib
import os
import time

import onnxruntime as ort

# =========================================================
# LOAD ENV VARIABLES
# =========================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
print(SUPABASE_URL)
print(SUPABASE_KEY[:20])
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)
CORS(app)


# =========================================================
# LOAD MODEL + SCALER
# =========================================================
MODEL_PATH = "model.onnx"
SCALER_PATH = "scaler.pkl"

print("[INFO] Loading ONNX model...")

session = ort.InferenceSession(MODEL_PATH)

input_name = session.get_inputs()[0].name

print("[INFO] Loading scaler...")
scaler = joblib.load(SCALER_PATH)

print("[INFO] Model and scaler loaded successfully")

# =========================================================
# GLOBAL BUFFER
# =========================================================

WINDOW_SIZE = 20

telemetry_buffer = deque(maxlen=WINDOW_SIZE)

# For simple degradation tracking
residual_history = deque(maxlen=100)


# =========================================================
# HEALTH LOGIC
# =========================================================

def compute_health_index(residual):

    # Simple normalized degradation logic
    normalized = min(residual / 1000, 1.0)

    health_index = max(0.0, 1.0 - normalized)

    return round(health_index, 3)


def classify_health(health_index):

    if health_index >= 0.8:
        return "HEALTHY"

    elif health_index >= 0.5:
        return "MINOR"

    elif health_index >= 0.2:
        return "MODERATE"

    return "CRITICAL"


def estimate_rul(health_index):

    # Simple RUL estimation
    rul = int(health_index * 200)

    return max(rul, 0)


def compute_trend():

    if len(residual_history) < 2:
        return 0.0

    x = np.arange(len(residual_history))
    y = np.array(residual_history)

    slope = np.polyfit(x, y, 1)[0]

    return round(float(slope), 6)


# =========================================================
# ROOT
# =========================================================

@app.route("/")
def home():

    return jsonify({
        "status": "online",
        "service": "Predictive Maintenance Digital Twin API"
    })


# =========================================================
# TELEMETRY INGESTION
# =========================================================

@app.route("/telemetry", methods=["POST"])
def telemetry():

    try:

        start_time = time.time()

        data = request.json

        # -------------------------------------------------
        # INPUTS
        # -------------------------------------------------

        cpu_usage = float(data["cpu_usage"])
        temperature = float(data["temperature"])
        power = float(data["power"])
        frequency = float(data["frequency"])
        fan_rpm = float(data["fan_rpm"])

        timestamp = data.get(
            "timestamp",
            datetime.utcnow().isoformat()
        )

        # -------------------------------------------------
        # PREPARE INPUT
        # -------------------------------------------------

        features = np.array([[
            cpu_usage,
            temperature,
            power,
            frequency,
            fan_rpm
        ]])

        scaled_features = scaler.transform(features)

        telemetry_buffer.append(scaled_features[0])

        # -------------------------------------------------
        # WAIT FOR BUFFER
        # -------------------------------------------------

        if len(telemetry_buffer) < WINDOW_SIZE:

            return jsonify({
                "status": "buffering",
                "message": f"Collecting telemetry window ({len(telemetry_buffer)}/{WINDOW_SIZE})"
            })

        # -------------------------------------------------
        # CREATE SEQUENCE
        # -------------------------------------------------

        sequence = np.array(telemetry_buffer)

        sequence = sequence.reshape(
            1,
            WINDOW_SIZE,
            scaled_features.shape[1]
        ).astype(np.float32)

        # -------------------------------------------------
        # ONNX INFERENCE
        # -------------------------------------------------

        # ONNX INFERENCE

        prediction = session.run(
            None,
            {
                input_name: sequence.astype(np.float32)
            }
        )

        predicted_rpm = float(prediction[0][0][0])

        # -------------------------------------------------
        # RESIDUAL ANALYSIS
        # -------------------------------------------------

        residual = abs(predicted_rpm - fan_rpm)

        residual_history.append(residual)

        health_index = compute_health_index(residual)

        health_state = classify_health(health_index)

        estimated_rul = estimate_rul(health_index)

        trend_slope = compute_trend()

        inference_time_ms = round(
            (time.time() - start_time) * 1000,
            2
        )

        # -------------------------------------------------
        # STORE IN SUPABASE
        # -------------------------------------------------

        payload = {
            "timestamp": timestamp,

            "cpu_usage": cpu_usage,
            "temperature": temperature,
            "power": power,
            "frequency": frequency,
            "fan_rpm": fan_rpm,

            "predicted_rpm": predicted_rpm,
            "residual": residual,

            "health_index": health_index,
            "health_state": health_state,

            "estimated_rul": estimated_rul,
            "trend_slope": trend_slope,

            "inference_time_ms": inference_time_ms
        }

        supabase.table("telemetry_logs").insert(payload).execute()

        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        return jsonify({

            "status": "success",

            "telemetry": {
                "cpu_usage": cpu_usage,
                "temperature": temperature,
                "power": power,
                "frequency": frequency,
                "fan_rpm": fan_rpm
            },

            "prediction": {
                "predicted_rpm": round(predicted_rpm, 2),
                "actual_rpm": fan_rpm,
                "residual": round(residual, 2)
            },

            "health": {
                "health_index": health_index,
                "health_state": health_state
            },

            "forecast": {
                "estimated_rul": estimated_rul,
                "trend_slope": trend_slope
            },

            "system": {
                "inference_time_ms": inference_time_ms
            }

        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# =========================================================
# GET HISTORY
# =========================================================

@app.route("/history", methods=["GET"])
def history():

    try:

        response = (
            supabase
            .table("telemetry_logs")
            .select("*")
            .order("timestamp", desc=True)
            .limit(200)
            .execute()
        )

        return jsonify(response.data)

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# =========================================================
# SUMMARY
# =========================================================

@app.route("/summary", methods=["GET"])
def summary():

    try:

        response = (
            supabase
            .table("telemetry_logs")
            .select("*")
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
        )

        if not response.data:

            return jsonify({
                "status": "no_data"
            })

        latest = response.data[0]

        return jsonify({

            "current_health_index":
                latest["health_index"],

            "current_rul":
                latest["estimated_rul"],

            "health_state":
                latest["health_state"],

            "latest_residual":
                latest["residual"],

            "trend_slope":
                latest["trend_slope"]

        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )