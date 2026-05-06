from flask import Flask, request, jsonify
from flask_cors import CORS

from datetime import datetime

from supabase import create_client

import numpy as np
import joblib
import os
import time

import onnxruntime as ort


# =========================================================
# ENV VARIABLES
# =========================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)
CORS(app)


# =========================================================
# MODEL + SCALER
# =========================================================

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "model.onnx")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

WINDOW_SIZE = 20

print("[INFO] Loading ONNX model...")

session = ort.InferenceSession(MODEL_PATH)

input_name = session.get_inputs()[0].name

print("[INFO] Loading scaler...")

scaler = joblib.load(SCALER_PATH)

print("[INFO] Backend initialized successfully")


# =========================================================
# HEALTH LOGIC
# =========================================================

def compute_health_index(residual):

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

    return max(0, int(health_index * 200))


def compute_trend(residuals):

    if len(residuals) < 2:
        return 0.0

    x = np.arange(len(residuals))
    y = np.array(residuals)

    slope = np.polyfit(x, y, 1)[0]

    return round(float(slope), 6)


# =========================================================
# ROOT
# =========================================================

@app.route("/")
def home():

    return jsonify({
        "status": "online",
        "service": "Predictive Maintenance API"
    })


# =========================================================
# TELEMETRY ENDPOINT
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
        # STORE RAW TELEMETRY
        # -------------------------------------------------

        raw_payload = {

            "timestamp": timestamp,

            "cpu_usage": cpu_usage,
            "temperature": temperature,
            "power": power,
            "frequency": frequency,
            "fan_rpm": fan_rpm
        }

        supabase.table(
            "telemetry_logs"
        ).insert(raw_payload).execute()

        # -------------------------------------------------
        # FETCH LAST 20 ROWS
        # -------------------------------------------------

        response = (
            supabase
            .table("telemetry_logs")
            .select("*")
            .order("timestamp", desc=True)
            .limit(WINDOW_SIZE)
            .execute()
        )

        rows = list(reversed(response.data))

        # -------------------------------------------------
        # WAIT FOR WINDOW
        # -------------------------------------------------

        if len(rows) < WINDOW_SIZE:

            return jsonify({
                "status": "buffering",
                "message":
                    f"Collecting telemetry window ({len(rows)}/{WINDOW_SIZE})"
            })

        # -------------------------------------------------
        # BUILD SEQUENCE
        # -------------------------------------------------

        sequence_data = []

        for row in rows:

            sequence_data.append([
                row["cpu_usage"],
                row["temperature"],
                row["power"],
                row["frequency"],
                row["fan_rpm"]
            ])

        # -------------------------------------------------
        # SCALE FEATURES
        # -------------------------------------------------

        scaled_sequence = scaler.transform(sequence_data)

        sequence = np.array(
            scaled_sequence
        ).reshape(
            1,
            WINDOW_SIZE,
            5
        ).astype(np.float32)

        # -------------------------------------------------
        # ONNX INFERENCE
        # -------------------------------------------------

        prediction = session.run(
            None,
            {
                input_name: sequence
            }
        )

        predicted_rpm = float(
            prediction[0][0][0]
        )

        # -------------------------------------------------
        # RESIDUAL ANALYSIS
        # -------------------------------------------------

        residual = abs(
            predicted_rpm - fan_rpm
        )

        residuals = []

        for row in rows:

            if row.get("residual") is not None:
                residuals.append(row["residual"])

        residuals.append(residual)

        health_index = compute_health_index(
            residual
        )

        health_state = classify_health(
            health_index
        )

        estimated_rul = estimate_rul(
            health_index
        )

        trend_slope = compute_trend(
            residuals
        )

        inference_time_ms = round(
            (time.time() - start_time) * 1000,
            2
        )

        # -------------------------------------------------
        # UPDATE LATEST ROW
        # -------------------------------------------------

        latest_id = rows[-1]["id"]

        update_payload = {

            "predicted_rpm":
                predicted_rpm,

            "residual":
                residual,

            "health_index":
                health_index,

            "health_state":
                health_state,

            "estimated_rul":
                estimated_rul,

            "trend_slope":
                trend_slope,

            "inference_time_ms":
                inference_time_ms
        }

        (
            supabase
            .table("telemetry_logs")
            .update(update_payload)
            .eq("id", latest_id)
            .execute()
        )

        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        return jsonify({

            "status": "success",

            "telemetry": {

                "cpu_usage":
                    cpu_usage,

                "temperature":
                    temperature,

                "power":
                    power,

                "frequency":
                    frequency,

                "fan_rpm":
                    fan_rpm
            },

            "prediction": {

                "predicted_rpm":
                    round(predicted_rpm, 2),

                "actual_rpm":
                    fan_rpm,

                "residual":
                    round(residual, 2)
            },

            "health": {

                "health_index":
                    health_index,

                "health_state":
                    health_state
            },

            "forecast": {

                "estimated_rul":
                    estimated_rul,

                "trend_slope":
                    trend_slope
            },

            "system": {

                "inference_time_ms":
                    inference_time_ms
            }

        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# =========================================================
# HISTORY
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
                latest.get("health_index"),

            "current_rul":
                latest.get("estimated_rul"),

            "health_state":
                latest.get("health_state"),

            "latest_residual":
                latest.get("residual"),

            "trend_slope":
                latest.get("trend_slope")
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )