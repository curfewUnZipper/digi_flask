from preprocess import load_csv
from preprocess import transform, create_sequences

from rul_estimator import estimate_rul

import pandas as pd
import numpy as np
import joblib
import os

from tensorflow.keras.models import load_model

# =====================================================
# MODEL + SCALER
# =====================================================

MODEL = "models/lstm_model.keras"
SCALER = "models/scaler.pkl"

model = load_model(MODEL)

scaler = joblib.load(SCALER)

# =====================================================
# FILES TO TEST
# =====================================================

FILES = [

    # Healthy baseline
    "data/raw/normal.csv",

    # Real degraded system
    "data/raw/degraded.csv",

    # Synthetic progression
    "data/synthetic/month_1.csv",
    "data/synthetic/month_3.csv",
    "data/synthetic/month_6.csv",
    "data/synthetic/year_1.csv",
    "data/synthetic/year_2.csv",
    "data/synthetic/year_3.csv",
]

# =====================================================
# RUN EACH FILE
# =====================================================

for FILE in FILES:

    # ---------------------------------------------
    # Check file exists
    # ---------------------------------------------

    if not os.path.exists(FILE):

        print("\nMissing:", FILE)

        continue

    # ---------------------------------------------
    # LOAD
    # ---------------------------------------------

    df = load_csv(FILE)

    scaled = transform(df, scaler)

    X, y = create_sequences(scaled)

    # ---------------------------------------------
    # PREDICT
    # ---------------------------------------------

    pred = model.predict(
        X,
        verbose=0
    ).flatten()

    residuals = np.abs(y - pred)

    # ---------------------------------------------
    # HEALTH INDEX
    # ---------------------------------------------

    health = np.exp(
        -1.5 *
        pd.Series(residuals)
        .rolling(50)
        .mean()
    )

    latest_health = health.iloc[-1]

    # ---------------------------------------------
    # RUL
    # ---------------------------------------------

    rul = estimate_rul(residuals)

    # ---------------------------------------------
    # STATUS LOGIC
    # ---------------------------------------------

    if latest_health > 0.8:
        status = "HEALTHY"

    elif latest_health > 0.5:
        status = "MINOR DEGRADATION"

    elif latest_health > 0.2:
        status = "MODERATE DEGRADATION"

    else:
        status = "CRITICAL"

    # ---------------------------------------------
    # PRINT
    # ---------------------------------------------

    print("\n")
    print("=" * 60)

    print("FILE:")
    print(FILE)

    print("=" * 60)

    print("\nSYSTEM STATUS:")
    print(status)

    print("\nResidual Mean:")
    print(round(residuals.mean(), 6))

    print("\nResidual Std:")
    print(round(residuals.std(), 6))

    print("\nLatest Health Index:")
    print(round(float(latest_health), 6))

    print("\nEstimated Maintenance Horizon:")
    print(round(float(rul["estimated_rul"]), 2))

    print("\nTrend Slope:")
    print(rul["slope"])

    print("\nCritical Threshold:")
    print(rul["critical_threshold"])

    print("\n")