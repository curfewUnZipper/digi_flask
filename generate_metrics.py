# generate_metrics.py

from preprocess import load_csv
from preprocess import transform, create_sequences

from rul_estimator import estimate_rul

import pandas as pd
import numpy as np
import joblib
import os

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =====================================================
# MODEL + SCALER
# =====================================================

MODEL = "models/lstm_model.keras"
SCALER = "models/scaler.pkl"

model = load_model(MODEL)

scaler = joblib.load(SCALER)

# =====================================================
# FILES
# =====================================================

FILES = [

    ("data/raw/normal.csv", "HEALTHY"),

    ("data/raw/degraded.csv", "MODERATE"),

    ("data/synthetic/month_1.csv", "MINOR"),

    ("data/synthetic/month_3.csv", "MINOR"),

    ("data/synthetic/month_6.csv", "MODERATE"),

    ("data/synthetic/year_1.csv", "CRITICAL"),

    ("data/synthetic/year_2.csv", "CRITICAL"),

    ("data/synthetic/year_3.csv", "CRITICAL"),
]

# =====================================================
# HEALTH CLASSIFICATION
# =====================================================

def classify_health(hi):

    if hi > 0.8:
        return "HEALTHY"

    elif hi > 0.5:
        return "MINOR"

    elif hi > 0.2:
        return "MODERATE"

    return "CRITICAL"

# =====================================================
# GLOBAL ARRAYS
# =====================================================

true_labels = []
pred_labels = []

# =====================================================
# PROCESS EACH FILE
# =====================================================

for FILE, true_class in FILES:

    if not os.path.exists(FILE):

        print("\nMissing:", FILE)

        continue

    # -------------------------------------------------
    # LOAD
    # -------------------------------------------------

    df = load_csv(FILE)

    scaled = transform(df, scaler)

    X, y = create_sequences(scaled)

    # -------------------------------------------------
    # PREDICT
    # -------------------------------------------------

    pred = model.predict(
        X,
        verbose=0
    ).flatten()

    # -------------------------------------------------
    # CONVERT BACK TO REAL RPM
    # -------------------------------------------------

    dummy_pred = np.zeros((len(pred), 5))
    dummy_true = np.zeros((len(y), 5))

    dummy_pred[:, 4] = pred
    dummy_true[:, 4] = y

    pred_real = scaler.inverse_transform(dummy_pred)[:, 4]
    true_real = scaler.inverse_transform(dummy_true)[:, 4]

    # -------------------------------------------------
    # RESIDUALS
    # -------------------------------------------------

    residuals = np.abs(
        true_real - pred_real
    )

    # -------------------------------------------------
    # REGRESSION METRICS
    # -------------------------------------------------

    mae = mean_absolute_error(
        true_real,
        pred_real
    )

    mse = mean_squared_error(
        true_real,
        pred_real
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        true_real,
        pred_real
    )

    # -------------------------------------------------
    # HEALTH INDEX
    # -------------------------------------------------

    health = np.exp(
        -1.5 *
        pd.Series(residuals)
        .rolling(50)
        .mean()
    )

    latest_health = float(
        health.iloc[-1]
    )

    predicted_class = classify_health(
        latest_health
    )

    # -------------------------------------------------
    # STORE LABELS
    # -------------------------------------------------

    true_labels.append(true_class)
    pred_labels.append(predicted_class)

    # -------------------------------------------------
    # RUL
    # -------------------------------------------------

    rul = estimate_rul(residuals)

    # -------------------------------------------------
    # PRINT
    # -------------------------------------------------

    print("\n")
    print("=" * 70)

    print("FILE:")
    print(FILE)

    print("=" * 70)

    print("\nTRUE CLASS:")
    print(true_class)

    print("\nPREDICTED CLASS:")
    print(predicted_class)

    print("\nMAE:")
    print(round(mae, 4))

    print("\nMSE:")
    print(round(mse, 4))

    print("\nRMSE:")
    print(round(rmse, 4))

    print("\nR2 SCORE:")
    print(round(r2, 4))

    print("\nResidual Mean:")
    print(round(residuals.mean(), 4))

    print("\nResidual Std:")
    print(round(residuals.std(), 4))

    print("\nLatest Health Index:")
    print(round(latest_health, 4))

    print("\nEstimated Maintenance Horizon:")
    print(round(float(rul["estimated_rul"]), 2))

    print("\nTrend Slope:")
    print(rul["slope"])

    print("\nCritical Threshold:")
    print(rul["critical_threshold"])

# =====================================================
# CLASSIFICATION METRICS
# =====================================================

print("\n\n")
print("=" * 70)
print("OVERALL CLASSIFICATION METRICS")
print("=" * 70)

precision = precision_score(
    true_labels,
    pred_labels,
    average="weighted"
)

recall = recall_score(
    true_labels,
    pred_labels,
    average="weighted"
)

f1 = f1_score(
    true_labels,
    pred_labels,
    average="weighted"
)

print("\nPrecision:")
print(round(precision, 4))

print("\nRecall:")
print(round(recall, 4))

print("\nF1 Score:")
print(round(f1, 4))

# =====================================================
# CONFUSION MATRIX
# =====================================================

labels = [
    "HEALTHY",
    "MINOR",
    "MODERATE",
    "CRITICAL"
]

cm = confusion_matrix(
    true_labels,
    pred_labels,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\nCONFUSION MATRIX:")
print(cm_df)

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\nCLASSIFICATION REPORT:")
print(
    classification_report(
        true_labels,
        pred_labels
    )
)