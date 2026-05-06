import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.models import load_model

from preprocess import (
    load_csv,
    transform,
    create_sequences
)

os.makedirs("plots", exist_ok=True)

MODEL_PATH = "models/lstm_model.keras"
SCALER_PATH = "models/scaler.pkl"

TEST_FILE = "data/raw/degraded.csv"

# -----------------------------------
# Load
# -----------------------------------

model = load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

df = load_csv(TEST_FILE)

scaled = transform(df, scaler)

X, y = create_sequences(scaled)

# -----------------------------------
# Predict
# -----------------------------------

pred = model.predict(X).flatten()

# -----------------------------------
# Residual
# -----------------------------------

residuals = np.abs(y - pred)

# -----------------------------------
# Health Index
# -----------------------------------

moving = pd.Series(residuals).rolling(50).mean()

health = np.exp(-5 * moving)

# -----------------------------------
# Threshold
# -----------------------------------

threshold = residuals.mean() + 3 * residuals.std()

print("\nThreshold:", threshold)

# -----------------------------------
# Save metrics
# -----------------------------------

print("\nResidual Mean:", residuals.mean())
print("Residual Std:", residuals.std())

# -----------------------------------
# Plot residuals
# -----------------------------------

plt.figure(figsize=(12, 5))

plt.plot(residuals)

plt.axhline(
    threshold,
    linestyle="--"
)

plt.title("Residual Error")
plt.xlabel("Time")
plt.ylabel("Residual")

plt.savefig("plots/residuals.png")

# -----------------------------------
# Plot health index
# -----------------------------------

plt.figure(figsize=(12, 5))

plt.plot(health)

plt.title("Health Index")
plt.xlabel("Time")
plt.ylabel("HI")

plt.savefig("plots/health_index.png")

print("\nPlots saved.")