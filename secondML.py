# =========================
# 📦 IMPORTS
# =========================
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

# =========================
# ⚙️ CONFIG
# =========================
DATA_FILE = "synthetic_degrading.csv"

WINDOW = 20
HORIZON = 30

THRESHOLD = 0.3   # 🔥 tune this later

BASE_FEATURES = [
    "cpu_usage", "cpu_temp", "gpu_temp", "power", "cpu_freq"
]

FEATURES = [
    "cpu_temp",
    "cpu_usage",
    "power",
    "fan1",
    "predicted_fan",
    "fan_error",
    "anomaly_score",
    "anomaly_diff",
    "fan_error_trend"
]

# =========================
# 📂 LOAD DATA
# =========================
df = pd.read_csv(DATA_FILE)

df = df.sort_values("timestamp").reset_index(drop=True)

# =========================
# 🧠 LOAD MODELS
# =========================
reg = joblib.load("fan_regressor_v4.pkl")
stats = joblib.load("z_stats_v4.pkl")

# =========================
# 🔧 FEATURE ENGINEERING
# =========================
df["predicted_fan"] = reg.predict(df[BASE_FEATURES])
df["fan_error"] = df["fan1"] - df["predicted_fan"]

df = df.merge(stats, on="load", how="left")
df["sigma"] = df["sigma"].replace(0, 1e-6).fillna(1)

df["z"] = (df["fan_error"] - df["mu"]) / df["sigma"]
df["z"] = df["z"].clip(-5, 5)

df["anomaly_score"] = np.log1p(np.abs(df["z"])) * 40

# 🔥 TREND FEATURES
df["anomaly_diff"] = df["anomaly_score"].diff().fillna(0)
df["fan_error_trend"] = df["fan_error"].rolling(5).mean().fillna(0)

df = df.dropna()

# =========================
# 🔄 NORMALIZATION
# =========================
scaler = MinMaxScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])

joblib.dump(scaler, "lstm_scaler.pkl")

# =========================
# 📊 BUILD SEQUENCES
# =========================
X, y = [], []

for i in range(len(df) - WINDOW - HORIZON):

    seq = df.iloc[i:i+WINDOW][FEATURES].values

    current = df.iloc[i+WINDOW]["anomaly_score"]
    future = df.iloc[i+WINDOW+HORIZON]["anomaly_score"]

    delta = future - current

    # 🔥 LABEL
    if delta > 0.05:
        label = 1   # worsening
    else:
        label = 0   # stable

    X.append(seq)
    y.append(label)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Class distribution:", np.bincount(y))

# =========================
# ✂️ SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# =========================
# ⚖️ CLASS WEIGHTS
# =========================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# =========================
# 🤖 MODEL
# =========================
model = Sequential([
    LSTM(32, input_shape=(WINDOW, len(FEATURES))),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# =========================
# 🏋️ TRAIN
# =========================
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop],
    class_weight=class_weights   # 🔥 critical
)

# =========================
# 📊 EVALUATION
# =========================
y_pred_prob = model.predict(X_test).flatten()

print("\nSample probabilities:", y_pred_prob[:20])

# 🔥 threshold tuning
y_pred = (y_pred_prob > THRESHOLD).astype(int)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# 📈 PROBABILITY PLOT
# =========================
plt.figure(figsize=(10,4))
plt.plot(y_pred_prob[:200], label="Predicted Prob")
plt.title("Prediction Probabilities")
plt.legend()
plt.savefig("trend_prob_plot.png")
plt.close()

print("✅ Plot saved!")

# =========================
# 💾 SAVE MODEL
# =========================
model.save("lstm_trend_model.keras")

print("✅ Trend model saved!")