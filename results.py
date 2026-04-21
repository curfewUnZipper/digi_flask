# =========================
# 📦 IMPORTS
# =========================
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

# =========================
# CONFIG
# =========================
DATA_FILE = "new.csv"

REG_MODEL_FILE = "fan_regressor_v3.pkl"
CLF_MODEL_FILE = "fan_classifier_v3.pkl"

TARGET = "fan1"
LOAD_COL = "load"

BASE_FEATURES = [
    "cpu_usage",
    "cpu_temp",
    "gpu_temp",
    "power",
    "cpu_freq"
]

# =========================
# LOAD
# =========================
df = pd.read_csv(DATA_FILE).dropna()
df = df.sort_values(by="timestamp")

reg_model = joblib.load(REG_MODEL_FILE)
clf_model = joblib.load(CLF_MODEL_FILE)

# =========================
# 🔮 DIGITAL TWIN
# =========================
df["predicted_fan"] = reg_model.predict(df[BASE_FEATURES])

# =========================
# ⚡ MISMATCH FEATURES
# =========================
df["fan_error"] = df[TARGET] - df["predicted_fan"]
df["fan_error"] = df["fan_error"].clip(-30, 30)

df["fan_ratio"] = df[TARGET] / (df["predicted_fan"] + 1e-6)
df["temp_per_fan"] = df["cpu_temp"] / (df[TARGET] + 1e-6)
df["power_per_fan"] = df["power"] / (df[TARGET] + 1e-6)

df["fan_error_pct"] = df["fan_error"] / (df["predicted_fan"] + 1e-6)

# =========================
# ⏱️ TEMPORAL FEATURES
# =========================
WINDOW = 20

df["fan_trend"] = df[TARGET].rolling(WINDOW).mean()
df["temp_trend"] = df["cpu_temp"].rolling(WINDOW).mean()

df["fan_slope"] = df[TARGET].diff()
df["temp_slope"] = df["cpu_temp"].diff()

df["fan_variance"] = df[TARGET].rolling(WINDOW).std()
df["temp_variance"] = df["cpu_temp"].rolling(WINDOW).std()

df["fan_error_long"] = df["fan_error"].rolling(WINDOW).mean()
df["temp_trend_long"] = df["cpu_temp"].rolling(WINDOW).mean()

df["fan_error_cumsum"] = df["fan_error"].cumsum()

# FIX deprecated warning
df = df.bfill()

# =========================
# 🏷️ LABELS (same logic)
# =========================
df["residual"] = df[TARGET] - df["predicted_fan"]

stats = df.groupby(LOAD_COL)["residual"].agg(["mean", "std"]).reset_index()
stats.columns = [LOAD_COL, "mu", "sigma"]

df = df.merge(stats, on=LOAD_COL)
df["sigma"] = df["sigma"].replace(0, 1e-6)

df["z_score"] = (df["residual"] - df["mu"]) / df["sigma"]
df["z_abs"] = df["z_score"].abs()

def classify(z):
    if z < 1.5:
        return 0
    elif z < 3:
        return 1
    elif z < 5:
        return 2
    else:
        return 3

df["true_label"] = df["z_abs"].apply(classify)

# =========================
# 🔥 EXACT SAME FEATURES AS TRAINING
# =========================
clf_features = [
    "fan_error",
    "fan_ratio",
    "fan_error_pct",
    "temp_per_fan",
    "power_per_fan",
    "fan_trend",
    "temp_trend",
    "fan_slope",
    "temp_slope",
    "fan_variance",
    "temp_variance",
    "fan_error_long",
    "temp_trend_long",
    "fan_error_cumsum",
    "cpu_usage",
    "cpu_temp",
    "power"
]

X = df[clf_features]

# =========================
# PREDICTION
# =========================
df["pred"] = clf_model.predict(X)

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(df["true_label"], df["pred"])

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (v3)")
plt.savefig("confusion_matrix_v3.png")
plt.close()

# =========================
# REPORT
# =========================
report = classification_report(df["true_label"], df["pred"])

with open("classification_report_v3.txt", "w") as f:
    f.write(report)

print("\n✅ Evaluation updated successfully!")