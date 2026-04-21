# =========================
# 📦 IMPORTS
# =========================
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# =========================
# ⚙️ CONFIG
# =========================
DATA_FILE = "new.csv"
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
# 📂 LOAD DATA
# =========================
df = pd.read_csv(DATA_FILE).dropna()
df = df.sort_values(by="timestamp")

# =========================
# 🧠 SPLIT FIRST (IMPORTANT)
# =========================
X = df[BASE_FEATURES]
y = df[TARGET]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 🔮 REGRESSION MODEL
# =========================
reg_model = RandomForestRegressor(n_estimators=50, max_depth=10)
reg_model.fit(X_train_reg, y_train_reg)

df["predicted_fan"] = reg_model.predict(X)

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

df = df.fillna(method="bfill")

# =========================
# 🏷️ LABELS (z-score)
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

df["label"] = df["z_abs"].apply(classify)

# =========================
# 🤖 CLASSIFIER FEATURES
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

X_clf = df[clf_features]
y_clf = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# =========================
# 🧪 MODEL TRAINING
# =========================
models = {
    "LogisticRegression": LogisticRegression(max_iter=300),
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced"
    )
}

best_model = None
best_score = 0
best_name = ""

print("\n📊 Model Comparison:")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    print(f"{name} → Accuracy: {acc:.3f}, F1: {f1:.3f}")

    if f1 > best_score:
        best_score = f1
        best_model = model
        best_name = name

print(f"\n🏆 Best Model: {best_name} (F1: {best_score:.3f})")

# =========================
# 💾 SAVE
# =========================
joblib.dump(reg_model, "fan_regressor_v3.pkl")
joblib.dump(best_model, "fan_classifier_v3.pkl")

print("\n✅ Updated models saved!")