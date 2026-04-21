# =========================
# IMPORTS
# =========================
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# =========================
# CONFIG
# =========================
DATA_FILE = "new.csv"
TARGET = "fan1"
LOAD_COL = "load"

BASE_FEATURES = [
    "cpu_usage","cpu_temp","gpu_temp","power","cpu_freq"
]

WINDOW = 20

# =========================
# LOAD
# =========================
df = pd.read_csv(DATA_FILE).dropna().sort_values("timestamp")

# =========================
# REGRESSOR (DIGITAL TWIN)
# =========================
X = df[BASE_FEATURES]
y = df[TARGET]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

reg = RandomForestRegressor(n_estimators=80, max_depth=12)
reg.fit(X_tr, y_tr)

df["predicted_fan"] = reg.predict(X)

# =========================
# MISMATCH + TEMPORAL
# =========================
df["fan_error"] = (df[TARGET] - df["predicted_fan"]).clip(-30, 30)
df["fan_error_pct"] = df["fan_error"] / (df["predicted_fan"] + 1e-6)

df["fan_trend"] = df[TARGET].rolling(WINDOW).mean()
df["temp_trend"] = df["cpu_temp"].rolling(WINDOW).mean()

df["fan_slope"] = df[TARGET].diff()
df["temp_slope"] = df["cpu_temp"].diff()

df["fan_var"] = df[TARGET].rolling(WINDOW).std()
df["temp_var"] = df["cpu_temp"].rolling(WINDOW).std()

df = df.bfill()

# =========================
# Z-SCORE PER LOAD
# =========================
df["residual"] = df["fan_error"]

stats = df.groupby(LOAD_COL)["residual"].agg(["mean","std"]).reset_index()
stats.columns = [LOAD_COL, "mu", "sigma"]
stats["sigma"] = stats["sigma"].replace(0, 1e-6)
stats["sigma"] = stats["sigma"].clip(lower=2.0)
df = df.merge(stats, on=LOAD_COL)

df["z"] = (df["residual"] - df["mu"]) / df["sigma"]
df["z_abs"] = df["z"].abs()

# =========================
# SCORE CALIBRATION (robust)
# =========================
# Use percentiles instead of fixed thresholds
p90 = df["z_abs"].quantile(0.90)
p99 = df["z_abs"].quantile(0.99)

calibration = {
    "p90": float(p90),
    "p99": float(p99)
}

# =========================
# SAVE
# =========================
joblib.dump(reg, "fan_regressor_v4.pkl")
joblib.dump(stats, "z_stats_v4.pkl")
joblib.dump(calibration, "calibration_v4.pkl")

print("✅ Saved regressor + stats + calibration")