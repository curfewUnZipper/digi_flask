import pandas as pd
import numpy as np
import os

from preprocess import load_csv

INPUT_FILE = "data/raw/normal.csv"
OUTPUT_DIR = "data/synthetic"

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)


def apply_thermal_lag(temp, lag_strength):
    out = []

    prev = temp[0]

    for t in temp:
        new_temp = prev + (t - prev) * (1 - lag_strength)
        out.append(new_temp)
        prev = new_temp

    return np.array(out)


def degrade_dataset(df, level):

    degraded = df.copy()

    # ---------------------------------------
    # Aging parameters
    # ---------------------------------------

    temp_increase = level * 8
    rpm_efficiency_loss = level * 0.15
    response_lag = level * 0.4
    noise_strength = level * 200

    # ---------------------------------------
    # Increase temperatures
    # ---------------------------------------

    degraded["temperature"] = (
        degraded["temperature"]
        + temp_increase
        + np.random.normal(0, level * 1.5, len(df))
    )

    # ---------------------------------------
    # Simulate thermal lag
    # ---------------------------------------

    degraded["temperature"] = apply_thermal_lag(
        degraded["temperature"].values,
        response_lag
    )

    # ---------------------------------------
    # Fan loses efficiency
    # ---------------------------------------

    degraded["fan_rpm"] = (
        degraded["fan_rpm"]
        * (1 - rpm_efficiency_loss)
    )

    # ---------------------------------------
    # Delayed fan response
    # ---------------------------------------

    rpm = degraded["fan_rpm"].values.copy()

    for i in range(1, len(rpm)):
        rpm[i] = (
            0.85 * rpm[i - 1]
            + 0.15 * rpm[i]
        )

    degraded["fan_rpm"] = rpm

    # ---------------------------------------
    # Add instability/noise
    # ---------------------------------------

    degraded["fan_rpm"] += np.random.normal(
        0,
        noise_strength,
        len(df)
    )

    degraded["temperature"] += np.random.normal(
        0,
        level,
        len(df)
    )

    return degraded


def main():

    df = load_csv(INPUT_FILE)

    stages = {
        "month_1": 0.05,
        "month_3": 0.10,
        "month_6": 0.20,
        "year_1": 0.35,
        "year_2": 0.55,
        "year_3": 0.75,
    }

    for name, level in stages.items():

        out = degrade_dataset(df, level)

        out_path = os.path.join(
            OUTPUT_DIR,
            f"{name}.csv"
        )

        out.to_csv(out_path, index=False)

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()