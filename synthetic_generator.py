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

    temp_increase = level * 10

    rpm_efficiency_loss = level * 0.12

    response_lag = level * 0.35

    rpm_noise = level * 4

    thermal_noise = level * 2

    # ---------------------------------------
    # Increase temperatures
    # ---------------------------------------

    degraded["temperature"] = (
        degraded["temperature"]
        + temp_increase
        + np.random.normal(
            0,
            thermal_noise,
            len(df)
        )
    )

    # ---------------------------------------
    # Thermal lag
    # ---------------------------------------

    degraded["temperature"] = apply_thermal_lag(
        degraded["temperature"].values,
        response_lag
    )

    # ---------------------------------------
    # Fan efficiency reduction
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
            0.9 * rpm[i - 1]
            + 0.1 * rpm[i]
        )

    degraded["fan_rpm"] = rpm

    # ---------------------------------------
    # RPM instability
    # ---------------------------------------

    degraded["fan_rpm"] += np.random.normal(
        0,
        rpm_noise,
        len(df)
    )

    # ---------------------------------------
    # Clamp values
    # ---------------------------------------

    degraded["fan_rpm"] = degraded[
        "fan_rpm"
    ].clip(0, 100)

    degraded["temperature"] = degraded[
        "temperature"
    ].clip(0, 100)

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