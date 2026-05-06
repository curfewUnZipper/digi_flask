import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


def estimate_rul(residuals):

    residuals = np.array(residuals)

    # ---------------------------------------
    # Smooth degradation trend
    # ---------------------------------------

    smooth = pd.Series(residuals)\
        .rolling(100)\
        .mean()\
        .dropna()

    if len(smooth) < 20:
        return {
            "estimated_rul": -1,
            "slope": 0,
            "critical_threshold": 0
        }

    y = smooth.values

    x = np.arange(len(y))

    # ---------------------------------------
    # Fit trend
    # ---------------------------------------

    model = LinearRegression()

    model.fit(
        x.reshape(-1, 1),
        y
    )

    slope = model.coef_[0]
    intercept = model.intercept_

    # ---------------------------------------
    # Critical threshold
    # ---------------------------------------

    critical = y.mean() + 2 * y.std()

    # ---------------------------------------
    # Avoid unstable division
    # ---------------------------------------

    if slope <= 1e-5:

        return {
            "estimated_rul": 999999,
            "slope": slope,
            "critical_threshold": critical
        }

    # ---------------------------------------
    # Predict threshold crossing
    # ---------------------------------------

    crossing = (critical - intercept) / slope

    rul = crossing - len(y)

    rul = max(rul, 0)

    return {
        "estimated_rul": rul,
        "slope": slope,
        "critical_threshold": critical
    }