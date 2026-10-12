import numpy as np


def regression_metrics(actual, pred) -> dict[str, float]:
    """MAE / RMSE (MW), MAPE (%), R². Inputs must be in original units, not standardized."""
    a, p = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    err = a - p
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err / a)) * 100),
        "r2": float(1 - np.sum(err ** 2) / np.sum((a - a.mean()) ** 2)),
        "n": int(len(a)),
    }
