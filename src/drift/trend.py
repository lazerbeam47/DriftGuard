
import numpy as np


def drift_trend_status(psi_history, min_points=3):
    """
    Detects if drift is trending upward.
    psi_history: list of PSI values ordered by time
    """
    if len(psi_history) < min_points:
        return "INSUFFICIENT DATA"

    # simple trend: last value significantly higher than earlier average
    recent = psi_history[-1]
    past_mean = np.mean(psi_history[:-1])

    if recent > 0.25 and recent > past_mean * 1.5:
        return "🚨 DRIFT INCREASING"
    elif recent > past_mean * 1.2:
        return "⚠️ DRIFT TRENDING"
    else:
        return "STABLE"
