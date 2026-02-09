# src/performance/decay.py

import numpy as np
from sklearn.metrics import roc_auc_score


def rolling_auc(y_true, y_pred, window=200):
    """
    Robust rolling AUC for binary classification.
    Hard-enforces binary labels {0,1}.
    """

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # 🔒 FORCE binary labels
    y_true = (y_true == 1).astype(int)

    aucs = []

    for i in range(window, len(y_true) + 1):
        y_t = y_true[i - window:i]
        y_p = y_pred[i - window:i]

        # Must contain both classes
        if len(np.unique(y_t)) < 2:
            continue

        auc = roc_auc_score(y_t, y_p)
        aucs.append(auc)

    return np.array(aucs)


def performance_status(current_auc, baseline_auc):
    drop = baseline_auc - current_auc

    if drop > 0.08:
        return "🚨 PERFORMANCE DROP"
    elif drop > 0.03:
        return "⚠️ PERFORMANCE DEGRADING"
    else:
        return "OK"

