# src/monitor.py

import pandas as pd
import os
import joblib
from collections import defaultdict
import numpy as np

from drift.psi import calculate_psi
from drift.ks import calculate_ks, ks_status
from drift.prediction_drift import prediction_entropy, entropy_status
from drift.trend import drift_trend_status
from performance.decay import rolling_auc, performance_status

REFERENCE_PATH = "data/reference.csv"
REFERENCE_TARGET = "data/reference_target.csv"
PRODUCTION_DIR = "data/production"
MODEL_PATH = "model.pkl"


def psi_status(psi):
    if psi > 0.25:
        return "🚨 DRIFT"
    elif psi > 0.1:
        return "⚠️ WARNING"
    else:
        return "OK"


def main():
    reference = pd.read_csv(REFERENCE_PATH)
    ref_y = pd.read_csv(REFERENCE_TARGET).values.ravel().astype(int)
    model = joblib.load(MODEL_PATH)

    psi_history = defaultdict(list)

    # ----- Baseline -----
    ref_preds = model.predict_proba(reference)[:, 1]
    ref_entropy = prediction_entropy(ref_preds)

    ref_aucs = rolling_auc(ref_y, ref_preds)
    baseline_auc = ref_aucs.mean() if len(ref_aucs) > 0 else None

    print(f"\nBaseline entropy: {ref_entropy:.3f}")
    if baseline_auc is not None:
        print(f"Baseline rolling AUC: {baseline_auc:.3f}")
    else:
        print("Baseline rolling AUC: insufficient data")

    all_preds = []
    all_true = []

    for file in sorted(os.listdir(PRODUCTION_DIR)):
        if not file.endswith(".csv") or file.endswith("_labels.csv"):
            continue

        print(f"\n=== Monitoring {file} ===")
        prod = pd.read_csv(f"{PRODUCTION_DIR}/{file}")

        # ---------- FEATURE DRIFT ----------
        for col in reference.columns:
            psi = calculate_psi(reference[col], prod[col])
            psi_history[col].append(psi)

            psi_flag = psi_status(psi)
            ks_stat, p_val = calculate_ks(reference[col], prod[col])
            ks_flag = ks_status(ks_stat, p_val)
            trend_flag = drift_trend_status(psi_history[col])

            if psi_flag != "OK" or ks_flag != "OK":
                print(
                    f"[Feature] {col:15s} "
                    f"PSI={psi:.3f} {psi_flag} | "
                    f"KS={ks_stat:.3f} {ks_flag} | "
                    f"Trend={trend_flag}"
                )

        # ---------- PREDICTION DRIFT ----------
        prod_preds = model.predict_proba(prod)[:, 1]
        prod_entropy = prediction_entropy(prod_preds)
        entropy_flag = entropy_status(prod_entropy, ref_entropy)

        print(f"[Prediction] Entropy={prod_entropy:.3f} → {entropy_flag}")

        # ---------- LOAD LABELS (REQUIRED) ----------
        label_path = f"{PRODUCTION_DIR}/{file.replace('.csv', '_labels.csv')}"

        if not os.path.exists(label_path):
            print("[Performance] Labels not available yet")
            continue

        prod_y = pd.read_csv(label_path).values.ravel().astype(int)

        for p, y in zip(prod_preds, prod_y):
            all_preds.append(p)
            all_true.append(y)  

        # ---------- PERFORMANCE DECAY ----------
        aucs = rolling_auc(
            np.array(all_true),
            np.array(all_preds)
        )
        if all(
            drift_trend_status(psi_history[col]) == "INSUFFICIENT DATA"
            for col in psi_history
        ):
            print("[Performance] Waiting for drift trend confirmation")
            continue
        if len(aucs) == 0 or baseline_auc is None:
            print("[Performance] Not enough stable data yet")
            continue

        recent_auc = aucs[-1]
        perf_flag = performance_status(recent_auc, baseline_auc)

        print(
            f"[Performance] Rolling AUC={recent_auc:.3f} → {perf_flag}"
        )


if __name__ == "__main__":
    main()
