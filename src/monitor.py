# src/monitor.py

import pandas as pd
import os
import joblib
from collections import defaultdict
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from drift.psi import calculate_psi
from drift.ks import calculate_ks, ks_status
from drift.prediction_drift import prediction_entropy, entropy_status
from drift.trend import drift_trend_status
from performance.decay import rolling_auc, performance_status
from alerts.slack_alert import send_slack_alert

# Import the shared smart alert brain
# This same module is also used by dashboard.py
# so the alert logic is IDENTICAL in both places
from alerts.smart_alert import compute_risk_scores, build_slack_message

# ── Load thresholds from config.yaml ─────────────────────────────────────────
# config.yaml is the single source of truth shared with the dashboard.
# Change thresholds in the dashboard → click Save → restart monitor.
# ─────────────────────────────────────────────────────────────────────────────
import yaml

_CONFIG_PATH = "config.yaml"

def _load_config():
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH) as f:
            full = yaml.safe_load(f)
            # Merge data + monitoring sections into one flat dict
            # so we can read paths and thresholds the same way
            merged = {}
            merged.update(full.get("data", {}))
            merged.update(full.get("monitoring", {}))
            return merged
    return {}

_cfg = _load_config()

# ── All paths come from config.yaml ──────────────────────────────────────────
# User edits config.yaml to point at their own data and model.
# Nothing is hardcoded here.
# ─────────────────────────────────────────────────────────────────────────────
REFERENCE_PATH   = _cfg.get("reference_path",        "data/reference.csv")
REFERENCE_TARGET = _cfg.get("reference_target_path", "data/reference_target.csv")
PRODUCTION_DIR   = _cfg.get("production_dir",        "data/production")
MODEL_PATH       = _cfg.get("model_path",            "model.pkl")
TARGET_COLUMN    = _cfg.get("target_column",         "target")

# ── Monitoring thresholds ─────────────────────────────────────────────────────
PSI_CRITICAL     = _cfg.get("psi_critical", 0.2)
PSI_WARNING      = _cfg.get("psi_warning", 0.1)
CONSECUTIVE_DAYS = _cfg.get("consecutive_days", 3)

print(f"Config loaded → PSI critical: {PSI_CRITICAL} | PSI warning: {PSI_WARNING} | Consecutive days: {CONSECUTIVE_DAYS}")
# ─────────────────────────────────────────────────────────────────────────────


def psi_status(psi):
    if psi > PSI_CRITICAL:
        return "🚨 DRIFT"
    elif psi > PSI_WARNING:
        return "⚠️ WARNING"
    else:
        return "OK"


def main():
    reference = pd.read_csv(REFERENCE_PATH)
    ref_y     = pd.read_csv(REFERENCE_TARGET).values.ravel().astype(int)
    model     = joblib.load(MODEL_PATH)

    # psi_history tracks PSI for each feature across all days
    # e.g. {"PAY_0": [0.05, 0.12, 0.25], "LIMIT_BAL": [0.03, 0.04, 0.03], ...}
    # This is what the smart alert uses to check consecutive days
    psi_history = defaultdict(list)

    # ----- Baseline -----
    ref_preds    = model.predict_proba(reference)[:, 1]
    ref_entropy  = prediction_entropy(ref_preds)
    ref_aucs     = rolling_auc(ref_y, ref_preds)
    baseline_auc = ref_aucs.mean() if len(ref_aucs) > 0 else None

    print(f"\nBaseline entropy: {ref_entropy:.3f}")
    if baseline_auc is not None:
        print(f"Baseline rolling AUC: {baseline_auc:.3f}")
    else:
        print("Baseline rolling AUC: insufficient data")

    all_preds = []
    all_true  = []

    for file in sorted(os.listdir(PRODUCTION_DIR)):
        if not file.endswith(".csv") or file.endswith("_labels.csv"):
            continue

        print(f"\n=== Monitoring {file} ===")
        prod = pd.read_csv(f"{PRODUCTION_DIR}/{file}")

        # ---------- FEATURE DRIFT ----------
        for col in reference.columns:
            psi = calculate_psi(reference[col], prod[col])

            # Append today's PSI to this feature's history list
            # e.g. after day 3: psi_history["PAY_0"] = [0.05, 0.12, 0.25]
            psi_history[col].append(psi)

            psi_flag  = psi_status(psi)
            ks_stat, p_val = calculate_ks(reference[col], prod[col])
            ks_flag   = ks_status(ks_stat, p_val)
            trend_flag = drift_trend_status(psi_history[col])

            # Only print features that have something worth noting
            if psi_flag != "OK" or ks_flag != "OK":
                print(
                    f"[Feature] {col:15s} "
                    f"PSI={psi:.3f} {psi_flag} | "
                    f"KS={ks_stat:.3f} {ks_flag} | "
                    f"Trend={trend_flag}"
                )

        # ---------- SMART ALERTING ----------
        # Now that we've updated psi_history for this day,
        # run the smart alert logic across all features.
        #
        # compute_risk_scores() looks at:
        # - PSI × feature importance = risk score
        # - whether PSI has been high for CONSECUTIVE_DAYS in a row
        # And returns a ranked list of features with alert decisions.

        feature_names = list(reference.columns)

        risk_results = compute_risk_scores(
            model            = model,
            feature_names    = feature_names,
            psi_history      = psi_history,
            psi_critical     = PSI_CRITICAL,
            consecutive_days = CONSECUTIVE_DAYS,
        )

        # Split into features that should alert vs features to just watch
        alert_features   = [r for r in risk_results if r["should_alert"]]
        warning_features = [
            r for r in risk_results
            if not r["should_alert"] and r["latest_psi"] >= PSI_WARNING
        ]

        # Build one clean Slack message for this batch
        # Returns None if nothing worth reporting
        # Only send Slack when sustained drift confirmed (should_alert = True)
        # Watch-level features logged to terminal only — no Slack spam
        if alert_features:
            slack_msg = build_slack_message(file, alert_features, warning_features=[])
            print(f"\n[Alert] Sending Slack message for {file}...")
            send_slack_alert(slack_msg)
        else:
            if warning_features:
                names = ", ".join(w["feature"] for w in warning_features)
                print(f"[Alert] Watching: {names} — not yet sustained. No Slack sent.")
            else:
                print(f"[Alert] All clear. No Slack sent.")

        # ---------- PREDICTION DRIFT ----------
        prod_preds  = model.predict_proba(prod)[:, 1]
        prod_entropy = prediction_entropy(prod_preds)
        entropy_flag = entropy_status(prod_entropy, ref_entropy)

        print(f"[Prediction] Entropy={prod_entropy:.3f} → {entropy_flag}")

        # ---------- LOAD LABELS (OPTIONAL) ----------
        label_path = f"{PRODUCTION_DIR}/{file.replace('.csv', '_labels.csv')}"

        if not os.path.exists(label_path):
            print("[Performance] Labels not available yet")
            continue

        prod_y = pd.read_csv(label_path)[TARGET_COLUMN].values.astype(int)

        # Only use as many labels as we have production rows
        # (labels file might be larger than production file)
        n = len(prod)
        prod_y = prod_y[:n]

        for p, y in zip(prod_preds, prod_y):
            all_preds.append(p)
            all_true.append(y)

        # ---------- PERFORMANCE DECAY ----------
        aucs = rolling_auc(np.array(all_true), np.array(all_preds))

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
        perf_flag  = performance_status(recent_auc, baseline_auc)

        print(f"[Performance] Rolling AUC={recent_auc:.3f} → {perf_flag}")

    print("\n✅ Monitoring completed.")


if __name__ == "__main__":
    main()