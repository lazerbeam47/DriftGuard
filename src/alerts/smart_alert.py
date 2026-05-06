# src/alerts/smart_alert.py
#
# This is the SHARED brain for smart alerting.
# Both monitor.py and dashboard.py call this file.
# This ensures the same logic decides what's worth alerting —
# whether you're running from terminal or looking at the dashboard.

import numpy as np


def get_feature_importance(model, feature_names):
    """
    Extract normalized feature importance from the trained model.

    Handles both plain LogisticRegression and Pipeline (scaler + classifier).
    We take abs() because direction doesn't matter — magnitude does.
    We normalize so all importances add up to 1, making them comparable.

    Returns a dict: {"PAY_0": 0.57, "LIMIT_BAL": 0.40, "AGE": 0.02, ...}
    """

    # Check if the model is a Pipeline
    # e.g. Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression())])
    # If so, go inside it to get to the actual LogisticRegression coefficients
    if hasattr(model, "named_steps"):
        coef = model.named_steps["classifier"].coef_[0]
    else:
        # Plain model — access coef_ directly
        coef = model.coef_[0]

    raw = np.abs(coef)

    # Normalize: divide each weight by the total so they sum to 1
    # e.g. [0.85, 0.60, 0.03] / 1.48 = [0.57, 0.40, 0.02]
    total = raw.sum()
    normalized = raw / total if total > 0 else raw

    # Zip feature names with their normalized importance
    # Returns a plain dictionary for easy lookup
    return dict(zip(feature_names, normalized))


def compute_risk_scores(model, feature_names, psi_history, psi_critical, consecutive_days=3):
    """
    Compute a Risk Score for every feature and decide whether to alert.

    Risk Score = PSI (how much it drifted) × Importance (how much model relies on it)

    A feature only triggers an alert if:
    1. Its risk score is above the threshold (high PSI + high importance)
    2. It has been above the PSI critical threshold for N consecutive days (not a one-off spike)

    Arguments:
        model           — trained sklearn model (needs .coef_)
        feature_names   — list of feature column names
        psi_history     — dict of {feature: [psi_day1, psi_day2, ...]}
                          tracks PSI over time for each feature
        psi_critical    — the PSI threshold above which drift is considered critical
        consecutive_days — how many days in a row PSI must be high before alerting

    Returns a list of dicts, one per feature, sorted by risk score descending.
    """

    # Step 1: get importance for every feature from the model
    importance_map = get_feature_importance(model, feature_names)

    results = []

    for feature in feature_names:

        # Get this feature's PSI history (list of daily PSI values)
        # e.g. [0.05, 0.12, 0.25, 0.31, 0.28]
        history = psi_history.get(feature, [])

        # If we have no history yet, skip this feature
        if not history:
            continue

        # Latest PSI = most recent day's value (last item in the list)
        latest_psi = history[-1]

        # How much does the model rely on this feature?
        importance = importance_map.get(feature, 0)

        # Risk Score: combines drift severity with model reliance
        # High PSI + high importance = dangerous
        # High PSI + low importance = probably fine
        risk_score = latest_psi * importance

        # Step 2: Check if drift has been SUSTAINED for N consecutive days
        # We only look at the last N days of history
        recent = history[-consecutive_days:]

        # sustained = True only if ALL of those days are above the critical threshold
        # e.g. [0.25, 0.31, 0.28] → all > 0.2 → sustained = True
        # e.g. [0.05, 0.31, 0.28] → not all > 0.2 → sustained = False (was fine on day 1)
        sustained = (
            len(recent) >= consecutive_days and
            all(p >= psi_critical for p in recent)
        )

        # Step 3: Should we actually alert?
        # Only if BOTH conditions are true:
        # - risk score is meaningful (not just a low-importance feature drifting)
        # - drift has been sustained (not a one-off spike)
        should_alert = sustained and risk_score > 0.05

        results.append({
            "feature":       feature,
            "importance":    round(importance, 4),
            "latest_psi":    round(latest_psi, 4),
            "risk_score":    round(risk_score, 4),
            "sustained":     sustained,
            "should_alert":  should_alert,
            "history":       history,   # full history for trend charts in dashboard
        })

    # Sort by risk score descending — most dangerous features first
    results.sort(key=lambda x: x["risk_score"], reverse=True)

    return results


def build_slack_message(file_name, alert_features, warning_features):
    """
    Build a clean, single Slack message summarizing the day's drift situation.

    Instead of sending one message per feature (alert fatigue),
    we send ONE message per day with a clear summary.

    Arguments:
        file_name        — name of the production batch file (e.g. "day_03.csv")
        alert_features   — list of feature dicts that should_alert == True
        warning_features — list of feature dicts that are elevated but not yet alerting

    Returns a formatted Slack message string, or None if nothing to report.
    """

    # If nothing is alerting and nothing is in warning, don't send anything
    # Silence is good — it means the model is healthy
    if not alert_features and not warning_features:
        return None

    lines = [f"🛡️ *DriftGuard Daily Report* — `{file_name}`", ""]

    # --- Critical alerts section ---
    if alert_features:
        lines.append(f"🚨 *{len(alert_features)} feature(s) require immediate attention:*")
        for f in alert_features:
            lines.append(
                f"  • *{f['feature']}* — "
                f"PSI={f['latest_psi']:.3f} | "
                f"Importance={f['importance']:.3f} | "
                f"Risk Score={f['risk_score']:.3f}"
            )
        lines.append("")

    # --- Warning section ---
    if warning_features:
        lines.append(f"⚠️ *{len(warning_features)} feature(s) to watch:*")
        for f in warning_features:
            lines.append(
                f"  • *{f['feature']}* — "
                f"PSI={f['latest_psi']:.3f} | "
                f"Risk Score={f['risk_score']:.3f} (not yet sustained)"
            )
        lines.append("")

    lines.append("→ Check the DriftGuard dashboard for full details.")

    # Join all lines into a single message string
    return "\n".join(lines)