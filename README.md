# 🛡️ DriftGuard

> Production-style ML monitoring system for detecting data drift, prediction behavior changes, and delayed performance decay — built for credit risk models.

---

## What problem does it solve?

Once a model is deployed, the world keeps changing. Customer behavior shifts, data pipelines evolve, and your model silently degrades. DriftGuard watches your model every day and raises the alarm before bad predictions cause real damage.

---

## How it works

DriftGuard monitors three failure modes that affect deployed models:

**1. Feature Drift** — Are today's inputs still similar to what the model was trained on?
Uses PSI (Population Stability Index) and the KS test on every feature, plus multi-day trend analysis to distinguish real drift from one-off noise.

**2. Prediction Drift** — Is the model's behavior changing, even before labels arrive?
Tracks the entropy of predicted probabilities. If the model starts predicting everything as 50/50 instead of confidently 0 or 1, that's a signal — no ground truth needed.

**3. Performance Decay** — Is the model actually getting things wrong more often?
Tracks rolling AUC once labels arrive. Handles the real-world problem of delayed labels gracefully — if labels for a given day aren't available yet, that day's check is skipped rather than crashing.

**4. Smart Alerting** — Not every drift is worth waking someone up for.
Computes a Risk Score per feature (`PSI × feature importance`) and only fires an alert if a high-importance feature has been drifting for 3+ consecutive days. This prevents alert fatigue — the most common reason monitoring systems get ignored.

---

## Streamlit Dashboard

A visual monitoring dashboard with 5 tabs:

| Tab                 | What it shows                                                      |
| ------------------- | ------------------------------------------------------------------ |
| 📊 Overview         | KPI cards, PSI bar chart, feature summary table                    |
| 🔍 Feature Drift    | Distribution overlays, PSI trends, stats comparison, heatmap       |
| 🎯 Prediction Drift | Probability distributions, entropy trend over time                 |
| 📉 Performance      | Rolling AUC, predicted probability by true class                   |
| 🚨 Smart Alerts     | Risk-ranked features, daily summary banner, sustained drift trends |

---

## Quick Start

**1. Clone and enter the project:**

```bash
git clone https://github.com/lazerbeam47/DriftGuard.git
cd DriftGuard
```

**2. Create and activate a Python 3.11 virtual environment:**

```bash
/usr/local/opt/python@3.11/bin/python3.11 -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4. Train the model:**

Open `main.ipynb` and run all cells. This produces:

- `data/reference.csv` — baseline feature distribution
- `data/reference_target.csv` — baseline labels
- `model.pkl` — trained Logistic Regression model

**5. Simulate production data (optional):**

```bash
python3.11 src/data/simulate_production.py
```

**6. Run the CLI monitor:**

```bash
python3.11 -m src.monitor
```

**7. Run the dashboard:**

```bash
pip install streamlit plotly scipy
streamlit run dashboard.py
```

---

## Project Structure

```
DriftGuard/
├── data/
│   ├── raw.csv                        # source dataset (credit card defaults)
│   ├── reference.csv                  # training distribution baseline
│   ├── reference_target.csv           # baseline labels
│   └── production/
│       ├── day_01.csv                 # daily production feature batches
│       ├── day_01_labels.csv          # ground truth labels (may arrive late)
│       └── ...
├── src/
│   ├── monitor.py                     # orchestrator — ties everything together
│   ├── data/
│   │   └── simulate_production.py     # generates synthetic production batches
│   ├── drift/
│   │   ├── psi.py                     # Population Stability Index
│   │   ├── ks.py                      # Kolmogorov-Smirnov test
│   │   ├── trend.py                   # multi-day drift trend analysis
│   │   └── prediction_drift.py        # entropy-based output monitoring
│   └── performance/
│       └── decay.py                   # rolling AUC with late-label handling
├── dashboard.py                       # Streamlit monitoring dashboard
├── main.ipynb                         # training notebook
├── requirements.txt
└── README.md
```

---

## Design Decisions

**Why Logistic Regression?**
Credit default prediction is a regulated domain. Lenders are legally required to explain why a loan was denied. Logistic regression's coefficients directly map to feature importance, making both the model and the monitoring transparent and auditable. This aligns with real-world credit scoring systems (CIBIL, FICO).

**Why PSI + KS together?**
PSI is binning-based and catches gradual distribution shifts. KS is rank-based and catches changes in distribution shape. Using both together reduces false negatives — they catch different kinds of drift.

**Why not auto-retrain?**
Conservative by design. The system alerts humans and logs everything, but never triggers automatic retraining. In production, retraining decisions should involve human review of root cause — drift in features could mean bad data, upstream pipeline failure, or genuine population shift, each requiring a different response.

**Why the 3-day consecutive rule?**
A single spike could be sampling noise, a data pipeline hiccup, or a one-off event. Requiring 3 consecutive days of elevated drift before alerting drastically reduces false positives and keeps alert channels useful.

---

## Configuration (config.yaml)

Both the dashboard and `src/monitor.py` read monitoring thresholds and paths from a shared `config.yaml`. This lets you tune alert thresholds and point the monitor at different data/model locations without changing code.

Example `config.yaml`:

```yaml
# config.yaml (example)
monitoring:
  psi_critical: 0.20
  psi_warning: 0.10
  ks_critical: 0.10
  consecutive_days: 3
  risk_score_threshold: 0.05

# optional paths (overrides defaults in monitor.py)
data:
  reference_path: "data/reference.csv"
  reference_target_path: "data/reference_target.csv"
  production_dir: "data/production"
  model_path: "model.pkl"
  target_column: "target"
```

After editing `config.yaml` in the dashboard UI you can click Save; then restart the monitor to pick up the new values.

## Slack integration (alerts)

The monitor supports sending Slack alerts when sustained, high-risk feature drift is detected. To enable Slack alerts:

1. Create a Slack app or use an Incoming Webhook in your workspace and obtain a webhook URL (or a bot token) — do NOT commit this secret to the repo.

2. Create a local `.env` file in the project root and add your Slack credentials (example):

```env
# .env (DO NOT COMMIT)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX
SLACK_CHANNEL=#drift-alerts
```

3. Ensure `.env` is ignored by Git (the project `.gitignore` includes `.env`).

4. Restart the monitor; it reads environment variables via `python-dotenv` and will call Slack only for features that meet the alert policy.

Security note: If you accidentally commit secrets, rotate them immediately and remove the secrets from Git history before pushing (do not push them to GitHub).

## Where to edit thresholds

- Dashboard: open `dashboard.py` → adjust sliders → Save to write `config.yaml`.
- Monitor: edit `config.yaml` directly or change values in the dashboard and restart the monitor.

## Example workflow with Slack alerts enabled

1. Train and save model (run `main.ipynb`).
2. Generate or place production files in `data/production/` (`day_01.csv`, etc.).
3. Start the monitor:

```bash
python3.11 -m src.monitor
```

If sustained high-risk drift is detected the monitor prints the issue and sends a single Slack message summarizing the affected features. Slack messages are limited to reduce alert fatigue.

## Features (in depth)

- Feature-level drift detection
  - Population Stability Index (PSI) with percentile-based binning to measure magnitude of shift.
  - Kolmogorov–Smirnov (KS) two-sample test for statistical significance.
  - Per-feature trend tracking to require multi-day confirmation before escalation.

- Prediction-behavior monitoring
  - Entropy of predicted probabilities to surface increasing model uncertainty before labels arrive.
  - Per-day and time-series views in the dashboard to inspect model confidence shifts.

- Smart alerting
  - Risk score = PSI × feature importance (from model coefficients) to prioritize actionable drift.
  - Consecutive-day policy to reduce false positives and alert fatigue.
  - Slack integration for concise daily summaries (only for sustained, high-risk features).

- Delayed performance validation
  - Rolling AUC computed once labels arrive, with warm-up safeguards and skip logic for insufficient classes.
  - Performance status flags (OK / degrading / drop) for human review.

- Safety-first orchestration
  - No automatic retraining — monitoring is a decision support tool for humans.
  - Config-driven thresholds (via `config.yaml`) so teams can tune sensitivity.

## Future scope / roadmap

Short-term (low-effort, high-impact)

- Add unit and integration tests (pytest) for drift and performance modules.
- Export metrics (JSON/CSV) and add a Prometheus exporter for long-term tracking.
- Replace print statements with structured logging and add verbosity levels.

Medium-term (improve robustness & UX)

- CLI with subcommands (train, simulate, monitor, dashboard) and config profiles.
- Schema validation for incoming CSVs (good columns, dtypes, nullable rules).
- Improve dashboard with interactive filters, per-segment drift, and historical comparisons.
- Add automated daily job examples (Airflow DAG, cron job wrapper) and Docker packaging.

Long-term (enterprise / production)

- Integrate with a feature store and central metric store (e.g., Feast + Prometheus/Grafana).
- Add fine-grained alert policies (per-team, per-feature, severity routing) and incident playbooks.
- Implement a canary retraining pipeline: shadow training → validation → gated promotion with human approval.
- Add model explainability & root-cause helpers (SHAP + per-feature contribution over time).
- Support streaming detection (Kafka consumer) for near-real-time monitoring.

Research / advanced ideas

- Importance-weighted PSI (weights = business impact) to prioritize features with higher dollar impact.
- Drift attribution: cluster drifted samples and run lightweight explainers to identify upstream causes.
- Auto-tuning thresholds via historical backtesting to calibrate PSI/KS levels to actual performance impact.

## Screenshots

 The current images in the repo are:

- `assets/overview_ss.png` — Dashboard overview (Overview tab)
- `assets/featuredrift_ss.png` — Feature drift deep-dive (distributions + heatmap)
- `assets/performance_ss.png` — Performance / rolling AUC view
- `assets/prediction_ss.png` — Prediction drift (probability distributions + entropy)
- `assets/smartalert_ss.png` — Smart Alerts (risk-ranked features + summary banner)


![Dashboard overview](assets/overview_ss.png)

![Feature drift deep-dive](assets/featuredrift_ss.png)

![Prediction drift](assets/prediction_ss.png)

![Performance rolling AUC](assets/performance_ss.png)

![Smart alerts](assets/smartalert_ss.png)


