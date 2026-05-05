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

| Tab | What it shows |
|---|---|
| 📊 Overview | KPI cards, PSI bar chart, feature summary table |
| 🔍 Feature Drift | Distribution overlays, PSI trends, stats comparison, heatmap |
| 🎯 Prediction Drift | Probability distributions, entropy trend over time |
| 📉 Performance | Rolling AUC, predicted probability by true class |
| 🚨 Smart Alerts | Risk-ranked features, daily summary banner, sustained drift trends |

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

## Dataset

UCI Credit Card Default dataset — 30,000 records of Taiwan credit card holders with 23 features including payment history, bill amounts, and demographics. Target: whether the customer defaulted next month.

- Training set: 24,000 rows
- Reference (test) set: 6,000 rows
- Production batches: ~1,800 rows per day

---

## Gotchas

- **Use the same Python + scikit-learn version for training and monitoring.** The project pins `scikit-learn==1.8.0`. Pickle/joblib files are version-sensitive — mismatches cause `UnpicklingError`.
- **Label files must match production files row-for-row.** The monitor skips performance evaluation gracefully if labels are missing, but incorrect label files will give misleading AUC scores.
- **Run `streamlit run dashboard.py` from the project root**, not from inside `src/`. The dashboard uses relative paths to find `model.pkl` and `data/`.

---

## What's next

- [ ] Slack / email alerting integration
- [ ] Scheduled runs via cron or APScheduler
- [ ] `config.yaml` for threshold tuning without code changes
- [ ] Chi-square test for categorical feature drift
- [ ] GitHub Actions CI pipeline
- [ ] Unit tests for `src/drift` and `src/performance`
- [ ] Containerization (Docker)

---

## License

MIT