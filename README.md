# DriftGuard

Production-style ML monitoring to detect data drift, prediction behavior changes, and delayed performance decay.

## Features

- Feature drift: PSI + KS with trend analysis
- Prediction drift: entropy monitoring of probabilities
- Performance decay: rolling AUC with safeguards for late labels
- Orchestrator: `src/monitor.py` aggregates signals and prints actionable statuses

## Project Structure

```
ml-drift-detection/
├─ data/
│  ├─ raw.csv
│  ├─ reference.csv
│  ├─ reference_target.csv
│  └─ production/
│     ├─ day_01.csv
│     └─ day_01_labels.csv (optional; required for performance)
├─ src/
│  ├─ __init__.py
│  ├─ monitor.py
│  ├─ drift/
│  │  ├─ __init__.py
│  │  ├─ psi.py
│  │  ├─ ks.py
│  │  ├─ trend.py
│  │  └─ prediction_drift.py
│  └─ performance/
│     ├─ __init__.py
│     └─ decay.py
└─ main.ipynb
```

## Setup

- Python 3.11 recommended
- scikit-learn 1.8.0 to match the saved model

Install deps:

```
pip3 install -r requirements.txt
```

## Train and Save Model (Notebook `main.ipynb`)

- Trains LogisticRegression
- Saves reference data and `model.pkl`
- Prints sanity info for sklearn/Python

## Run Monitor

From project root:

```
/usr/local/opt/python@3.11/bin/python3.11 -m src.monitor
```

Or:

```
python3.11 src/monitor.py
```

## Example Output

```
=== Monitoring day_03.csv ===
[Feature] LIMIT_BAL PSI=0.549 🚨 DRIFT | KS=0.231 🚨 DRIFT | Trend=🚨 DRIFT INCREASING
[Prediction] Entropy=0.623 → OK
[Performance] Labels not available yet
```

## Notes

- Ensure production label files match their feature files in row count.
- If labels are missing, performance lines will be skipped.
- Pin versions to avoid pickle incompatibilities.

## License

MIT
