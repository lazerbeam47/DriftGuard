"""
DriftGuard Dashboard — dashboard.py
Place this file in the project root (same level as main.ipynb).
Run with: streamlit run dashboard.py
"""

import os
import glob
import importlib.util
import sys
import pickle
import joblib

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import yaml

# ─────────────────────────────────────────────
# Load config.yaml
# This is the single source of truth for thresholds.
# Both dashboard and monitor.py read from here.
# ─────────────────────────────────────────────
CONFIG_PATH = "config.yaml"

def load_config():
    """Read config.yaml and return the full config dict."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    # Fallback defaults if config.yaml doesn't exist
    return {
        "data": {
            "reference_path": "data/reference.csv",
            "reference_target_path": "data/reference_target.csv",
            "production_dir": "data/production",
            "model_path": "model.pkl",
            "target_column": "target",
        },
        "monitoring": {
            "psi_critical": 0.2,
            "psi_warning": 0.1,
            "ks_critical": 0.1,
            "consecutive_days": 3,
            "risk_score_threshold": 0.05,
        }
    }

def get_data_config():
    """Shortcut to get the data section of config."""
    return load_config().get("data", {})

def save_config(psi_critical, psi_warning, ks_critical, consecutive_days, risk_score_threshold):
    """
    Write updated thresholds back to config.yaml.
    Preserves the data section so user paths are never overwritten
    when saving monitoring thresholds.
    """
    # Load existing config first so we don't lose the data section
    existing = load_config()

    config = {
        # Keep existing data paths exactly as they are
        "data": existing.get("data", {}),
        # Update only the monitoring thresholds
        "monitoring": {
            "psi_critical": round(float(psi_critical), 2),
            "psi_warning": round(float(psi_warning), 2),
            "ks_critical": round(float(ks_critical), 2),
            "consecutive_days": int(consecutive_days),
            "risk_score_threshold": round(float(risk_score_threshold), 2),
        }
    }
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DriftGuard",
    page_icon="🛡️",
    layout="wide",
)

# ─────────────────────────────────────────────
# Helpers — load project src modules dynamically
# ─────────────────────────────────────────────
def load_src_module(module_path: str, module_name: str):
    """Load a module from src/ without requiring an installed package."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

# ─────────────────────────────────────────────
# PSI & KS — inline fallbacks if src not importable
# ─────────────────────────────────────────────
def compute_psi(reference: np.ndarray, production: np.ndarray, bins: int = 10) -> float:
    try:
        psi_mod = load_src_module("src/drift/psi.py", "psi")
        return psi_mod.compute_psi(reference, production)
    except Exception:
        pass
    # fallback
    breakpoints = np.linspace(0, 100, bins + 1)
    ref_perc = np.percentile(reference, breakpoints)
    ref_perc = np.unique(ref_perc)
    ref_counts, _ = np.histogram(reference, bins=ref_perc)
    prod_counts, _ = np.histogram(production, bins=ref_perc)
    ref_perc_dist = ref_counts / len(reference) + 1e-8
    prod_perc_dist = prod_counts / len(production) + 1e-8
    psi = np.sum((prod_perc_dist - ref_perc_dist) * np.log(prod_perc_dist / ref_perc_dist))
    return float(psi)

def compute_ks(reference: np.ndarray, production: np.ndarray) -> float:
    try:
        ks_mod = load_src_module("src/drift/ks.py", "ks")
        return ks_mod.compute_ks(reference, production)
    except Exception:
        pass
    stat, _ = stats.ks_2samp(reference, production)
    return float(stat)

def compute_entropy(probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-8, 1 - 1e-8)
    return float(-np.mean(probs * np.log(probs) + (1 - probs) * np.log(1 - probs)))

def rolling_auc(y_true, y_score, window=500):
    from sklearn.metrics import roc_auc_score
    aucs, indices = [], []
    for i in range(window, len(y_true) + 1, window // 2):
        yt = y_true[i - window:i]
        ys = y_score[i - window:i]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
        indices.append(i)
    return indices, aucs

PSI_WARN = 0.1
PSI_CRIT = 0.2
KS_WARN = 0.05
KS_CRIT = 0.1

def psi_status(v):
    if v >= PSI_CRIT: return "🚨 DRIFT"
    if v >= PSI_WARN: return "⚠️ WARNING"
    return "✅ OK"

def ks_status(v):
    if v >= KS_CRIT: return "🚨 DRIFT"
    if v >= KS_WARN: return "⚠️ WARNING"
    return "✅ OK"

# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
@st.cache_data
def load_reference():
    # Read paths from config.yaml — not hardcoded
    dcfg = get_data_config()
    ref_path    = dcfg.get("reference_path", "data/reference.csv")
    target_path = dcfg.get("reference_target_path", "data/reference_target.csv")

    ref = pd.read_csv(ref_path)
    target = None
    if os.path.exists(target_path):
        target = pd.read_csv(target_path)
    return ref, target

@st.cache_data
def load_production_files():
    # Read production dir from config.yaml
    dcfg = get_data_config()
    prod_dir = dcfg.get("production_dir", "data/production")

    # Find all CSVs in production dir, exclude label files
    files = sorted(glob.glob(f"{prod_dir}/*.csv"))
    files = [f for f in files if "label" not in f]
    return files

@st.cache_data
def load_model():
    # Read model path from config.yaml
    dcfg = get_data_config()
    model_path = dcfg.get("model_path", "model.pkl")

    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            return None
    return None

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.image("https://img.shields.io/badge/DriftGuard-ML%20Monitoring-blue?style=for-the-badge", width=280)
st.sidebar.title("🛡️ DriftGuard")
st.sidebar.markdown("Production ML Monitoring Dashboard")
st.sidebar.divider()

# Check data availability
ref_ok = os.path.exists(get_data_config().get("reference_path", "data/reference.csv"))
prod_files = load_production_files() if ref_ok else []

if not ref_ok:
    st.error(f"❌ Reference data not found at `{get_data_config().get('reference_path', 'data/reference.csv')}`. Please run `main.ipynb` first.")
    st.stop()

reference_df, reference_target = load_reference()
model = load_model()

numeric_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()

st.sidebar.markdown(f"**Reference rows:** {len(reference_df):,}")
st.sidebar.markdown(f"**Features:** {len(numeric_cols)}")
st.sidebar.markdown(f"**Production days found:** {len(prod_files)}")

selected_day = None
if prod_files:
    day_labels = [os.path.basename(f) for f in prod_files]
    selected_label = st.sidebar.selectbox("📅 Select production day", day_labels)
    selected_day = prod_files[day_labels.index(selected_label)]

st.sidebar.divider()

# ── Load current config values as slider defaults ──
# Sliders always start at whatever is saved in config.yaml
_cfg = load_config()["monitoring"]

st.sidebar.markdown("### ⚙️ Thresholds")

psi_thresh = st.sidebar.slider(
    "PSI critical threshold",
    0.05, 0.5,
    float(_cfg["psi_critical"]),
    0.05
)

ks_thresh = st.sidebar.slider(
    "KS critical threshold",
    0.01, 0.3,
    float(_cfg["ks_critical"]),
    0.01
)

# Consecutive days slider — moved here so all three sliders are together
_cd_default = min(
    int(_cfg["consecutive_days"]),
    len(prod_files) if prod_files else 7
)
consecutive_days = st.sidebar.slider(
    "Consecutive days before alert",
    1, 7,
    value=_cd_default,
    step=1,
    key="consecutive_days"  # saved to session_state so Save button can read it
)

# ── Save Settings button — after all 3 sliders ──
# Writes current slider values to config.yaml
# Next time monitor.py runs, it picks up the new values
st.sidebar.divider()
st.sidebar.markdown("### 💾 Save Settings")
st.sidebar.caption("Saves thresholds to config.yaml. Restart monitor to apply.")

if st.sidebar.button("💾 Save to config.yaml"):
    save_config(
        psi_critical=psi_thresh,
        psi_warning=round(psi_thresh * 0.5, 2),
        ks_critical=ks_thresh,
        consecutive_days=consecutive_days,
        risk_score_threshold=0.05
    )
    st.sidebar.success("✅ Saved! Restart monitor to apply.")



# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("🛡️ DriftGuard — ML Monitoring Dashboard")
st.caption("Detect data drift, prediction behavior changes, and performance decay.")

if not prod_files:
    _prod_dir = get_data_config().get("production_dir", "data/production")
    st.warning(f"No production files found in `{_prod_dir}/`. Add `day_01.csv`, `day_02.csv`, etc.")
    st.stop()

# ─────────────────────────────────────────────
# Load selected production day
# ─────────────────────────────────────────────
prod_df = pd.read_csv(selected_day)

# Check for label file
label_file = selected_day.replace(".csv", "_labels.csv")
has_labels = os.path.exists(label_file)
labels_df = pd.read_csv(label_file) if has_labels else None

# ─────────────────────────────────────────────
# Compute metrics for ALL days (for trend charts)
# ─────────────────────────────────────────────
@st.cache_data
def compute_all_days(prod_files, psi_thresh, ks_thresh):
    records = []
    for f in prod_files:
        day_label = os.path.basename(f).replace(".csv", "")
        df = pd.read_csv(f)
        ref = pd.read_csv(get_data_config().get("reference_path", "data/reference.csv"))
        common = [c for c in ref.select_dtypes(include=[np.number]).columns if c in df.columns]
        day_rec = {"day": day_label}
        for col in common:
            psi = compute_psi(ref[col].dropna().values, df[col].dropna().values)
            ks = compute_ks(ref[col].dropna().values, df[col].dropna().values)
            day_rec[f"{col}_psi"] = psi
            day_rec[f"{col}_ks"] = ks
        # prediction drift if model available
        m = load_model()
        if m is not None:
            try:
                probs = m.predict_proba(df[common])[:, 1]
                day_rec["entropy"] = compute_entropy(probs)
            except Exception:
                day_rec["entropy"] = None
        records.append(day_rec)
    return pd.DataFrame(records)

all_days_df = compute_all_days(tuple(prod_files), psi_thresh, ks_thresh)

# ─────────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# SMART ALERTING — import shared logic
# compute_risk_scores lives in src/alerts/smart_alert.py
# Both this dashboard and monitor.py import from there,
# so the alert logic is IDENTICAL in both places.
# ─────────────────────────────────────────────
import sys as _sys
_sys.path.insert(0, "src")
try:
    from alerts.smart_alert import compute_risk_scores as _compute_risk_scores
    _smart_alert_available = True
except Exception:
    _smart_alert_available = False


def all_days_df_to_psi_history(all_days_df, feature_names):
    """
    Converts all_days_df (dashboard format) into psi_history (monitor format).

    all_days_df looks like:
        day    | LIMIT_BAL_psi | PAY_0_psi | AGE_psi | ...
        day_01 | 0.05          | 0.12      | 0.03    | ...
        day_02 | 0.08          | 0.25      | 0.04    | ...

    psi_history looks like:
        {"LIMIT_BAL": [0.05, 0.08], "PAY_0": [0.12, 0.25], ...}

    The shared compute_risk_scores() expects psi_history format.
    This bridges the two so the dashboard can use the same function.
    """
    psi_history = {}
    for feature in feature_names:
        psi_col = f"{feature}_psi"
        if psi_col in all_days_df.columns:
            # .tolist() converts the pandas column into a plain Python list
            psi_history[feature] = all_days_df[psi_col].tolist()
    return psi_history


def compute_risk_scores(model, feature_names, all_days_df, psi_thresh, consecutive_days=3):
    """
    Wrapper that converts all_days_df to psi_history format,
    then calls the shared compute_risk_scores from smart_alert.py.
    Returns a DataFrame (same format as before) for the dashboard UI.
    """
    if not _smart_alert_available:
        return pd.DataFrame()

    # Convert dashboard format → shared format
    psi_history = all_days_df_to_psi_history(all_days_df, feature_names)

    # Call the shared function (same one monitor.py uses)
    results = _compute_risk_scores(
        model=model,
        feature_names=feature_names,
        psi_history=psi_history,
        psi_critical=psi_thresh,
        consecutive_days=consecutive_days,
    )

    # Convert list of dicts → DataFrame with capitalized column names for display
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame([{
        "Feature":      r["feature"],
        "Importance":   r["importance"],
        "Latest PSI":   r["latest_psi"],
        "Risk Score":   r["risk_score"],
        "Sustained":    r["sustained"],
        "Should Alert": r["should_alert"],
    } for r in results])

    return df


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🔍 Feature Drift", "🎯 Prediction Drift", "📉 Performance", "🚨 Smart Alerts"
])

# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    st.subheader(f"Monitoring: `{os.path.basename(selected_day)}`")

    common_cols = [c for c in numeric_cols if c in prod_df.columns]

    # Compute per-feature metrics for selected day
    feature_results = []
    for col in common_cols:
        psi = compute_psi(reference_df[col].dropna().values, prod_df[col].dropna().values)
        ks = compute_ks(reference_df[col].dropna().values, prod_df[col].dropna().values)
        feature_results.append({
            "Feature": col,
            "PSI": round(psi, 4),
            "PSI Status": psi_status(psi),
            "KS": round(ks, 4),
            "KS Status": ks_status(ks),
        })
    results_df = pd.DataFrame(feature_results)

    # KPI cards
    n_drift = ((results_df["PSI"] >= psi_thresh) | (results_df["KS"] >= ks_thresh)).sum()
    n_warn  = (((results_df["PSI"] >= PSI_WARN) & (results_df["PSI"] < psi_thresh)) |
               ((results_df["KS"] >= KS_WARN) & (results_df["KS"] < ks_thresh))).sum()
    n_ok    = len(results_df) - n_drift - n_warn

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Features", len(results_df))
    c2.metric("🚨 Drifting", int(n_drift), delta=None)
    c3.metric("⚠️ Warning", int(n_warn))
    c4.metric("✅ Healthy", int(n_ok))

    st.divider()

    # Overall status banner
    if n_drift > 0:
        st.error(f"🚨 **{n_drift} feature(s) have significant drift!** Investigate immediately.")
    elif n_warn > 0:
        st.warning(f"⚠️ **{n_warn} feature(s) showing early warning signs.** Monitor closely.")
    else:
        st.success("✅ All features are within acceptable bounds.")

    st.divider()

    # PSI bar chart for all features
    fig = go.Figure()
    colors = []
    for _, row in results_df.iterrows():
        if row["PSI"] >= psi_thresh:
            colors.append("#ef4444")
        elif row["PSI"] >= PSI_WARN:
            colors.append("#f59e0b")
        else:
            colors.append("#22c55e")

    fig.add_trace(go.Bar(
        x=results_df["Feature"],
        y=results_df["PSI"],
        marker_color=colors,
        name="PSI"
    ))
    fig.add_hline(y=PSI_WARN, line_dash="dot", line_color="#f59e0b", annotation_text="Warning")
    fig.add_hline(y=psi_thresh, line_dash="dash", line_color="#ef4444", annotation_text="Critical")
    fig.update_layout(
        title="PSI by Feature",
        xaxis_title="Feature",
        yaxis_title="PSI Score",
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")

    # Summary table
    st.subheader("Feature Summary Table")
    def highlight_status(val):
        if "DRIFT" in str(val): return "background-color: #fecaca; color: #991b1b"
        if "WARNING" in str(val): return "background-color: #fef3c7; color: #92400e"
        return "background-color: #dcfce7; color: #166534"

    styled = results_df.style.map(highlight_status, subset=["PSI Status", "KS Status"])
    st.dataframe(styled, width="stretch", height=300)

# ══════════════════════════════════════════════
# TAB 2 — FEATURE DRIFT DEEP DIVE
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Feature Drift — Deep Dive")

    selected_feature = st.selectbox("Select a feature to inspect", common_cols)

    col_left, col_right = st.columns(2)

    with col_left:
        # Distribution overlay
        ref_vals = reference_df[selected_feature].dropna().values
        prod_vals = prod_df[selected_feature].dropna().values

        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=ref_vals, name="Reference", opacity=0.6,
            marker_color="#6366f1", histnorm="probability density"
        ))
        fig2.add_trace(go.Histogram(
            x=prod_vals, name="Production", opacity=0.6,
            marker_color="#f43f5e", histnorm="probability density"
        ))
        fig2.update_layout(
            barmode="overlay",
            title=f"Distribution: {selected_feature}",
            height=350,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, width="stretch")

    with col_right:
        # PSI trend over all days
        psi_col = f"{selected_feature}_psi"
        if psi_col in all_days_df.columns:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=all_days_df["day"], y=all_days_df[psi_col],
                mode="lines+markers", name="PSI",
                line=dict(color="#6366f1", width=2),
                marker=dict(size=7)
            ))
            fig3.add_hline(y=PSI_WARN, line_dash="dot", line_color="#f59e0b")
            fig3.add_hline(y=psi_thresh, line_dash="dash", line_color="#ef4444")
            fig3.update_layout(
                title=f"PSI Trend: {selected_feature}",
                height=350,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig3, width="stretch")

    # Stats comparison
    st.subheader("Descriptive Statistics Comparison")
    ref_stats = reference_df[selected_feature].describe().rename("Reference")
    prod_stats = prod_df[selected_feature].describe().rename("Production")
    stats_df = pd.concat([ref_stats, prod_stats], axis=1)
    stats_df["Delta"] = (stats_df["Production"] - stats_df["Reference"]).round(4)
    st.dataframe(stats_df.style.highlight_max(axis=1, color="#fecaca"), width="stretch")

    # PSI heatmap over all days x all features
    st.subheader("PSI Heatmap — All Features × All Days")
    psi_cols = [c for c in all_days_df.columns if c.endswith("_psi")]
    if psi_cols:
        heatmap_df = all_days_df[["day"] + psi_cols].set_index("day")
        heatmap_df.columns = [c.replace("_psi", "") for c in heatmap_df.columns]
        fig_heat = px.imshow(
            heatmap_df.T,
            color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
            zmin=0, zmax=0.4,
            aspect="auto",
            title="PSI Heatmap (green=OK, yellow=warning, red=drift)"
        )
        fig_heat.update_layout(height=400)
        st.plotly_chart(fig_heat, width="stretch")

# ══════════════════════════════════════════════
# TAB 3 — PREDICTION DRIFT
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Prediction Drift — Model Output Monitoring")

    if model is None:
        st.warning("⚠️ `model.pkl` not found. Cannot compute prediction metrics.")
    else:
        common_cols_pred = [c for c in numeric_cols if c in prod_df.columns and c in reference_df.columns]
        try:
            ref_probs  = model.predict_proba(reference_df[common_cols_pred])[:, 1]
            prod_probs = model.predict_proba(prod_df[common_cols_pred])[:, 1]

            ref_entropy  = compute_entropy(ref_probs)
            prod_entropy = compute_entropy(prod_probs)

            c1, c2, c3 = st.columns(3)
            c1.metric("Reference Entropy", f"{ref_entropy:.4f}")
            c2.metric("Production Entropy", f"{prod_entropy:.4f}",
                      delta=f"{prod_entropy - ref_entropy:+.4f}")
            c3.metric("Status", "🚨 DRIFT" if prod_entropy > 0.65 else "✅ OK")

            st.divider()

            col_a, col_b = st.columns(2)

            with col_a:
                # Probability distribution overlay
                fig_p = go.Figure()
                fig_p.add_trace(go.Histogram(
                    x=ref_probs, name="Reference", opacity=0.6,
                    marker_color="#6366f1", histnorm="probability density", nbinsx=30
                ))
                fig_p.add_trace(go.Histogram(
                    x=prod_probs, name="Production", opacity=0.6,
                    marker_color="#f43f5e", histnorm="probability density", nbinsx=30
                ))
                fig_p.update_layout(
                    barmode="overlay",
                    title="Predicted Probability Distribution",
                    xaxis_title="Predicted Probability",
                    height=350,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_p, width="stretch")

            with col_b:
                # Entropy trend across days
                if "entropy" in all_days_df.columns:
                    fig_e = go.Figure()
                    fig_e.add_trace(go.Scatter(
                        x=all_days_df["day"],
                        y=all_days_df["entropy"],
                        mode="lines+markers",
                        line=dict(color="#f43f5e", width=2),
                        marker=dict(size=7),
                        name="Entropy"
                    ))
                    fig_e.add_hline(y=0.65, line_dash="dash", line_color="#ef4444",
                                    annotation_text="Drift threshold")
                    fig_e.update_layout(
                        title="Prediction Entropy Over Time",
                        height=350,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        xaxis_tickangle=-30,
                    )
                    st.plotly_chart(fig_e, width="stretch")

        except Exception as e:
            st.error(f"Error computing predictions: {e}")

# ══════════════════════════════════════════════
# TAB 4 — PERFORMANCE DECAY
# ══════════════════════════════════════════════
with tab4:
    st.subheader("Performance Decay — Rolling AUC")

    if not has_labels:
        st.info(f"⏳ Labels not available yet for `{os.path.basename(selected_day)}`. "
                "Add a `_labels.csv` file matching the production data to enable performance tracking.")
    else:
        try:
            from sklearn.metrics import roc_auc_score, classification_report

            common_cols_perf = [c for c in numeric_cols if c in prod_df.columns]

            # The labels file has multiple columns — grab "target" specifically
            # Get target column name from config — not hardcoded
            _target_col = get_data_config().get("target_column", "target")
            y_true = labels_df[_target_col].values.astype(int)

            # Labels file might be larger than production file (e.g. full dataset vs one day)
            # Trim labels to match the number of rows in production data
            n = len(prod_df)
            y_true = y_true[:n]

            # Get predicted probabilities for class 1 (default = yes)
            y_score = model.predict_proba(prod_df[common_cols_perf])[:, 1]

            y_pred  = (y_score >= 0.5).astype(int)

            # Binary classification — 0 vs 1
            overall_auc = roc_auc_score(y_true, y_score)

            c1, c2, c3 = st.columns(3)
            c1.metric("Overall AUC", f"{overall_auc:.4f}")
            c2.metric("Samples", len(y_true))
            c3.metric("Status",
                      "🚨 LOW" if overall_auc < 0.65 else
                      "⚠️ WARNING" if overall_auc < 0.75 else "✅ GOOD")

            st.divider()

            col_x, col_y = st.columns(2)

            with col_x:
                # Rolling AUC
                indices, aucs = rolling_auc(y_true, y_score)
                if aucs:
                    fig_auc = go.Figure()
                    fig_auc.add_trace(go.Scatter(
                        x=indices, y=aucs,
                        mode="lines+markers",
                        line=dict(color="#6366f1", width=2),
                        marker=dict(size=6),
                        name="Rolling AUC"
                    ))
                    fig_auc.add_hline(y=0.75, line_dash="dot", line_color="#f59e0b",
                                      annotation_text="Warning")
                    fig_auc.add_hline(y=0.65, line_dash="dash", line_color="#ef4444",
                                      annotation_text="Critical")
                    fig_auc.update_layout(
                        title="Rolling AUC (window=500)",
                        xaxis_title="Sample index",
                        yaxis_title="AUC",
                        height=350,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_auc, width="stretch")

            with col_y:
                # Prediction confidence vs actual
                fig_conf = go.Figure()
                fig_conf.add_trace(go.Box(
                    y=y_score[y_true == 0], name="Actual: 0",
                    marker_color="#6366f1"
                ))
                fig_conf.add_trace(go.Box(
                    y=y_score[y_true == 1], name="Actual: 1",
                    marker_color="#f43f5e"
                ))
                fig_conf.update_layout(
                    title="Predicted Probability by True Class",
                    yaxis_title="Predicted Probability",
                    height=350,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_conf, width="stretch")

        except Exception as e:
            st.error(f"Error computing performance metrics: {e}")

# ══════════════════════════════════════════════
# TAB 5 — SMART ALERTS
# ══════════════════════════════════════════════
with tab5:
    st.subheader("🚨 Smart Alert Center")
    st.caption("Only alerts that actually matter — filtered by feature importance and sustained drift.")

    # We need the model to get feature importances.
    # If model isn't loaded, we can't do smart alerting.
    if model is None:
        st.warning("⚠️ Model not loaded. Smart alerts require `model.pkl`. Re-run `main.ipynb` to generate it.")
    elif len(all_days_df) < 1:
        st.info("Need at least 1 day of production data to compute alerts.")
    else:
        # Get the features that exist in both reference data and all_days_df
        # These are the features we can actually compute PSI for
        common_alert_cols = [
            c for c in numeric_cols          # all numeric columns in reference data
            if f"{c}_psi" in all_days_df.columns  # that also have PSI computed
        ]

        # How many consecutive days of drift before we alert?
        # Putting this in the sidebar so the user can tune it


        # --- Run the smart alerting logic ---
        # This calls the function we defined above
        # Returns a dataframe: one row per feature, sorted by Risk Score
        risk_df = compute_risk_scores(
            model=model,
            feature_names=common_alert_cols,
            all_days_df=all_days_df,
            psi_thresh=psi_thresh,
            consecutive_days=consecutive_days
        )

        # --- Split features into three groups ---

        # Features that SHOULD be alerted on right now
        # Should Alert = True means: high risk score AND sustained drift
        alert_features = risk_df[risk_df["Should Alert"] == True]

        # Features that are drifting but either not important enough
        # or not sustained long enough to alert on
        watch_features = risk_df[
            (risk_df["Should Alert"] == False) &   # not alerting yet
            (risk_df["Latest PSI"] >= PSI_WARN)    # but PSI is elevated (above warning level)
        ]

        # Everything else — features that are healthy, ignore them
        healthy_features = risk_df[
            (risk_df["Should Alert"] == False) &
            (risk_df["Latest PSI"] < PSI_WARN)
        ]

        # ── Daily Summary Banner ──
        # This is what you'd put in a Slack message — one clean summary
        st.markdown("### 📋 Daily Summary")

        # Show one banner based on overall situation
        if len(alert_features) > 0:
            # Build a comma-separated list of features that need action
            # e.g. "PAY_0, LIMIT_BAL"
            alert_names = ", ".join(alert_features["Feature"].tolist())
            st.error(
                f"🚨 **{len(alert_features)} feature(s) require immediate attention:** {alert_names}\n\n"
                f"These features are both high-importance AND have been drifting for "
                f"{consecutive_days}+ consecutive days."
            )
        elif len(watch_features) > 0:
            watch_names = ", ".join(watch_features["Feature"].tolist())
            st.warning(
                f"⚠️ **{len(watch_features)} feature(s) to watch:** {watch_names}\n\n"
                f"Drift detected but not yet sustained or not high enough risk to alert."
            )
        else:
            st.success(
                f"✅ **All clear.** {len(healthy_features)} features monitored. No action needed."
            )

        st.divider()

        # ── KPI cards ──
        c1, c2, c3 = st.columns(3)
        c1.metric("🚨 Alert Now",   len(alert_features))
        c2.metric("⚠️ Watch",       len(watch_features))
        c3.metric("✅ Healthy",     len(healthy_features))

        st.divider()

        # ── Risk Score Bar Chart ──
        # Shows all features ranked by risk score
        # Color: red = alerting, yellow = watching, green = healthy
        st.markdown("### 📊 Feature Risk Score Ranking")
        st.caption("Risk Score = PSI × Feature Importance. Higher = more dangerous.")

        # Build a color list: one color per feature row
        colors = []
        for _, row in risk_df.iterrows():
            if row["Should Alert"]:
                colors.append("#ef4444")   # red — alert
            elif row["Latest PSI"] >= PSI_WARN:
                colors.append("#f59e0b")   # yellow — watch
            else:
                colors.append("#22c55e")   # green — healthy

        fig_risk = go.Figure()
        fig_risk.add_trace(go.Bar(
            # x axis = feature names
            x=risk_df["Feature"],
            # y axis = risk score (PSI × importance)
            y=risk_df["Risk Score"],
            marker_color=colors,
            # Show PSI and importance on hover
            customdata=risk_df[["Latest PSI", "Importance"]].values,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Risk Score: %{y:.4f}<br>"
                "PSI: %{customdata[0]:.4f}<br>"
                "Importance: %{customdata[1]:.4f}<br>"
                "<extra></extra>"
            )
        ))

        # Add a reference line showing the alert threshold
        # Any bar above this line is in the danger zone
        fig_risk.add_hline(
            y=0.05,                          # the threshold we used in Should Alert
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="Alert threshold"
        )

        fig_risk.update_layout(
            title="Feature Risk Scores (PSI × Importance)",
            xaxis_title="Feature",
            yaxis_title="Risk Score",
            height=400,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-45,             # angle the labels so they don't overlap
        )
        st.plotly_chart(fig_risk, width="stretch")

        st.divider()

        # ── Detailed breakdown table ──
        st.markdown("### 🔬 Full Feature Breakdown")
        st.caption("Sorted by Risk Score. Only red rows need action.")

        # Color the Should Alert column for quick scanning
        def highlight_alert(val):
            # If the value is True (should alert), make it red
            if val == True:
                return "background-color: #fecaca; color: #991b1b; font-weight: bold"
            return "background-color: #dcfce7; color: #166534"

        styled_risk = risk_df.style.map(highlight_alert, subset=["Should Alert"])
        st.dataframe(styled_risk, width="stretch", height=400)

        st.divider()

        # ── PSI trend for top 3 riskiest features ──
        # Instead of showing all features, only show the ones worth watching
        st.markdown("### 📈 Trend — Top 3 Riskiest Features")
        st.caption(
            f"PSI over time for your highest-risk features. "
            f"Sustained lines above the red threshold = alert."
        )

        # Get top 3 features by risk score
        # .head(3) = take first 3 rows (already sorted by risk score descending)
        top_features = risk_df.head(3)["Feature"].tolist()

        fig_trend = go.Figure()

        # Plot one line per top feature
        # enumerate gives us (0, "PAY_0"), (1, "LIMIT_BAL"), etc.
        colors_trend = ["#ef4444", "#6366f1", "#f59e0b"]  # red, purple, yellow
        for i, feature in enumerate(top_features):
            psi_col = f"{feature}_psi"
            if psi_col not in all_days_df.columns:
                continue

            fig_trend.add_trace(go.Scatter(
                x=all_days_df["day"],           # x axis = day labels
                y=all_days_df[psi_col],         # y axis = PSI score for this feature
                mode="lines+markers",
                name=feature,                   # shown in legend
                line=dict(color=colors_trend[i], width=2),
                marker=dict(size=7)
            ))

        # Draw the critical threshold line so it's obvious when features cross it
        fig_trend.add_hline(
            y=psi_thresh,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"Critical ({psi_thresh})"
        )
        fig_trend.add_hline(
            y=PSI_WARN,
            line_dash="dot",
            line_color="#f59e0b",
            annotation_text="Warning"
        )

        fig_trend.update_layout(
            title="PSI Trend — Top Risk Features",
            xaxis_title="Day",
            yaxis_title="PSI Score",
            height=380,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-30,
            legend=dict(orientation="h", y=-0.2)   # legend below chart
        )
        st.plotly_chart(fig_trend, width="stretch")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.caption("DriftGuard Dashboard · Built with Streamlit & Plotly")