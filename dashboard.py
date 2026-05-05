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
    ref = pd.read_csv("data/reference.csv")
    target = None
    if os.path.exists("data/reference_target.csv"):
        target = pd.read_csv("data/reference_target.csv")
    return ref, target

@st.cache_data
def load_production_files():
    files = sorted(glob.glob("data/production/day_*.csv"))
    # exclude label files
    files = [f for f in files if "label" not in f]
    return files

@st.cache_data
def load_model():
    if os.path.exists("model.pkl"):
        try:
            return joblib.load("model.pkl")
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
ref_ok = os.path.exists("data/reference.csv")
prod_files = load_production_files() if ref_ok else []

if not ref_ok:
    st.error("❌ `data/reference.csv` not found. Please run `main.ipynb` first to generate reference data.")
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
psi_thresh = st.sidebar.slider("PSI critical threshold", 0.05, 0.5, PSI_CRIT, 0.05)
ks_thresh = st.sidebar.slider("KS critical threshold", 0.01, 0.3, KS_CRIT, 0.01)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("🛡️ DriftGuard — ML Monitoring Dashboard")
st.caption("Detect data drift, prediction behavior changes, and performance decay.")

if not prod_files:
    st.warning("No production files found in `data/production/`. Add `day_01.csv`, `day_02.csv`, etc.")
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
        ref = pd.read_csv("data/reference.csv")
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
# SMART ALERTING LOGIC
# This is the core anti-alert-fatigue system.
# Instead of screaming about every feature that moves,
# we only care about features the model actually relies on.
# ─────────────────────────────────────────────

def compute_risk_scores(model, feature_names, all_days_df, psi_thresh, consecutive_days=3):
    """
    Computes a Risk Score per feature = PSI × Feature Importance.
    Then checks if that risk score has been high for N consecutive days.
    Only features that are BOTH important AND consistently drifting get flagged.

    Returns a dataframe with one row per feature, sorted by risk score.
    """

    # --- Step 1: Get feature importance from the model ---
    # model.coef_ gives us the weights the logistic regression learned.
    # A higher weight (positive or negative) means the model leans on that feature more.
    # We take abs() because direction doesn't matter — we care about magnitude.
    # Result: a numpy array like [0.3, 0.8, 0.05, ...] — one number per feature
    raw_importance = np.abs(model.coef_[0])

    # Normalize so importances add up to 1 (makes them easier to compare)
    # e.g. [0.3, 0.8, 0.05] → [0.25, 0.67, 0.04]
    total = raw_importance.sum()
    normalized_importance = raw_importance / total if total > 0 else raw_importance

    # Build a dictionary: {"LIMIT_BAL": 0.25, "PAY_0": 0.67, ...}
    importance_map = dict(zip(feature_names, normalized_importance))

    # --- Step 2: Get the PSI for each feature on the MOST RECENT day ---
    # all_days_df has one row per production day, one column per feature's PSI
    # e.g. columns: ["day", "LIMIT_BAL_psi", "PAY_0_psi", ...]
    # .iloc[-1] grabs the last row = most recent day
    latest_day = all_days_df.iloc[-1]

    # --- Step 3: For each feature, compute Risk Score = PSI × Importance ---
    results = []
    for feature in feature_names:

        psi_col = f"{feature}_psi"   # column name in all_days_df for this feature's PSI

        # Skip if we don't have PSI data for this feature
        if psi_col not in all_days_df.columns:
            continue

        # Get this feature's PSI on the latest day
        latest_psi = latest_day[psi_col]

        # Get this feature's importance (how much the model relies on it)
        importance = importance_map.get(feature, 0)

        # Risk Score: high PSI + high importance = danger
        # e.g. PSI=0.8, importance=0.67 → risk=0.536 (very dangerous)
        # e.g. PSI=0.8, importance=0.04 → risk=0.032 (not worth alerting)
        risk_score = latest_psi * importance

        # --- Step 4: Check if drift has been sustained for N consecutive days ---
        # We don't want to alert on one-off spikes.
        # Only alert if the feature has been above threshold for 3+ days in a row.

        # Get the PSI values for this feature across ALL days as a list
        # e.g. [0.05, 0.12, 0.25, 0.31, 0.28] — one per day
        psi_history = all_days_df[psi_col].tolist()

        # Look at only the last N days
        # e.g. if consecutive_days=3, look at last 3 values
        recent_psi = psi_history[-consecutive_days:]

        # Check if ALL of those recent values are above the critical threshold
        # all([True, True, True]) → True (sustained drift → alert)
        # all([True, False, True]) → False (not sustained → no alert)
        sustained = len(recent_psi) >= consecutive_days and all(p >= psi_thresh for p in recent_psi)

        results.append({
            "Feature":    feature,
            "Importance": round(importance, 4),   # how much the model uses this feature
            "Latest PSI": round(latest_psi, 4),   # how much this feature has drifted today
            "Risk Score": round(risk_score, 4),   # combined danger signal
            "Sustained":  sustained,               # has it been drifting for 3+ days?
            "Should Alert": sustained and risk_score > 0.05,  # final yes/no alert decision
        })

    # Sort by Risk Score descending — most dangerous features first
    results_df = pd.DataFrame(results).sort_values("Risk Score", ascending=False)

    return results_df


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
            y_true = labels_df["target"].values.astype(int)

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
        consecutive_days = st.sidebar.slider(
            "Consecutive days before alert", 1, 7,
            value=min(3, len(all_days_df)),   # default 3, but cap at available days
            step=1
        )

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