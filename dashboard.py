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
            with open("model.pkl", "rb") as f:
                return pickle.load(f)
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
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", "🔍 Feature Drift", "🎯 Prediction Drift", "📉 Performance"
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
            y_true  = labels_df.iloc[:, 0].values
            y_score = model.predict_proba(prod_df[common_cols_perf])[:, 1]
            y_pred  = (y_score >= 0.5).astype(int)

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

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.caption("DriftGuard Dashboard · Built with Streamlit & Plotly")