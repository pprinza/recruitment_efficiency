# ==========================================================
# Recruitment Efficiency Insight Dashboard (Final Version)
# ==========================================================
# Author: NeuraLens
# Purpose: Dashboard for recruitment analytics & prediction
# ==========================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ----------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------
st.set_page_config(page_title="Recruitment Efficiency Dashboard", layout="wide")
st.title("Recruitment Efficiency Insight Dashboard")

st.markdown("""
This dashboard helps HR teams analyze and predict recruitment efficiency across
departments, sources, and job roles. It uses three key metrics:

- **Time to Hire**
- **Cost per Hire**
- **Offer Acceptance Rate**
""")

# ----------------------------------------------------------
# MODEL HANDLING
# ----------------------------------------------------------
MODEL_DIR = "retrain_outputs"

MODEL_FILES = {
    "time_to_hire_days": "model_time_to_hire_days_FEv3.pkl",
    "cost_per_hire": "model_cost_per_hire_FEv3.pkl",
    "offer_acceptance_rate": "model_offer_acceptance_rate_FEv3.pkl",
}

@st.cache_resource
def load_models():
    models = {}
    missing = []
    for key, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            try:
                models[key] = joblib.load(path)
            except Exception as e:
                st.warning(f"⚠️ Could not load {fname}: {e}")
                models[key] = None
        else:
            models[key] = None
            missing.append(fname)
    return models, missing

models, missing_models = load_models()
models_available = any(m is not None for m in models.values())

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
@st.cache_data
def load_data():
    if os.path.exists("final_recruitment_data_FEv3.csv"):
        return pd.read_csv("final_recruitment_data_FEv3.csv")
    else:
        return None

base_df = load_data()

# ----------------------------------------------------------
# EFFICIENCY SCORE CALCULATION
# ----------------------------------------------------------
def compute_efficiency(df):
    df = df.copy()
    for metric in ["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate"]:
        if metric not in df.columns:
            df[metric] = np.nan

    df["time_score"] = 1 - (df["time_to_hire_days"] - df["time_to_hire_days"].min()) / (
        df["time_to_hire_days"].max() - df["time_to_hire_days"].min()
    )
    df["cost_score"] = 1 - (df["cost_per_hire"] - df["cost_per_hire"].min()) / (
        df["cost_per_hire"].max() - df["cost_per_hire"].min()
    )
    df["accept_score"] = (df["offer_acceptance_rate"] - df["offer_acceptance_rate"].min()) / (
        df["offer_acceptance_rate"].max() - df["offer_acceptance_rate"].min()
    )
    df["efficiency_score"] = (
        0.4 * df["time_score"] + 0.3 * df["cost_score"] + 0.3 * df["accept_score"]
    )
    return df

# ----------------------------------------------------------
# CREATE TABS
# ----------------------------------------------------------
tab_exec, tab_dept, tab_source, tab_job, tab_top10, tab_predict = st.tabs([
    "Executive Summary",
    "Department Efficiency",
    "Source Efficiency",
    "Job Role Efficiency",
    "Top 10 Most Efficient Recruitments",
    "Batch Prediction"
])

# ----------------------------------------------------------
# EXECUTIVE SUMMARY
# ----------------------------------------------------------
with tab_exec:
    st.header("Recruitment KPI Scorecard — Executive Overview")

    if base_df is None:
        st.warning("Default dataset not found. Please upload via Batch Prediction tab.")
    else:
        df = compute_efficiency(base_df)

        avg_time = round(df["time_to_hire_days"].mean(), 1)
        avg_cost = round(df["cost_per_hire"].mean(), 1)
        avg_accept = round(df["offer_acceptance_rate"].mean() * 100, 1)

        c1, c2, c3 = st.columns(3)
        c1.metric("Average Time to Hire (days)", f"{avg_time}")
        c2.metric("Average Cost per Hire ($)", f"{avg_cost:,}")
        c3.metric("Offer Acceptance Rate (%)", f"{avg_accept}%")

        st.divider()

        st.subheader("Most Efficient Highlights")
        dept_best = df.groupby("department")["efficiency_score"].mean().sort_values(ascending=False)
        src_best = df.groupby("source")["efficiency_score"].mean().sort_values(ascending=False)
        job_best = df.groupby("job_title")["efficiency_score"].mean().sort_values(ascending=False)

        col1, col2, col3 = st.columns(3)
        col1.metric("Most Efficient Department", dept_best.index[0])
        col2.metric("Most Efficient Job Title", job_best.index[0])
        col3.metric("Most Efficient Source", src_best.index[0])

# ----------------------------------------------------------
# TAB 6: BATCH PREDICTION — CSV Upload + Download Result
# ----------------------------------------------------------
with tab_predict:
    st.header("Batch Prediction — Predict Recruitment KPIs from CSV")

    uploaded_file = st.file_uploader("📤 Upload your recruitment dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("📊 Data Preview")
        st.dataframe(data.head(), use_container_width=True)

        if models_available:
            try:
                with st.spinner("Running predictions... Please wait."):
                    # Jalankan prediksi untuk tiap model
                    data["pred_time_to_hire_days"] = models["time_to_hire_days"].predict(data)
                    data["pred_cost_per_hire"] = models["cost_per_hire"].predict(data)
                    data["pred_offer_acceptance_rate"] = models["offer_acceptance_rate"].predict(data)

                st.success("✅ Prediction completed successfully!")

                # Kolom hasil utama
                display_cols = [
                    "department", "source", "job_title",
                    "pred_time_to_hire_days", "pred_cost_per_hire", "pred_offer_acceptance_rate"
                ]

                st.subheader("📈 Prediction Results")
                st.dataframe(data[display_cols].head(15), use_container_width=True)

                # Ringkasan
                st.subheader("📊 Summary Statistics (Predictions)")
                col1, col2, col3 = st.columns(3)
                col1.metric("Avg Predicted Time to Hire (days)", f"{data['pred_time_to_hire_days'].mean():.1f}")
                col2.metric("Avg Predicted Cost per Hire ($)", f"{data['pred_cost_per_hire'].mean():,.0f}")
                col3.metric("Avg Predicted Offer Acceptance Rate (%)", f"{data['pred_offer_acceptance_rate'].mean() * 100:.1f}%")

                # Feature Importance
                st.subheader("🔍 Feature Importance Summary")
                for key, model in models.items():
                    if hasattr(model, "feature_importances_"):
                        imp_df = pd.DataFrame({
                            "Feature": model.feature_names_in_,
                            "Importance": model.feature_importances_
                        }).sort_values("Importance", ascending=False)
                        st.markdown(f"**Model: {key}**")
                        st.bar_chart(imp_df.set_index("Feature"))

                # --- 🔽 DOWNLOAD BUTTON ---
                st.divider()
                st.subheader("💾 Download Predicted Results")

                # Generate nama file dinamis
                date_str = datetime.now().strftime("%Y%m%d_%H%M")
                csv_name = f"recruitment_predictions_{date_str}.csv"

                csv_data = data[display_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Prediction Results (CSV)",
                    data=csv_data,
                    file_name=csv_name,
                    mime="text/csv"
                )
                st.caption(f"File: `{csv_name}` generated successfully.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.warning("⚠️ Model files not found in 'retrain_outputs/'. Please check your repository.")
    else:
        st.info("Upload a CSV file to run batch predictions.")
