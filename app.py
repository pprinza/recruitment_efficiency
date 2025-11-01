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
MODEL_DIR = "."
MODEL_FILES = {
    "time": "model_time_to_hire_days_FEv3.pkl",
    "cost": "model_cost_per_hire_FEv3.pkl",
    "offer": "model_offer_acceptance_rate_FEv3.pkl",
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
            except Exception:
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
# DEPARTMENT EFFICIENCY
# ----------------------------------------------------------
with tab_dept:
    st.header("Department Efficiency Overview")
    if base_df is not None:
        df = compute_efficiency(base_df)
        dept_summary = (
            df.groupby("department")[["time_to_hire_days", "cost_per_hire",
                                      "offer_acceptance_rate", "efficiency_score"]]
            .mean()
            .sort_values("efficiency_score", ascending=False)
        )
        st.dataframe(dept_summary.reset_index(), use_container_width=True, hide_index=True)

# ----------------------------------------------------------
# SOURCE EFFICIENCY
# ----------------------------------------------------------
with tab_source:
    st.header("Source Efficiency Overview")
    if base_df is not None:
        df = compute_efficiency(base_df)
        src_summary = (
            df.groupby("source")[["time_to_hire_days", "cost_per_hire",
                                  "offer_acceptance_rate", "efficiency_score"]]
            .mean()
            .sort_values("efficiency_score", ascending=False)
        )
        st.dataframe(src_summary.reset_index(), use_container_width=True, hide_index=True)

# ----------------------------------------------------------
# JOB ROLE EFFICIENCY
# ----------------------------------------------------------
with tab_job:
    st.header("Job Role Efficiency Overview")
    if base_df is not None:
        df = compute_efficiency(base_df)
        job_summary = (
            df.groupby("job_title")[["time_to_hire_days", "cost_per_hire",
                                     "offer_acceptance_rate", "efficiency_score"]]
            .mean()
            .sort_values("efficiency_score", ascending=False)
        )
        st.dataframe(job_summary.reset_index(), use_container_width=True, hide_index=True)

# ----------------------------------------------------------
# TOP 10 MOST EFFICIENT RECRUITMENTS
# ----------------------------------------------------------
with tab_top10:
    st.header("Top 10 Most Efficient Recruitments")
    if base_df is not None:
        df = compute_efficiency(base_df)
        top10 = df.sort_values("efficiency_score", ascending=False).head(10)
        st.dataframe(
            top10[["department", "source", "job_title",
                   "time_to_hire_days", "cost_per_hire",
                   "offer_acceptance_rate", "efficiency_score"]],
            use_container_width=True, hide_index=True,
        )

# ==========================================================
# BATCH PREDICTION TAB (Department & Source Input)
# ==========================================================
with tab_predict:
    st.header("Batch Prediction — Recruitment Outcome Estimator")
    st.markdown("Gunakan model untuk memprediksi hasil berdasarkan *department* dan *source* yang dipilih.")

    # Cek apakah model tersedia
    if not models_available:
        st.error("Model file (.pkl) tidak ditemukan. Silakan unggah model terlebih dahulu ke repository GitHub.")
    else:
        # Dropdown inputs
        dept_list = sorted(df['department'].dropna().unique())
        source_list = sorted(df['source'].dropna().unique())

        col1, col2 = st.columns(2)
        selected_dept = col1.selectbox("Pilih Department", dept_list)
        selected_source = col2.selectbox("Pilih Source", source_list)

        # Ambil contoh data dari dataset
        example_df = df[(df['department'] == selected_dept) & (df['source'] == selected_source)].copy()

        if example_df.empty:
            st.warning("Kombinasi data tidak ditemukan dalam dataset. Coba pilih kombinasi lain.")
        else:
            # Gunakan rata-rata dari kombinasi yang dipilih
            input_row = example_df.mean(numeric_only=True).to_frame().T

            st.write("**Input Sample (mean values):**")
            st.dataframe(input_row, use_container_width=True)

            try:
                # Gunakan model pkl untuk prediksi
                pred_time = models["time"].predict(input_row)[0]
                pred_cost = models["cost"].predict(input_row)[0]
                pred_offer = models["offer"].predict(input_row)[0]

                # Pastikan semua prediksi positif
                pred_time = np.clip(pred_time, 0, None)
                pred_cost = np.clip(pred_cost, 0, None)
                pred_offer = np.clip(pred_offer, 0, 1)

                # Tampilkan hasil prediksi
                st.success("Prediction completed successfully.")

                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted Time to Hire (days)", f"{pred_time:.1f}")
                c2.metric("Predicted Cost per Hire ($)", f"{pred_cost:,.0f}")
                c3.metric("Predicted Offer Acceptance Rate (%)", f"{pred_offer*100:.1f}%")

            except Exception as e:
                st.error(f"Gagal menjalankan prediksi: {e}")
