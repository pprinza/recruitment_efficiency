# ==========================================================
# Recruitment Efficiency Insight Dashboard
# ==========================================================
# Author: NeuraLens
# Purpose: Dnified Streamlit Dashboard for Analytical & Predictive Insight
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Recruitment Efficiency Dashboard", layout="wide")
st.title("Recruitment Efficiency Insight Dashboard")

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("final_recruitment_data_FEv3.csv")
    return df

df = load_data()

# ----------------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        models = {
            "time": joblib.load("model_time_to_hire_days_FEv3.pkl"),
            "cost": joblib.load("model_cost_per_hire_FEv3.pkl"),
            "offer": joblib.load("model_offer_acceptance_rate_FEv3.pkl")
        }
        return models
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

models = load_models()

# ----------------------------------------------------------
# FUNCTION — Compute Efficiency
# ----------------------------------------------------------
def compute_efficiency(df):
    df = df.copy()
    df['time_score'] = 1 - (df['time_to_hire_days'] - df['time_to_hire_days'].min()) / (df['time_to_hire_days'].max() - df['time_to_hire_days'].min())
    df['cost_score'] = 1 - (df['cost_per_hire'] - df['cost_per_hire'].min()) / (df['cost_per_hire'].max() - df['cost_per_hire'].min())
    df['accept_score'] = (df['offer_acceptance_rate'] - df['offer_acceptance_rate'].min()) / (df['offer_acceptance_rate'].max() - df['offer_acceptance_rate'].min())
    df['efficiency_score'] = 0.4 * df['time_score'] + 0.3 * df['cost_score'] + 0.3 * df['accept_score']
    return df

df = compute_efficiency(df)

# ----------------------------------------------------------
# CREATE TABS
# ----------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Executive Summary",
    "Department Efficiency",
    "Source Efficiency",
    "Job Role Efficiency",
    "Top 10 Most Efficient Recruitments",
    "Batch Prediction"
])

# ==========================================================
# TAB 1-5: Existing
# ==========================================================
with tab1:
    st.header("Recruitment KPI Scorecard — Executive Overview")

    avg_time = round(df['time_to_hire_days'].mean())
    avg_cost = round(df['cost_per_hire'].mean())
    avg_accept = round(df['offer_acceptance_rate'].mean() * 100)

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Time to Hire (days)", f"{avg_time}")
    col2.metric("Average Cost per Hire ($)", f"{avg_cost:,}")
    col3.metric("Offer Acceptance Rate (%)", f"{avg_accept}%")

    st.divider()
    st.subheader("The Most Efficient Department")
    best_dept = df.groupby("department")[["time_to_hire_days","cost_per_hire","offer_acceptance_rate","efficiency_score"]].mean().sort_values("efficiency_score", ascending=False)
    st.dataframe(best_dept.head(1).reset_index(), use_container_width=True)

with tab2:
    st.header("Department Efficiency Overview (Sorted by Efficiency Score)")
    dept_summary = df.groupby("department")[["time_to_hire_days","cost_per_hire","offer_acceptance_rate","efficiency_score"]].mean().sort_values("efficiency_score", ascending=False)
    st.dataframe(dept_summary.reset_index(), use_container_width=True)

with tab3:
    st.header("Source Efficiency Overview (Sorted by Efficiency Score)")
    source_summary = df.groupby("source")[["time_to_hire_days","cost_per_hire","offer_acceptance_rate","efficiency_score"]].mean().sort_values("efficiency_score", ascending=False)
    st.dataframe(source_summary.reset_index(), use_container_width=True)

with tab4:
    st.header("Job Role Efficiency Overview (Sorted by Efficiency Score)")
    job_summary = df.groupby("job_title")[["time_to_hire_days","cost_per_hire","offer_acceptance_rate","efficiency_score"]].mean().sort_values("efficiency_score", ascending=False)
    st.dataframe(job_summary.reset_index(), use_container_width=True)

with tab5:
    st.header("Top 10 Most Efficient Recruitments (Individual Level)")
    top10 = df.sort_values("efficiency_score", ascending=False).head(10)
    st.dataframe(top10[["department","source","job_title","time_to_hire_days","cost_per_hire","offer_acceptance_rate","efficiency_score"]],
                 use_container_width=True)

# ==========================================================
# TAB 6: BATCH PREDICTION
# ==========================================================
with tab6:
    st.header("Batch Prediction — Predict Recruitment KPIs from CSV")

    uploaded_file = st.file_uploader("Upload your recruitment dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.dataframe(data.head(), use_container_width=True)

        if models:
            try:
                # Run predictions
                data["pred_time_to_hire_days"] = models["time"].predict(data)
                data["pred_cost_per_hire"] = models["cost"].predict(data)
                data["pred_offer_acceptance_rate"] = models["offer"].predict(data)

                st.success("✅ Prediction completed successfully.")
                st.dataframe(data.head(10), use_container_width=True)

                st.subheader("Summary Statistics (Predictions)")
                col1, col2, col3 = st.columns(3)
                col1.metric("Avg Predicted Time to Hire (days)", f"{data['pred_time_to_hire_days'].mean():.1f}")
                col2.metric("Avg Predicted Cost per Hire ($)", f"{data['pred_cost_per_hire'].mean():,.0f}")
                col3.metric("Avg Predicted Offer Acceptance Rate (%)", f"{data['pred_offer_acceptance_rate'].mean() * 100:.1f}%")

                # Downloadable CSV
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv,
                    file_name="recruitment_predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.warning("Models not loaded. Please check your .pkl files in GitHub or Streamlit environment.")
    else:
        st.info("Upload a CSV file to run predictions.")
