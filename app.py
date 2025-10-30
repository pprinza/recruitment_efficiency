# ==========================================================
# Recruitment Efficiency Modeling Dashboard (Multi-Model)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Recruitment Efficiency Dashboard",
    page_icon="📊",
    layout="wide"
)

# ----------------------------
# HEADER
# ----------------------------
st.title("Recruitment Efficiency Modeling Dashboard")
st.markdown("""
Analyze and predict **recruitment efficiency** using machine learning.

This dashboard supports three HR business objectives:
- Reduce Hiring Duration  
- Reduce Cost per Hire  
- Increase Offer Acceptance Rate  
---
""")

# ----------------------------
# LOAD MODELS
# ----------------------------
@st.cache_resource
def load_models():
    models = {}
    try:
        models["duration"] = joblib.load("model_duration.pkl")
        models["cost"] = joblib.load("model_cost.pkl")
        models["acceptance"] = joblib.load("model_acceptance.pkl")
        st.success("All models loaded successfully!")
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models

models = load_models()

# ----------------------------
# UPLOAD DATA
# ----------------------------
uploaded_file = st.file_uploader("Upload your recruitment dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # ----------------------------
    # KEY METRICS OVERVIEW
    # ----------------------------
    st.subheader("Key Metrics Overview")
    col1, col2, col3 = st.columns(3)

    avg_duration = round(df["hiring_duration"].mean(), 1) if "hiring_duration" in df.columns else "-"
    avg_cost = round(df["cost_per_hire"].mean(), 1) if "cost_per_hire" in df.columns else "-"
    avg_accept = f"{round(df['acceptance_rate'].mean() * 100, 1)}%" if "acceptance_rate" in df.columns else "-"

    col1.metric("Avg Hiring Duration (days)", avg_duration)
    col2.metric("Avg Cost per Hire ($)", avg_cost)
    col3.metric("Offer Acceptance Rate (%)", avg_accept)

    # ----------------------------
    # VISUALIZATION SECTION
    # ----------------------------
    st.subheader("Recruitment Insights")

    if "department" in df.columns and "cost_per_hire" in df.columns:
        fig = px.bar(df, x="department", y="cost_per_hire", color="department",
                     title="Average Cost per Hire by Department")
        st.plotly_chart(fig, use_container_width=True)

    if "source" in df.columns and "hiring_duration" in df.columns:
        fig2 = px.box(df, x="source", y="hiring_duration", color="source",
                      title="Hiring Duration by Source")
        st.plotly_chart(fig2, use_container_width=True)

    if "job_level" in df.columns and "acceptance_rate" in df.columns:
        fig3 = px.bar(df, x="job_level", y="acceptance_rate", color="job_level",
                      title="Offer Acceptance Rate by Job Level")
        st.plotly_chart(fig3, use_container_width=True)

    # ----------------------------
    # PREDICTION SECTION
    # ----------------------------
    st.subheader("Predict Recruitment Outcomes")

    if models:
        model_choice = st.radio(
            "Select prediction target:",
            ["Hiring Duration", "Cost per Hire", "Offer Acceptance Rate"]
        )

        X_input = df.select_dtypes(include=[np.number, "object"])

        if st.button("Run Prediction"):
            try:
                if model_choice == "Hiring Duration":
                    preds = models["duration"].predict(X_input)
                    df["Predicted_Hiring_Duration"] = preds
                    st.success("Hiring Duration predicted successfully!")
                    st.dataframe(df[["Predicted_Hiring_Duration"]].head())

                elif model_choice == "Cost per Hire":
                    preds = models["cost"].predict(X_input)
                    df["Predicted_Cost_per_Hire"] = preds
                    st.success("Cost per Hire predicted successfully!")
                    st.dataframe(df[["Predicted_Cost_per_Hire"]].head())

                else:
                    preds = models["acceptance"].predict_proba(X_input)[:, 1]
                    df["Predicted_Acceptance_Prob"] = preds
                    st.success("Offer Acceptance Probability predicted successfully!")
                    st.dataframe(df[["Predicted_Acceptance_Prob"]].head())

                # ----------------------------
                # SUMMARY INSIGHTS
                # ----------------------------
                st.markdown("### 📉 Prediction Summary")
                if "Predicted_Hiring_Duration" in df:
                    st.metric("Avg Predicted Hiring Duration (days)",
                              round(df["Predicted_Hiring_Duration"].mean(), 2))
                if "Predicted_Cost_per_Hire" in df:
                    st.metric("Avg Predicted Cost per Hire ($)",
                              round(df["Predicted_Cost_per_Hire"].mean(), 2))
                if "Predicted_Acceptance_Prob" in df:
                    st.metric("Avg Predicted Acceptance Probability (%)",
                              round(df["Predicted_Acceptance_Prob"].mean() * 100, 2))

            except Exception as e:
                st.error(f"Prediction failed: {e}")

else:
    st.info("Please upload a CSV file to begin analysis.")

st.markdown("---")
st.caption("Developed for HR Analytics — Recruitment Efficiency Modeling © 2025")
