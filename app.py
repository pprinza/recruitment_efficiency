# ==========================================================
# Recruitment Efficiency Modeling Dashboard (Final)
# Compatible with combined model_recruitment.pkl (3 models)
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
This interactive dashboard helps HR teams analyze and predict **recruitment efficiency** using machine learning.

It supports three main business goals:
- Reduce Hiring Duration  
- Reduce Cost per Hire  
- Increase Offer Acceptance Rate
---
""")

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    try:
        model_bundle = joblib.load("model_recruitment.pkl")
        if not all(k in model_bundle.keys() for k in [
            "hiring_duration_model", "cost_per_hire_model", "acceptance_rate_model"
        ]):
            st.error("The loaded model file is not compatible. Please check your model_recruitment.pkl.")
            return None
        return model_bundle
    except Exception as e:
        st.error(f"Error loading model_recruitment.pkl: {e}")
        return None

model_bundle = load_model()

# ----------------------------
# UPLOAD DATA
# ----------------------------
uploaded_file = st.file_uploader("📂 Upload your recruitment dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # ----------------------------
    # KEY METRICS OVERVIEW
    # ----------------------------
    st.subheader("Key Metrics Overview")
    col1, col2, col3 = st.columns(3)

    if "hiring_duration" in df.columns:
        avg_duration = round(df["hiring_duration"].mean(), 1)
    else:
        avg_duration = "-"

    if "cost_per_hire" in df.columns:
        avg_cost = round(df["cost_per_hire"].mean(), 1)
    else:
        avg_cost = "-"

    if "acceptance_rate" in df.columns:
        avg_accept = round(df["acceptance_rate"].mean() * 100, 1)
    else:
        avg_accept = "-"

    col1.metric("Avg Hiring Duration (days)", avg_duration)
    col2.metric("Avg Cost per Hire ($)", avg_cost)
    col3.metric("Offer Acceptance Rate (%)", avg_accept)

    # ----------------------------
    # VISUALIZATION SECTION
    # ----------------------------
    st.subheader("📈 Recruitment Insights")

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
    st.subheader("Predict Recruitment Efficiency")

    if model_bundle is not None:
        try:
            X_input = df.select_dtypes(include=[np.number, "object"])

            st.write("Click below to run all 3 predictive models:")
            if st.button("Run Prediction"):
                preds = {}

                preds["Predicted_Hiring_Duration"] = model_bundle["hiring_duration_model"].predict(X_input)
                preds["Predicted_Cost_per_Hire"] = model_bundle["cost_per_hire_model"].predict(X_input)
                preds["Predicted_Acceptance_Prob"] = model_bundle["acceptance_rate_model"].predict_proba(X_input)[:, 1]

                df["Predicted_Hiring_Duration"] = preds["Predicted_Hiring_Duration"]
                df["Predicted_Cost_per_Hire"] = preds["Predicted_Cost_per_Hire"]
                df["Predicted_Acceptance_Prob"] = preds["Predicted_Acceptance_Prob"]

                st.success("Predictions completed successfully!")
                st.dataframe(df.head())

                # ----------------------------
                # RESULT INSIGHTS
                # ----------------------------
                st.markdown("### 📉 Prediction Summary")
                summary = pd.DataFrame({
                    "Metric": ["Avg Predicted Duration", "Avg Predicted Cost", "Avg Acceptance Probability"],
                    "Value": [
                        round(df["Predicted_Hiring_Duration"].mean(), 2),
                        round(df["Predicted_Cost_per_Hire"].mean(), 2),
                        round(df["Predicted_Acceptance_Prob"].mean() * 100, 2)
                    ]
                })
                st.table(summary)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.info("Please upload a CSV file to begin the analysis.")

st.markdown("---")
st.caption("Developed for HR Analytics — Recruitment Efficiency Modeling © 2025")
