# ============================
# Recruitment Efficiency Modeling App (Final Version)
# ============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
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
# TITLE & DESCRIPTION
# ----------------------------
st.title("Recruitment Efficiency Modeling Dashboard")
st.markdown("""
This interactive web app helps HR teams analyze and predict **recruitment efficiency** using machine learning.
It supports three main business goals:

- Reduce Hiring Duration  
- Reduce Cost per Hire  
- Increase Offer Acceptance Rate  
---
""")

# ----------------------------
# LOAD MODEL (Updated - No Cache + Auto Validation)
# ----------------------------
import joblib

def load_model():
    """
    Load trained recruitment efficiency model.
    This version ensures no old cache remains active and verifies model validity.
    """
    try:
        model = joblib.load("model_recruitment.pkl")
        
        # Cek apakah model sesuai dengan domain recruitment
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
            # Jika masih ada kolom 'Latitude' atau 'MedInc', berarti model salah
            invalid_features = {"Latitude", "Longitude", "MedInc", "AveRooms", "AveBedrms"}
            if any(f in invalid_features for f in feature_names):
                st.error("The loaded model is not the Recruitment Efficiency model. "
                         "Please re-upload the correct `model_recruitment.pkl` to GitHub.")
                return None
        
        st.success("Recruitment Efficiency model loaded successfully!")
        return model

    except FileNotFoundError:
        st.warning("No model file found (`model_recruitment.pkl`). "
                   "The app will still display analytics without predictions.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Load model saat startup
model = load_model()

# ----------------------------
# UPLOAD DATA
# ----------------------------
uploaded_file = st.file_uploader("Upload your recruitment dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # ----------------------------
    # BASIC STATS
    # ----------------------------
    st.subheader("Key Metrics Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Hiring Duration (days)", round(df["hiring_duration"].mean(), 1) if "hiring_duration" in df else "-")
    with col2:
        st.metric("Avg Cost per Hire ($)", round(df["cost_per_hire"].mean(), 1) if "cost_per_hire" in df else "-")
    with col3:
        # Support both "acceptance_rate" and "offer_acceptance_rate"
        acc_col = "acceptance_rate" if "acceptance_rate" in df else (
            "offer_acceptance_rate" if "offer_acceptance_rate" in df else None
        )
        if acc_col:
            st.metric("Offer Acceptance Rate (%)", f"{round(df[acc_col].mean() * 100, 1)}%")
        else:
            st.metric("Offer Acceptance Rate (%)", "-")

    # ----------------------------
    # VISUALIZATION
    # ----------------------------
    st.subheader("Recruitment Insights")

    # Cost per Hire by Department
    if "department" in df and "cost_per_hire" in df:
        fig = px.bar(df, x="department", y="cost_per_hire",
                     title="Average Cost per Hire by Department", color="department")
        st.plotly_chart(fig, use_container_width=True)

    # Hiring Duration by Source
    if "source" in df and "hiring_duration" in df:
        fig2 = px.box(df, x="source", y="hiring_duration",
                      title="Hiring Duration by Source", color="source")
        st.plotly_chart(fig2, use_container_width=True)

    # Offer Acceptance Rate by Department
    acc_col = "acceptance_rate" if "acceptance_rate" in df else (
        "offer_acceptance_rate" if "offer_acceptance_rate" in df else None
    )
    if acc_col and "department" in df:
        fig3 = px.box(df, x="department", y=acc_col, color="department",
                      title="Offer Acceptance Rate by Department")
        st.plotly_chart(fig3, use_container_width=True)

    # ----------------------------
    # PREDICTION SECTION
    # ----------------------------
    if model is not None:
        st.subheader("Predict Recruitment Efficiency")

        num_df = df.select_dtypes(include=[np.number])

        if st.button("Run Prediction"):
            try:
                preds = model.predict(num_df)
                df["Predicted_Efficiency"] = preds
                st.success("Prediction complete!")
                st.dataframe(df[["Predicted_Efficiency"]].head())

                # ----------------------------
                # SHAP EXPLAINABILITY
                # ----------------------------
                st.subheader("🔍 Model Explainability (SHAP)")
                explainer = shap.Explainer(model)
                shap_values = explainer(num_df)

                st.write("### Feature Importance Overview")
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, num_df, show=False)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Prediction failed. Check input columns. Details: {e}")

else:
    st.info("📂 Upload your dataset in CSV format to begin analysis.")
