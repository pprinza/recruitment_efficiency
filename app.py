# ==========================================================
# Recruitment Efficiency Prediction Dashboard (Final)
# ==========================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Recruitment Efficiency Prediction",
    layout="wide",
    page_icon="📊"
)

st.title("Recruitment Efficiency Prediction")
st.markdown(
    "Use this dashboard to predict **Hiring Duration**, **Cost per Hire**, and **Offer Acceptance Rate**."
)

# ==========================================================
# LOAD MODELS SAFELY
# ==========================================================
@st.cache_resource
def load_models():
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_duration = joblib.load(os.path.join(base_path, "model_duration.pkl"))
        model_cost = joblib.load(os.path.join(base_path, "model_cost.pkl"))
        model_accept = joblib.load(os.path.join(base_path, "model_accept.pkl"))
        st.success("Model loaded successfully!")
        return model_duration, model_cost, model_accept
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None, None

model_duration, model_cost, model_accept = load_models()

# ==========================================================
# PREDICTION TARGET SELECTION
# ==========================================================
target = st.radio(
    "Select prediction target:",
    ["Hiring Duration", "Cost per Hire", "Offer Acceptance Rate"]
)

# ==========================================================
# UPLOAD DATA
# ==========================================================
uploaded_file = st.file_uploader(
    "Upload recruitment dataset (.csv)",
    type=["csv"],
    help="Upload dataset containing recruitment features"
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded dataset preview:")
    st.dataframe(df.head())

    # Handle missing values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))

    # ==========================================================
    # MODEL SELECTION BASED ON TARGET
    # ==========================================================
    if st.button("Run Prediction"):
        try:
            if target == "Hiring Duration":
                if model_duration is None:
                    st.error("Model for Hiring Duration not found.")
                else:
                    pred = model_duration.predict(df)
                    st.success("Hiring Duration predicted successfully!")
                    st.dataframe(pd.DataFrame({"Predicted_Hiring_Duration (days)": pred}))
                    st.metric("Avg Predicted Duration", f"{np.mean(pred):.2f} days")

            elif target == "Cost per Hire":
                if model_cost is None:
                    st.error("Model for Cost per Hire not found.")
                else:
                    pred = model_cost.predict(df)
                    st.success("Cost per Hire predicted successfully!")
                    st.dataframe(pd.DataFrame({"Predicted_Cost_per_Hire ($)": pred}))
                    st.metric("Avg Predicted Cost", f"${np.mean(pred):,.2f}")

            else:
                if model_accept is None:
                    st.error("Model for Offer Acceptance Rate not found.")
                else:
                    pred = model_accept.predict(df)
                    st.success("Offer Acceptance Rate predicted successfully!")
                    st.dataframe(pd.DataFrame({"Predicted_Offer_Acceptance (%)": pred}))
                    st.metric("Avg Predicted Offer Acceptance", f"{np.mean(pred)*100:.2f}%")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.info("📂 Please upload a CSV file to start prediction.")

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("""
---
**Recruitment Efficiency Project**  
Predicting hiring duration, cost per hire, and offer acceptance using data-driven insights.  
Built with ❤️ by Patricia Prinza — 2025
""")
