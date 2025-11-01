# ==========================================================
# STREAMLIT APP (MULTI-MODEL VERSION)
# ==========================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Recruitment Efficiency Dashboard", page_icon="📊", layout="wide")

st.title("📊 Prediction Dashboard")
st.markdown("""
Use this dashboard to predict **Hiring Duration**, **Cost per Hire**, and **Offer Acceptance Rate**.
""")

# Load all models
@st.cache_resource
def load_models():
    try:
        duration = joblib.load("model_duration.pkl")
        cost = joblib.load("model_cost.pkl")
        acceptance = joblib.load("model_accept_v4.pkl")
        return duration, cost, acceptance
    except Exception as e:
        st.error(f"❌ Failed to load models: {e}")
        return None, None, None

model_duration, model_cost, model_accept = load_models()
if model_duration and model_cost and model_accept:
    st.success("✅ All models loaded successfully!")

# Upload data
target = st.radio("Select prediction target:", ["Hiring Duration", "Cost per Hire", "Offer Acceptance Rate"])
uploaded_file = st.file_uploader("Upload recruitment dataset (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Uploaded dataset preview:")
    st.dataframe(df.head())

    if st.button("Run Prediction"):
        try:
            if target == "Hiring Duration":
                preds = model_duration.predict(df)
                st.success("✅ Hiring Duration predicted successfully!")
                st.dataframe(pd.DataFrame({"Predicted_Hiring_Duration": preds}))
                st.metric("Avg Predicted Hiring Duration (days)", f"{np.mean(preds):.2f}")
            elif target == "Cost per Hire":
                preds = model_cost.predict(df)
                st.success("✅ Cost per Hire predicted successfully!")
                st.dataframe(pd.DataFrame({"Predicted_Cost_per_Hire": preds}))
                st.metric("Avg Predicted Cost per Hire ($)", f"{np.mean(preds):,.2f}")
            else:
                preds = model_accept.predict(df)
                st.success("✅ Offer Acceptance predicted successfully!")
                st.dataframe(pd.DataFrame({"Predicted_Offer_Acceptance (%)": preds * 100}))
                st.metric("Avg Predicted Offer Acceptance (%)", f"{np.mean(preds)*100:.2f}%")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
