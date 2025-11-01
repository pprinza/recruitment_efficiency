# ==========================================================
# Recruitment Efficiency Prediction App (Final Safe Version)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Recruitment Efficiency Dashboard", layout="centered")

# --- Title ---
st.title("📊 Recruitment Efficiency Prediction Dashboard")
st.write("Use this dashboard to predict **Hiring Duration**, **Cost per Hire**, and **Offer Acceptance Rate**.")

# --- Load model safely ---
@st.cache_resource
def load_model():
    try:
        model_dict = joblib.load("model_recruitment_final.pkl")
        st.success("✅ Model loaded successfully!")
        return model_dict
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure 'model_recruitment_final.pkl' is uploaded.")
        return None
    except EOFError:
        st.error("❌ Model file corrupted or incomplete. Re-upload compressed model version.")
        return None

model_dict = load_model()

# --- Select target ---
target = st.radio(
    "Select prediction target:",
    ["Hiring Duration", "Cost per Hire", "Offer Acceptance Rate"]
)

# --- File uploader ---
uploaded_file = st.file_uploader("Upload recruitment dataset (.csv)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Uploaded dataset preview:")
    st.dataframe(df.head())

    # --- Detect model based on target ---
    model_map = {
        "Hiring Duration": "duration_model",
        "Cost per Hire": "cost_model",
        "Offer Acceptance Rate": "offer_acceptance_model"
    }

    model_name = model_map[target]
    model = model_dict.get(model_name, None)

    if model is None:
        st.error(f"❌ Model for {target} not found in model file.")
    else:
        try:
            # --- Ensure all required features exist ---
            expected_features = list(model.feature_names_in_)
            missing_features = [f for f in expected_features if f not in df.columns]

            # Fill missing columns with neutral values (median-like defaults)
            for f in missing_features:
                df[f] = 0.5

            if missing_features:
                st.warning(f"⚠️ Missing columns auto-filled: {missing_features}")

            # Reorder columns according to model
            df = df[expected_features]

            # --- Run Prediction ---
            preds = model.predict(df)

            # --- Display Results ---
            if target == "Hiring Duration":
                st.success("✅ Hiring Duration predicted successfully!")
                result_df = pd.DataFrame({"Predicted_Hiring_Duration": preds})
                st.dataframe(result_df.head())
                st.metric("Avg Predicted Hiring Duration (days)", f"{np.mean(preds):.2f}")

            elif target == "Cost per Hire":
                st.success("✅ Cost per Hire predicted successfully!")
                result_df = pd.DataFrame({"Predicted_Cost_per_Hire": preds})
                st.dataframe(result_df.head())
                st.metric("Avg Predicted Cost per Hire ($)", f"{np.mean(preds):,.2f}")

            else:
                st.success("✅ Offer Acceptance Rate predicted successfully!")
                result_df = pd.DataFrame({"Predicted_Offer_Acceptance": preds})
                st.dataframe(result_df.head())
                st.metric("Avg Predicted Offer Acceptance (%)", f"{np.mean(preds) * 100:.2f}%")

        except Exception as e:
            st.error(f"🚨 Prediction failed: {e}")

else:
    st.info("📂 Please upload your dataset to start prediction.")
