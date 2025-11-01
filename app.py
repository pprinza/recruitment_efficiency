import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================================
# 1️⃣ LOAD MODEL
# ==============================================
st.set_page_config(page_title="Recruitment Efficiency Dashboard", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("model_recruitment_final.pkl")

model_dict = load_model()

# ==============================================
# 2️⃣ APP TITLE
# ==============================================
st.title("📊 Predict Recruitment Outcomes")
st.write("Use this dashboard to predict **Hiring Duration**, **Cost per Hire**, and **Offer Acceptance Rate**.")

# ==============================================
# 3️⃣ SELECT TARGET
# ==============================================
target = st.radio("Select prediction target:", ["Hiring Duration", "Cost per Hire", "Offer Acceptance Rate"])

uploaded_file = st.file_uploader("Upload recruitment dataset (.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset Preview")
    st.dataframe(df.head())

    X = df.copy()

    if st.button("Run Prediction"):
        try:
            if target == "Hiring Duration":
                model = model_dict["hiring_duration_model"]
                pred = model.predict(X)
                avg = np.mean(pred)
                st.success("Hiring Duration predicted successfully!")
                st.write(pd.DataFrame(pred, columns=["Predicted_Hiring_Duration"]).head())
                st.metric("Avg Predicted Hiring Duration (days)", f"{avg:.2f}")
            
            elif target == "Cost per Hire":
                model = model_dict["cost_per_hire_model"]
                pred = model.predict(X)
                avg = np.mean(pred)
                st.success("Cost per Hire predicted successfully!")
                st.write(pd.DataFrame(pred, columns=["Predicted_Cost_per_Hire"]).head())
                st.metric("Avg Predicted Cost per Hire ($)", f"{avg:,.2f}")
            
            elif target == "Offer Acceptance Rate":
                model = model_dict["offer_acceptance_model"]
                pred = model.predict(X)
                avg = np.mean(pred) * 100
                st.success("Offer Acceptance Probability predicted successfully!")
                st.write(pd.DataFrame(pred, columns=["Predicted_Acceptance_Prob"]).head())
                st.metric("Avg Predicted Acceptance Probability (%)", f"{avg:.2f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("Please upload a dataset to start prediction.")
