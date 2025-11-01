import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model once
model_dict = joblib.load("model_recruitment_final.pkl")

st.title("🎯 Recruitment Efficiency Prediction App")

# Upload CSV
uploaded_file = st.file_uploader("Upload recruitment dataset (.csv)", type=["csv"])

# Prediction target
target = st.radio(
    "Select prediction target:",
    ("Hiring Duration", "Cost per Hire", "Offer Acceptance Rate")
)

# Run only if file uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Uploaded dataset shape:", df.shape)
    st.dataframe(df.head())

    # --- PRE-PROCESSING CONSISTENCY FIX ---
    if target == "Hiring Duration":
        model = model_dict["duration_model"]
    elif target == "Cost per Hire":
        model = model_dict["cost_model"]
    else:
        model = model_dict["offer_acceptance_model"]

    # --- 1️⃣ Pastikan semua fitur training tersedia ---
    expected_features = list(model.feature_names_in_)
    missing_features = [f for f in expected_features if f not in df.columns]

    for f in missing_features:
        df[f] = 0.5  # nilai netral/default

    if missing_features:
        st.warning(f"⚠️ Missing columns auto-filled: {missing_features}")

    # --- 2️⃣ Pastikan urutan kolom sesuai training ---
    df = df[expected_features]

    # --- 3️⃣ Jalankan prediksi ---
    pred = model.predict(df)

    # --- 4️⃣ Tampilkan hasil ---
    if target == "Hiring Duration":
        st.success("✅ Hiring Duration predicted successfully!")
        st.write(pd.DataFrame({"Predicted_Hiring_Duration": pred}))
        st.metric("Avg Predicted Hiring Duration (days)", f"{np.mean(pred):.2f}")

    elif target == "Cost per Hire":
        st.success("✅ Cost per Hire predicted successfully!")
        st.write(pd.DataFrame({"Predicted_Cost_per_Hire": pred}))
        st.metric("Avg Predicted Cost per Hire ($)", f"{np.mean(pred):,.2f}")

    else:
        st.success("✅ Offer Acceptance Rate predicted successfully!")
        st.write(pd.DataFrame({"Predicted_Offer_Acceptance": pred}))
        st.metric("Avg Predicted Offer Acceptance (%)", f"{np.mean(pred)*100:.2f}%")

else:
    st.info("⬆️ Please upload a dataset (.csv) to start prediction.")
