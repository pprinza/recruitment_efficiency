import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

# Pastikan dataset sudah dibaca
st.write("✅ Uploaded dataset shape:", df.shape)

# --- 1️⃣ Pastikan semua fitur training tersedia ---
expected_features = list(model.feature_names_in_)
missing_features = [f for f in expected_features if f not in df.columns]

for f in missing_features:
    df[f] = 0.5  # nilai netral/default, atau gunakan mean saat training

if missing_features:
    st.warning(f"⚠️ Missing columns auto-filled: {missing_features}")

# --- 2️⃣ Pastikan urutan kolom sesuai training ---
df = df[expected_features]

# --- 3️⃣ (Opsional) Tangani encoding categorical jika pipeline tidak termasuk encoder ---
# Jika model kamu menyertakan ColumnTransformer di dalam pipeline, tidak perlu encode manual.
# Tapi kalau hanya DecisionTreeRegressor tanpa preprocessing pipeline, kamu perlu encode ulang.

# --- 4️⃣ Jalankan prediksi ---
pred = model.predict(df)

# --- 5️⃣ Tampilkan hasil ---
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
