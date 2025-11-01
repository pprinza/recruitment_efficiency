# ==========================================================
# Recruitment Efficiency Insight Dashboard (Final Version)
# ==========================================================
# Author: NeuraLens
# Purpose: Dashboard for recruitment analytics & prediction
# ==========================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------
st.set_page_config(page_title="Recruitment Efficiency Dashboard", layout="wide")
st.title("Recruitment Efficiency Insight Dashboard")

st.markdown("""
This dashboard helps HR teams analyze and predict recruitment efficiency across
departments, sources, and job roles. It uses three key metrics:

- **Time to Hire**
- **Cost per Hire**
- **Offer Acceptance Rate**
""")

# ----------------------------------------------------------
# MODEL HANDLING
# ----------------------------------------------------------
MODEL_DIR = "."
MODEL_DIR = "retrain_outputs"

import os
import joblib

models = {}
try:
    models["time_to_hire_days"] = joblib.load(os.path.join(MODEL_DIR, "model_time_to_hire_days_FEv3.pkl"))
    models["cost_per_hire"] = joblib.load(os.path.join(MODEL_DIR, "model_cost_per_hire_FEv3.pkl"))
    models["offer_acceptance_rate"] = joblib.load(os.path.join(MODEL_DIR, "model_offer_acceptance_rate_FEv3.pkl"))
    print("✅ All models loaded successfully.")
except Exception as e:
    print("❌ Error loading model files:", e)

MODEL_FILES = {
    "time": "model_time_to_hire_days_FEv3.pkl",
    "cost": "model_cost_per_hire_FEv3.pkl",
    "offer": "model_offer_acceptance_rate_FEv3.pkl",
}


@st.cache_resource
def load_models():
    models = {}
    missing = []
    for key, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            try:
                models[key] = joblib.load(path)
            except Exception:
                models[key] = None
        else:
            models[key] = None
            missing.append(fname)
    return models, missing


models, missing_models = load_models()
models_available = any(m is not None for m in models.values())

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
@st.cache_data
def load_data():
    if os.path.exists("final_recruitment_data_FEv3.csv"):
        return pd.read_csv("final_recruitment_data_FEv3.csv")
    else:
        return None


base_df = load_data()

# ----------------------------------------------------------
# EFFICIENCY SCORE CALCULATION
# ----------------------------------------------------------
def compute_efficiency(df):
    df = df.copy()
    for metric in ["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate"]:
        if metric not in df.columns:
            df[metric] = np.nan

    df["time_score"] = 1 - (df["time_to_hire_days"] - df["time_to_hire_days"].min()) / (
        df["time_to_hire_days"].max() - df["time_to_hire_days"].min()
    )
    df["cost_score"] = 1 - (df["cost_per_hire"] - df["cost_per_hire"].min()) / (
        df["cost_per_hire"].max() - df["cost_per_hire"].min()
    )
    df["accept_score"] = (df["offer_acceptance_rate"] - df["offer_acceptance_rate"].min()) / (
        df["offer_acceptance_rate"].max() - df["offer_acceptance_rate"].min()
    )
    df["efficiency_score"] = (
        0.4 * df["time_score"] + 0.3 * df["cost_score"] + 0.3 * df["accept_score"]
    )
    return df


# ----------------------------------------------------------
# CREATE TABS
# ----------------------------------------------------------
tab_exec, tab_dept, tab_source, tab_job, tab_top10, tab_predict = st.tabs([
    "Executive Summary",
    "Department Efficiency",
    "Source Efficiency",
    "Job Role Efficiency",
    "Top 10 Most Efficient Recruitments",
    "Batch Prediction"
])

# ----------------------------------------------------------
# EXECUTIVE SUMMARY
# ----------------------------------------------------------
with tab_exec:
    st.header("Recruitment KPI Scorecard — Executive Overview")

    if base_df is None:
        st.warning("Default dataset not found. Please upload via Batch Prediction tab.")
    else:
        df = compute_efficiency(base_df)

        avg_time = round(df["time_to_hire_days"].mean(), 1)
        avg_cost = round(df["cost_per_hire"].mean(), 1)
        avg_accept = round(df["offer_acceptance_rate"].mean() * 100, 1)

        c1, c2, c3 = st.columns(3)
        c1.metric("Average Time to Hire (days)", f"{avg_time}")
        c2.metric("Average Cost per Hire ($)", f"{avg_cost:,}")
        c3.metric("Offer Acceptance Rate (%)", f"{avg_accept}%")

        st.divider()

        st.subheader("Most Efficient Highlights")
        dept_best = df.groupby("department")["efficiency_score"].mean().sort_values(ascending=False)
        src_best = df.groupby("source")["efficiency_score"].mean().sort_values(ascending=False)
        job_best = df.groupby("job_title")["efficiency_score"].mean().sort_values(ascending=False)

        col1, col2, col3 = st.columns(3)
        col1.metric("Most Efficient Department", dept_best.index[0])
        col2.metric("Most Efficient Job Title", job_best.index[0])
        col3.metric("Most Efficient Source", src_best.index[0])

# ----------------------------------------------------------
# DEPARTMENT EFFICIENCY
# ----------------------------------------------------------
with tab_dept:
    st.header("Department Efficiency Overview")
    if base_df is not None:
        df = compute_efficiency(base_df)
        dept_summary = (
            df.groupby("department")[["time_to_hire_days", "cost_per_hire",
                                      "offer_acceptance_rate", "efficiency_score"]]
            .mean()
            .sort_values("efficiency_score", ascending=False)
        )
        st.dataframe(dept_summary.reset_index(), use_container_width=True, hide_index=True)

# ----------------------------------------------------------
# SOURCE EFFICIENCY
# ----------------------------------------------------------
with tab_source:
    st.header("Source Efficiency Overview")
    if base_df is not None:
        df = compute_efficiency(base_df)
        src_summary = (
            df.groupby("source")[["time_to_hire_days", "cost_per_hire",
                                  "offer_acceptance_rate", "efficiency_score"]]
            .mean()
            .sort_values("efficiency_score", ascending=False)
        )
        st.dataframe(src_summary.reset_index(), use_container_width=True, hide_index=True)

# ----------------------------------------------------------
# JOB ROLE EFFICIENCY
# ----------------------------------------------------------
with tab_job:
    st.header("Job Role Efficiency Overview")
    if base_df is not None:
        df = compute_efficiency(base_df)
        job_summary = (
            df.groupby("job_title")[["time_to_hire_days", "cost_per_hire",
                                     "offer_acceptance_rate", "efficiency_score"]]
            .mean()
            .sort_values("efficiency_score", ascending=False)
        )
        st.dataframe(job_summary.reset_index(), use_container_width=True, hide_index=True)

# ----------------------------------------------------------
# TOP 10 MOST EFFICIENT RECRUITMENTS
# ----------------------------------------------------------
with tab_top10:
    st.header("Top 10 Most Efficient Recruitments")
    if base_df is not None:
        df = compute_efficiency(base_df)
        top10 = df.sort_values("efficiency_score", ascending=False).head(10)
        st.dataframe(
            top10[["department", "source", "job_title",
                   "time_to_hire_days", "cost_per_hire",
                   "offer_acceptance_rate", "efficiency_score"]],
            use_container_width=True, hide_index=True,
        )

# ==========================================================
# BATCH PREDICTION TAB (UPLOAD CSV + FEATURE IMPORTANCE)
# ==========================================================
with tab_predict:
    st.header("Batch Prediction — Recruitment Simulation & Model Explainability")

    st.markdown("""
    Unggah file CSV berisi data rekrutmen, jalankan prediksi untuk **Time to Hire**, **Cost per Hire**, 
    dan **Offer Acceptance Rate**, lalu unduh hasil prediksi.  
    Dashboard ini juga menampilkan **Feature Importance** dari setiap model untuk membantu menjelaskan pengaruh fitur.
    """)

    uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            # Load uploaded dataset
            user_df = pd.read_csv(uploaded_file)
            st.subheader("📋 Data Preview")
            st.dataframe(user_df.head(), use_container_width=True)

            # pastikan model tersedia
            if not models_available:
                st.error("Model file (.pkl) tidak ditemukan di direktori retrain_outputs/")
            else:
                st.divider()
                st.info("⏳ Running prediction... Please wait.")

                preds = {}

                # Prediksi untuk masing-masing target
                for key, model in models.items():
                    if model is None:
                        preds[key] = np.nan
                        continue

                    # pastikan kolom sesuai dengan feature_names_in_
                    if hasattr(model, "feature_names_in_"):
                        model_features = model.feature_names_in_
                        missing = [f for f in model_features if f not in user_df.columns]
                        for m in missing:
                            user_df[m] = 0
                        X_input = user_df[model_features]
                    else:
                        X_input = user_df.select_dtypes(include=[np.number])

                    preds[key] = model.predict(X_input)

                # gabungkan hasil prediksi ke dataframe
                user_df["pred_time_to_hire_days"] = np.clip(preds.get("time_to_hire_days", np.nan), 0, None)
                user_df["pred_cost_per_hire"] = np.clip(preds.get("cost_per_hire", np.nan), 0, None)
                user_df["pred_offer_acceptance_rate"] = np.clip(preds.get("offer_acceptance_rate", np.nan), 0, 1)

                st.success("✅ Prediction completed successfully!")
                st.subheader("📊 Prediction Results")
                st.dataframe(
                    user_df[
                        [
                            "department", "source", "job_title",
                            "pred_time_to_hire_days", "pred_cost_per_hire", "pred_offer_acceptance_rate"
                        ]
                    ].head(10),
                    use_container_width=True
                )

                # Download hasil prediksi
                csv_download = user_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Predicted Results (CSV)",
                    data=csv_download,
                    file_name="predicted_recruitment_outcomes.csv",
                    mime="text/csv",
                )

                # ==========================================================
                # FEATURE IMPORTANCE VISUALIZATION
                # ==========================================================
                st.divider()
                st.subheader("🔍 Model Feature Importance")

                import matplotlib.pyplot as plt

                for key, model in models.items():
                    if hasattr(model, "feature_importances_"):
                        importance_df = pd.DataFrame({
                            "Feature": model.feature_names_in_,
                            "Importance": model.feature_importances_
                        }).sort_values("Importance", ascending=False).head(10)

                        st.markdown(f"**Top 10 Important Features — {key.replace('_', ' ').title()}**")

                        fig, ax = plt.subplots()
                        ax.barh(importance_df["Feature"], importance_df["Importance"])
                        ax.set_xlabel("Importance")
                        ax.set_ylabel("Feature")
                        ax.invert_yaxis()
                        st.pyplot(fig)

                    elif hasattr(model, "coef_"):
                        # untuk model linear (ridge / lasso)
                        importance_df = pd.DataFrame({
                            "Feature": model.feature_names_in_,
                            "Coefficient": model.coef_.flatten()
                        }).sort_values("Coefficient", ascending=False).head(10)

                        st.markdown(f"**Top 10 Coefficients — {key.replace('_', ' ').title()}**")

                        fig, ax = plt.subplots()
                        ax.barh(importance_df["Feature"], importance_df["Coefficient"])
                        ax.set_xlabel("Coefficient")
                        ax.set_ylabel("Feature")
                        ax.invert_yaxis()
                        st.pyplot(fig)

        except Exception as e:
            st.error(f"Gagal menjalankan prediksi: {e}")
    else:
        st.info("📤 Silakan unggah file CSV terlebih dahulu untuk menjalankan simulasi prediksi.")
