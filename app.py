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
import io  
import matplotlib.pyplot as plt


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

# ----------------------------------------------------------
# BATCH PREDICTION TAB
# ----------------------------------------------------------
with tab_predict:
    st.header("Batch Prediction (Upload CSV)")
    st.write("Upload your recruitment dataset to predict Time to Hire, Cost per Hire, and Offer Acceptance Rate.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        user_df = pd.read_csv(uploaded)
        st.subheader("Data Preview")
        st.dataframe(user_df.head(), use_container_width=True)

        # try model prediction
        if models_available:
            preds = {}
            for key, model in models.items():
                if model is not None:
                    try:
                        preds[key] = model.predict(user_df)
                    except Exception as e:
                        preds[key] = None
                        st.warning(f"Model '{key}' failed: {e}")

            if any(v is not None for v in preds.values()):
                st.success("Prediction completed successfully.")
                user_df["pred_time_to_hire_days"] = preds.get("time", np.nan)
                user_df["pred_cost_per_hire"] = preds.get("cost", np.nan)
                user_df["pred_offer_acceptance_rate"] = preds.get("offer", np.nan)

                show_cols = [
                    "department", "source", "job_title",
                    "pred_time_to_hire_days", "pred_cost_per_hire",
                    "pred_offer_acceptance_rate"
                ]
                st.dataframe(user_df[show_cols].head(20), use_container_width=True)

                # ✅ --- DOWNLOAD BUTTON ---
                csv_buffer = io.BytesIO()
                user_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Prediction Results (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="prediction_results.csv",
                    mime="text/csv"
                )

# ✅ --- FEATURE IMPORTANCE SECTION (AUTO DETECT ENCODER) ---
st.subheader("🔍 Feature Importance Overview")

found_any = False
for key, model in models.items():
    if model is None:
        continue

    feature_names = list(user_df.columns)
    model_inner = None
    encoder_found = False

    # --- Deteksi pipeline ---
    if hasattr(model, "named_steps"):
        # Cari step yang punya get_feature_names_out() → biasanya encoder/transformer
        for step_name, step_obj in model.named_steps.items():
            if hasattr(step_obj, "get_feature_names_out"):
                try:
                    feature_names = step_obj.get_feature_names_out()
                    st.caption(f"🧩 Extracted encoded feature names from step '{step_name}' ({len(feature_names)} features).")
                    encoder_found = True
                    break
                except Exception:
                    pass

        # Cari step model utama (yang punya feature_importances_ atau coef_)
        for step_name, step_obj in model.named_steps.items():
            if hasattr(step_obj, "feature_importances_") or hasattr(step_obj, "coef_"):
                model_inner = step_obj
                break
    else:
        model_inner = model

    if model_inner is None:
        continue

    st.write(f"**Model:** {key.upper()}")

    try:
        if hasattr(model_inner, "feature_importances_"):
            importances = model_inner.feature_importances_
        elif hasattr(model_inner, "coef_"):
            importances = np.abs(model_inner.coef_).flatten()
        else:
            st.warning("This model does not expose feature importances or coefficients.")
            continue

        # --- Cek kesesuaian panjang ---
        if len(importances) != len(feature_names):
            st.warning(
                f"⚠️ Feature count mismatch for model '{key}'. "
                f"Model has {len(importances)} features, but name list has {len(feature_names)}. "
                f"Feature importance cannot be mapped."
            )
            continue

        # --- Tampilkan hasil ---
        fi = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(10)

        fig, ax = plt.subplots()
        ax.barh(fi["Feature"], fi["Importance"])
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")
        ax.invert_yaxis()
        st.pyplot(fig)
        st.dataframe(fi, use_container_width=True)
        st.divider()
        found_any = True

    except Exception as e:
        st.error(f"Error extracting feature importance for model '{key}': {e}")

if not found_any:
    st.info("No valid feature importance or coefficient data found for any model.")

# ----------------------------------------------------------
# FOOTER / MODEL STATUS
# ----------------------------------------------------------
st.divider()
if missing_models:
    st.info(f"Missing model files: {missing_models}")
else:
    st.caption("All model files loaded successfully.")
