# recruitment_dashboard_final.py
import os
import io
import datetime as dt

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------
st.set_page_config(page_title="Recruitment Efficiency Dashboard", layout="wide")
st.title("Recruitment Efficiency Prediction & Insight Dashboard")

st.markdown("""
This dashboard helps HR teams understand which departments, sources, and job roles 
contribute most to overall recruitment efficiency based on three key metrics:

- Time to Hire  
- Cost per Hire  
- Offer Acceptance Rate
""")

# ----------------------------------------------------------
# MODEL FILES (sesuaikan jika perlu)
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
            except Exception as e:
                st.warning(f"Failed to load model file {fname}: {e}")
                models[key] = None
        else:
            models[key] = None
            missing.append(fname)
    return models, missing

models, missing_models = load_models()
models_available = any(m is not None for m in models.values())

# ----------------------------------------------------------
# Load default dataset (opsional)
# ----------------------------------------------------------
@st.cache_data
def load_data():
    if os.path.exists("final_recruitment_data_FEv3.csv"):
        return pd.read_csv("final_recruitment_data_FEv3.csv")
    return None

base_df = load_data()

# ----------------------------------------------------------
# Utility: compute efficiency
# ----------------------------------------------------------
def compute_efficiency(df):
    df = df.copy()
    for metric in ["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate"]:
        if metric not in df.columns:
            df[metric] = np.nan

    # avoid division by zero when min==max
    def safe_scale(s):
        if s.max() - s.min() == 0:
            return pd.Series(0.5, index=s.index)
        return (s - s.min()) / (s.max() - s.min())

    df["time_score"] = 1 - safe_scale(df["time_to_hire_days"])
    df["cost_score"] = 1 - safe_scale(df["cost_per_hire"])
    df["accept_score"] = safe_scale(df["offer_acceptance_rate"])
    df["efficiency_score"] = 0.4 * df["time_score"] + 0.3 * df["cost_score"] + 0.3 * df["accept_score"]
    return df

# ----------------------------------------------------------
# Tabs
# ----------------------------------------------------------
tabs = st.tabs([
    "Executive Summary",
    "Department Efficiency",
    "Source Efficiency",
    "Job Role Efficiency",
    "Top 10 Most Efficient",
    "Batch Prediction",
    "Feature Importance"
])
tab_exec, tab_dept, tab_source, tab_job, tab_top10, tab_predict, tab_importance = tabs

# ----------------------------------------------------------
# Executive Summary
# ----------------------------------------------------------
with tab_exec:
    st.header("Recruitment KPI — Executive Overview")
    if base_df is None:
        st.warning("Default dataset not found. Please upload via Batch Prediction tab.")
    else:
        df = compute_efficiency(base_df)
        avg_time = int(df["time_to_hire_days"].mean()) if not df["time_to_hire_days"].isna().all() else "N/A"
        avg_cost = int(df["cost_per_hire"].mean()) if not df["cost_per_hire"].isna().all() else "N/A"
        avg_accept = int(df["offer_acceptance_rate"].mean() * 100) if not df["offer_acceptance_rate"].isna().all() else "N/A"

        c1, c2, c3 = st.columns(3)
        c1.metric("Average Time to Hire (days)", f"{avg_time}")
        c2.metric("Average Cost per Hire ($)", f"{avg_cost:,}")
        c3.metric("Offer Acceptance Rate (%)", f"{avg_accept}%")

        st.divider()
        st.subheader("Most Efficient Highlights")
        try:
            dept_best = df.groupby("department")["efficiency_score"].mean().sort_values(ascending=False)
            src_best = df.groupby("source")["efficiency_score"].mean().sort_values(ascending=False)
            job_best = df.groupby("job_title")["efficiency_score"].mean().sort_values(ascending=False)
            col1, col2, col3 = st.columns(3)
            col1.metric("Top Department", dept_best.index[0])
            col2.metric("Top Job Title", job_best.index[0])
            col3.metric("Top Source", src_best.index[0])
        except Exception:
            st.info("Not enough data to compute highlights.")

# ----------------------------------------------------------
# Department, Source, Job tabs
# ----------------------------------------------------------
with tab_dept:
    st.header("Department Efficiency Overview")
    if base_df is not None:
        df = compute_efficiency(base_df)
        dept_summary = df.groupby("department")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]].mean().sort_values("efficiency_score", ascending=False)
        st.dataframe(dept_summary.reset_index(), use_container_width=True, hide_index=True)

with tab_source:
    st.header("Source Efficiency Overview")
    if base_df is not None:
        df = compute_efficiency(base_df)
        src_summary = df.groupby("source")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]].mean().sort_values("efficiency_score", ascending=False)
        st.dataframe(src_summary.reset_index(), use_container_width=True, hide_index=True)

with tab_job:
    st.header("Job Role Efficiency Overview")
    if base_df is not None:
        df = compute_efficiency(base_df)
        job_summary = df.groupby("job_title")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]].mean().sort_values("efficiency_score", ascending=False)
        st.dataframe(job_summary.reset_index(), use_container_width=True, hide_index=True)

with tab_top10:
    st.header("Top 10 Most Efficient Recruitments")
    if base_df is not None:
        df = compute_efficiency(base_df)
        top10 = df.sort_values("efficiency_score", ascending=False).head(10)
        st.dataframe(top10[["department","source","job_title","time_to_hire_days","cost_per_hire","offer_acceptance_rate","efficiency_score"]], use_container_width=True, hide_index=True)

# ----------------------------------------------------------
# Batch Prediction
# ----------------------------------------------------------
with tab_predict:
    st.header("Batch Prediction (Upload CSV)")
    st.write("Upload your dataset to predict Time to Hire, Cost per Hire, and Offer Acceptance Rate.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        user_df = pd.read_csv(uploaded)
        st.subheader("Data Preview")
        st.dataframe(user_df.head(), use_container_width=True)

        if models_available:
            preds = {}
            for key, model in models.items():
                if model is not None:
                    try:
                        preds[key] = model.predict(user_df)
                    except Exception as e:
                        preds[key] = None
                        st.warning(f"Model '{key}' prediction failed: {e}")

            if any(v is not None for v in preds.values()):
                st.success("Prediction completed successfully.")

                # SAFE extraction & rounding (no decimals)
                # TIME predictions
                time_preds = preds.get("time")
                if time_preds is not None:
                    time_s = pd.Series(time_preds).round()
                else:
                    time_s = pd.Series([np.nan] * len(user_df))

                # COST predictions
                cost_preds = preds.get("cost")
                if cost_preds is not None:
                    cost_s = pd.Series(cost_preds).round()
                else:
                    cost_s = pd.Series([np.nan] * len(user_df))

                # OFFER predictions (rate) -> round 2 decimals for readability, but keep as float
                offer_preds = preds.get("offer")
                if offer_preds is not None:
                    offer_s = pd.Series(offer_preds).round(2)
                else:
                    offer_s = pd.Series([np.nan] * len(user_df))

                # Replace negative time with dummy minimal (3 days)
                time_s = time_s.apply(lambda x: 3 if pd.notna(x) and x < 0 else x)

                # Clip cost to >=0
                cost_s = cost_s.apply(lambda x: 0 if pd.notna(x) and x < 0 else x)

                # Convert to nullable integer for display where possible
                try:
                    user_df["pred_time_to_hire_days"] = time_s.astype("Int64")
                except Exception:
                    # fallback to float if Int64 conversion fails
                    user_df["pred_time_to_hire_days"] = time_s

                try:
                    user_df["pred_cost_per_hire"] = cost_s.astype("Int64")
                except Exception:
                    user_df["pred_cost_per_hire"] = cost_s

                user_df["pred_offer_acceptance_rate"] = offer_s

            
                # Show top rows
                show_cols = [
                    "department", "source", "job_title",
                    "pred_time_to_hire_days", "pred_hire_completion_date",
                    "pred_cost_per_hire", "pred_offer_acceptance_rate"
                ]
                # Only show columns that actually exist in user_df
                show_cols = [c for c in show_cols if c in user_df.columns]
                st.dataframe(user_df[show_cols].head(20), use_container_width=True)

                # Download button (CSV)
                try:
                    csv_buffer = io.BytesIO()
                    user_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Prediction Results (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name="prediction_results.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Failed to prepare download: {e}")

            else:
                st.error("All models failed to predict. Please check input features and model format.")
        else:
            st.warning("No model files found (.pkl). Please check deployment directory.")

# ----------------------------------------------------------
# Feature Importance Tab (table only)
# ----------------------------------------------------------
with tab_importance:
    st.header("Feature Importance Overview")
    found_any = False

    for key, model in models.items():
        if model is None:
            continue

        # default: raw input columns (if no encoder discovered)
        feature_names = None
        model_inner = None

        # If pipeline, attempt to detect encoder / transformer that can output names
        if hasattr(model, "named_steps"):
            # try to extract encoder/transformer feature names automatically
            for step_name, step_obj in model.named_steps.items():
                if hasattr(step_obj, "get_feature_names_out"):
                    try:
                        fnames = step_obj.get_feature_names_out()
                        # ensure list of strings
                        feature_names = [str(x) for x in fnames]
                        st.caption(f"🧩 Extracted encoded feature names from '{step_name}' ({len(feature_names)} features).")
                        break
                    except Exception:
                        feature_names = None

            # find final estimator step that exposes importances or coef
            for step_name, step_obj in model.named_steps.items():
                if hasattr(step_obj, "feature_importances_") or hasattr(step_obj, "coef_"):
                    model_inner = step_obj
                    break
        else:
            model_inner = model

        if model_inner is None:
            continue

        st.markdown(f"**Model:** {key.upper()}")

        try:
            if hasattr(model_inner, "feature_importances_"):
                importances = np.array(model_inner.feature_importances_)
            elif hasattr(model_inner, "coef_"):
                importances = np.abs(np.array(model_inner.coef_)).flatten()
            else:
                st.warning("Model does not provide feature importance or coefficients.")
                continue

            # if feature names not detected from encoder, fall back to saved 'feature_names_in_' if present
            if feature_names is None:
                if hasattr(model_inner, "feature_names_in_"):
                    try:
                        feature_names = [str(x) for x in model_inner.feature_names_in_]
                        st.caption(f"Using feature_names_in_ from estimator ({len(feature_names)} features).")
                    except Exception:
                        feature_names = None

            # if still None, attempt to use base_df or user_df column names if available
            if feature_names is None:
                if base_df is not None:
                    feature_names = list(base_df.columns)
                    st.caption(f"Falling back to base dataset columns ({len(feature_names)} features).")
                else:
                    st.warning("Cannot determine feature names for mapping importances — no base dataset available.")
                    continue

            # Check lengths
            if len(importances) != len(feature_names):
                st.warning(f"⚠️ Feature count mismatch for model '{key}'. Model has {len(importances)} features, but name list has {len(feature_names)}. Cannot display table.")
                continue

            fi = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values("Importance", ascending=False).head(15)

            st.dataframe(fi.reset_index(drop=True), use_container_width=True, hide_index=True)
            found_any = True
            st.divider()

        except Exception as e:
            st.error(f"Error extracting feature importance for model '{key}': {e}")

    if not found_any:
        st.info("No valid feature importance data found for any loaded model.")

# ----------------------------------------------------------
# Footer: model load status
# ----------------------------------------------------------
st.divider()
if missing_models:
    st.info(f"Missing model files: {missing_models}")
else:
    st.caption("✅ All model files loaded successfully.")
