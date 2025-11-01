# ==========================================================
# Recruitment Efficiency Insight Dashboard
# ==========================================================
# Author: NeuraLens
# Purpose: Dnified Streamlit Dashboard for Analytical & Predictive Insight
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Recruitment Efficiency Dashboard", layout="wide")
st.title("Recruitment Efficiency Insight Dashboard")

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("final_recruitment_data_FEv3.csv")
    return df

df = load_data()

# ----------------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        models = {
            "time": joblib.load("model_time_to_hire_days_FEv3.pkl"),
            "cost": joblib.load("model_cost_per_hire_FEv3.pkl"),
            "offer": joblib.load("model_offer_acceptance_rate_FEv3.pkl")
        }
        return models
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

models = load_models()

# ----------------------------------------------------------
# FUNCTION — Compute Efficiency
# ----------------------------------------------------------
def compute_efficiency(df):
    df = df.copy()
    df['time_score'] = 1 - (df['time_to_hire_days'] - df['time_to_hire_days'].min()) / (df['time_to_hire_days'].max() - df['time_to_hire_days'].min())
    df['cost_score'] = 1 - (df['cost_per_hire'] - df['cost_per_hire'].min()) / (df['cost_per_hire'].max() - df['cost_per_hire'].min())
    df['accept_score'] = (df['offer_acceptance_rate'] - df['offer_acceptance_rate'].min()) / (df['offer_acceptance_rate'].max() - df['offer_acceptance_rate'].min())
    df['efficiency_score'] = 0.4 * df['time_score'] + 0.3 * df['cost_score'] + 0.3 * df['accept_score']
    return df

df = compute_efficiency(df)

# ----------------------------------------------------------
# CREATE TABS
# ----------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Executive Summary",
    "Department Efficiency",
    "Source Efficiency",
    "Job Role Efficiency",
    "Top 10 Most Efficient Recruitments",
    "Batch Prediction"
])

# ==========================================================
# TAB 1-5: Existing
# ==========================================================
with tab1:
    st.header("Recruitment KPI Scorecard — Executive Overview")

    avg_time = round(df['time_to_hire_days'].mean())
    avg_cost = round(df['cost_per_hire'].mean())
    avg_accept = round(df['offer_acceptance_rate'].mean() * 100)

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Time to Hire (days)", f"{avg_time}")
    col2.metric("Average Cost per Hire ($)", f"{avg_cost:,}")
    col3.metric("Offer Acceptance Rate (%)", f"{avg_accept}%")

    st.divider()

    st.subheader("The Most Efficient Department")
    best_dept = (
        df.groupby("department")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]]
        .mean()
        .sort_values("efficiency_score", ascending=False)
    )
    st.dataframe(best_dept.head(1).reset_index(), use_container_width=True, hide_index=True)

    st.subheader("The Most Efficient Source")
    best_source = (
        df.groupby("source")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]]
        .mean()
        .sort_values("efficiency_score", ascending=False)
    )
    st.dataframe(best_source.head(1).reset_index(), use_container_width=True, hide_index=True)

    st.subheader("The Most Efficient Job Title")
    best_role = (
        df.groupby("job_title")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]]
        .mean()
        .sort_values("efficiency_score", ascending=False)
    )
    st.dataframe(best_role.head(1).reset_index(), use_container_width=True, hide_index=True)

with tab2:
    st.header("Department Efficiency Overview")
    dept_summary = df.groupby("department")[["time_to_hire_days","cost_per_hire","offer_acceptance_rate","efficiency_score"]].mean().sort_values("efficiency_score", ascending=False)
    st.dataframe(dept_summary.reset_index(), use_container_width=True)

with tab3:
    st.header("Source Efficiency Overview")
    source_summary = df.groupby("source")[["time_to_hire_days","cost_per_hire","offer_acceptance_rate","efficiency_score"]].mean().sort_values("efficiency_score", ascending=False)
    st.dataframe(source_summary.reset_index(), use_container_width=True)

with tab4:
    st.header("Job Role Efficiency Overview")
    job_summary = df.groupby("job_title")[["time_to_hire_days","cost_per_hire","offer_acceptance_rate","efficiency_score"]].mean().sort_values("efficiency_score", ascending=False)
    st.dataframe(job_summary.reset_index(), use_container_width=True)

with tab5:
    st.header("Top 10 Most Efficient Recruitments")
    top10 = df.sort_values("efficiency_score", ascending=False).head(10)
    top10_display = top10[
        ["department", "source", "job_title", "time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]
    ]
    st.dataframe(top10_display.reset_index(drop=True), use_container_width=True, hide_index=True)

# ==========================================================
# TAB 6: BATCH PREDICTION
# ==========================================================
with tab_predict:
    st.header("Prediction / Batch Analysis")
    st.markdown(
        """
        Upload CSV berisi baris rekrutmen (bisa mentah atau sudah melalui feature engineering).
        Jika model `.pkl` (FEv3) tersedia di repo, aplikasi akan mencoba memprediksi
        Time to Hire, Cost per Hire, dan Offer Acceptance Rate.
        Jika model tidak tersedia, aplikasi akan menampilkan ringkasan dan skor efisiensi (dihitung).
        """
    )

    uploaded = st.file_uploader("Upload recruitment CSV (format CSV)", type=["csv"])
    if uploaded is None and base_df is None:
        st.info("Anda belum mengupload file dan dataset default tidak tersedia.")
    else:
        if uploaded is not None:
            try:
                user_df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Gagal membaca CSV: {e}")
                st.stop()
        else:
            user_df = base_df.copy()

        st.subheader("Data Preview")
        st.dataframe(user_df.head(20), use_container_width=True)

        # compute local efficiency if possible
        user_df_proc = user_df.copy()
        user_df_proc = compute_efficiency(user_df_proc)

        # Try to predict if models available
        if models_available:
            # Attempt predictions for each model that is loaded
            preds = {}
            pred_errors = {}
            for key in ["time", "cost", "offer"]:
                model = models.get(key)
                if model is None:
                    preds[key] = None
                    continue
                try:
                    # Some scikit pipelines accept a DataFrame directly (good).
                    # We'll call predict and catch exceptions.
                    ypred = model.predict(user_df.copy())
                    preds[key] = np.array(ypred).reshape(-1)
                except Exception as e:
                    # try to give helpful message: compare model.feature_names_in_ if present
                    expected = None
                    try:
                        expected = getattr(model, "feature_names_in_", None)
                    except Exception:
                        expected = None
                    pred_errors[key] = {
                        "error": str(e),
                        "expected_features": expected.tolist() if isinstance(expected, (list, np.ndarray)) else expected,
                    }
                    preds[key] = None

            # Check if any prediction succeeded
            any_success = any(v is not None for v in preds.values())
            if not any_success:
                st.warning("Model terdeteksi tetapi prediksi gagal untuk semua target. Detail error:")
                for k, v in pred_errors.items():
                    st.write(f"- {k}: {v['error']}")
                    if v["expected_features"]:
                        st.write(f"  Expected model features: {v['expected_features']}")
                st.info("Silakan pastikan CSV mengandung fitur yang dibutuhkan model, atau gunakan dataset hasil feature-engineering.")
            else:
                # attach predictions (clip to sensible ranges)
                n = len(user_df_proc)
                if preds.get("time") is not None:
                    time_pred = np.clip(preds["time"], a_min=0, a_max=None)
                else:
                    time_pred = np.full(n, np.nan)

                if preds.get("cost") is not None:
                    cost_pred = np.clip(preds["cost"], a_min=0, a_max=None)
                else:
                    cost_pred = np.full(n, np.nan)

                if preds.get("offer") is not None:
                    offer_pred = np.clip(preds["offer"], a_min=0.0, a_max=1.0)
                else:
                    offer_pred = np.full(n, np.nan)

                user_df_proc["pred_time_to_hire_days"] = time_pred
                user_df_proc["pred_cost_per_hire"] = cost_pred
                user_df_proc["pred_offer_acceptance_rate"] = offer_pred

                # show concise results table (only important cols)
                key_cols = []
                for c in ["recruitment_id", "department", "source", "job_title"]:
                    if c in user_df_proc.columns:
                        key_cols.append(c)
                # add preds
                key_cols += ["pred_time_to_hire_days", "pred_cost_per_hire", "pred_offer_acceptance_rate"]
                available_key_cols = [c for c in key_cols if c in user_df_proc.columns]

                st.success("Prediction completed successfully.")
                # format numeric columns for display
                display_df = user_df_proc.copy()
                if "pred_time_to_hire_days" in display_df.columns:
                    display_df["pred_time_to_hire_days"] = display_df["pred_time_to_hire_days"].round(1)
                if "pred_cost_per_hire" in display_df.columns:
                    display_df["pred_cost_per_hire"] = display_df["pred_cost_per_hire"].round(2)
                if "pred_offer_acceptance_rate" in display_df.columns:
                    display_df["pred_offer_acceptance_rate"] = (display_df["pred_offer_acceptance_rate"] * 100).round(1).astype(str) + "%"

                st.subheader("Prediction Results (sample)")
                st.dataframe(display_df[available_key_cols].head(50).reset_index(drop=True), use_container_width=True, hide_index=True)

                # summary statistics
                st.markdown("---")
                st.subheader("Summary Statistics (Predictions)")
                cols_for_summary = {
                    "Avg Predicted Time to Hire (days)": "pred_time_to_hire_days",
                    "Avg Predicted Cost per Hire ($)": "pred_cost_per_hire",
                    "Avg Predicted Offer Acceptance Rate (%)": "pred_offer_acceptance_rate",
                }
                col_vals = {}
                for label, col in cols_for_summary.items():
                    if col in user_df_proc.columns and user_df_proc[col].notna().any():
                        if "offer" in col:
                            # we stored as percent string earlier; use original numeric array
                            if preds.get("offer") is not None:
                                avg_offer = np.nanmean(offer_pred) * 100
                                col_vals[label] = f"{avg_offer:.1f}%"
                            else:
                                col_vals[label] = "-"
                        else:
                            avgv = np.nanmean(user_df_proc[col])
                            if "Cost" in label:
                                col_vals[label] = f"{int(round(avgv)):,}"
                            else:
                                col_vals[label] = f"{avgv:.1f}"
                    else:
                        col_vals[label] = "-"

                s1, s2, s3 = st.columns(3)
                s1.metric("Avg Predicted Time to Hire (days)", col_vals[list(col_vals.keys())[0]])
                s2.metric("Avg Predicted Cost per Hire ($)", col_vals[list(col_vals.keys())[1]])
                s3.metric("Avg Predicted Offer Acceptance Rate (%)", col_vals[list(col_vals.keys())[2]])

        else:
            # models not available: show computed efficiency and allow download
            st.info("Model .pkl tidak ditemukan di repository. Menampilkan analisis berbasis dataset (computed efficiency).")
            st.subheader("Computed efficiency (sample)")
            disp_cols = ["department", "source", "job_title", "time_to_hire_days", "cost_per_hire", "offer_acceptance_rate", "efficiency_score"]
            available = [c for c in disp_cols if c in user_df_proc.columns]
            st.dataframe(user_df_proc[available].sort_values("efficiency_score", ascending=False).reset_index(drop=True).head(50), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("Export")
            csv = user_df_proc.to_csv(index=False).encode("utf-8")
            st.download_button("Download processed CSV (with efficiency_score)", data=csv, file_name="predictions_with_efficiency.csv", mime="text/csv")

# ----------------------------
# Footer: show missing model files if any
# ----------------------------
st.markdown("---")
if missing_model_files:
    st.info(f"Model files not found in repository: {missing_model_files}. Jika ingin prediksi otomatis, upload file .pkl ke repo (nama sesuai): {list(MODEL_FILES.values())}")
else:
    if not models_available:
        st.warning("Model files ada tetapi gagal dimuat (file ada namun terjadi error load). Cek log build / versi scikit-learn/xgboost.")
