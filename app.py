# ==========================================================
# Recruitment Efficiency Dashboard
# Author: NeuraLens
# ==========================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------
st.set_page_config(
    page_title="Recruitment Efficiency Predictor (FEv3)",
    page_icon="💼",
    layout="wide"
)

st.title("Recruitment Efficiency Predictor")
st.markdown("""
This dashboard predicts **Time to Hire**, **Cost per Hire**, and **Offer Acceptance Rate**  
based on your recruitment process data.

> It represents the most stable and well-balanced model — optimized for accuracy, fairness, and business efficiency.
""")

# ----------------------------------------------------------
# Load Models
# ----------------------------------------------------------
MODEL_DIR = "./"
@st.cache_resource
def load_models():
    models = {}
    try:
        models["time_to_hire_days"] = joblib.load(os.path.join(MODEL_DIR, "model_time_to_hire_days_FEv3.pkl"))
        models["cost_per_hire"] = joblib.load(os.path.join(MODEL_DIR, "model_cost_per_hire_FEv3.pkl"))
        models["offer_acceptance_rate"] = joblib.load(os.path.join(MODEL_DIR, "model_offer_acceptance_rate_FEv3.pkl"))
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

models = load_models()
if not models:
    st.stop()

# ==========================================================
# TAB NAVIGATION
# ==========================================================
tabs = st.tabs(["Prediction", "Business Simulation", "Department & Source Analysis"])

# ----------------------------------------------------------
# TAB 1: PREDICTION
# ----------------------------------------------------------
with tabs[0]:
    st.subheader("Recruitment Input Form")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            department = st.selectbox("Department", ["Engineering", "Product", "HR", "Sales", "Marketing", "Finance"])
            num_applicants = st.number_input("Number of Applicants", min_value=1, max_value=1000, value=150)
            process_efficiency = st.slider("Process Efficiency", 0.0, 1.0, 0.7)
            cost_intensity = st.slider("Cost Intensity", 0.0, 1.0, 0.5)
            engagement_score = st.slider("Engagement Score", 0.0, 1.0, 0.6)
        with col2:
            dept_efficiency = st.slider("Department Efficiency", 0.0, 1.0, 0.8)
            candidate_satisfaction = st.slider("Candidate Satisfaction", 0.0, 1.0, 0.7)
            offer_readiness = st.slider("Offer Readiness", 0.0, 1.0, 0.75)
            source = st.selectbox("Source of Candidate", ["Referral", "LinkedIn", "Recruiter", "Job Portal"])

        submitted = st.form_submit_button("Run Prediction")

    if submitted:
        # Prepare input data
        input_data = pd.DataFrame([{
            "department": department,
            "source": source,
            "num_applicants": num_applicants,
            "process_efficiency": process_efficiency,
            "cost_intensity": cost_intensity,
            "engagement_score": engagement_score,
            "dept_efficiency": dept_efficiency,
            "candidate_satisfaction": candidate_satisfaction,
            "offer_readiness": offer_readiness
        }])

        try:
            # Run predictions
            time_pred = max(0, models["time_to_hire_days"].predict(input_data)[0])
            cost_pred = max(0, models["cost_per_hire"].predict(input_data)[0])
            offer_pred = np.clip(models["offer_acceptance_rate"].predict(input_data)[0] * 100, 0, 100)

            st.success("Prediction completed successfully!")
            col1, col2, col3 = st.columns(3)
            col1.metric("Time to Hire (days)", f"{time_pred:.1f}")
            col2.metric("Cost per Hire ($)", f"{cost_pred:,.2f}")
            col3.metric("Offer Acceptance Rate", f"{offer_pred:.1f}%")

            st.session_state["baseline"] = {
                "time": time_pred,
                "cost": cost_pred,
                "offer": offer_pred
            }

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ----------------------------------------------------------
# TAB 2: BUSINESS SIMULATION
# ----------------------------------------------------------
with tabs[1]:
    st.subheader("Business Impact Simulation (Optimized Scenario)")

    if "baseline" not in st.session_state:
        st.info("Please run a prediction first to enable business simulation.")
    else:
        st.write("""
        This simulation assumes the following improvements:
        - +10% Offer Readiness  
        - +15% Engagement Score  
        - +10% Department Efficiency
        """)

        baseline = st.session_state["baseline"]

        # Apply optimization scenario
        df_opt = pd.DataFrame([{
            "department": department,
            "source": source,
            "num_applicants": num_applicants,
            "process_efficiency": process_efficiency,
            "cost_intensity": cost_intensity,
            "engagement_score": engagement_score * 1.15,
            "dept_efficiency": dept_efficiency * 1.10,
            "candidate_satisfaction": candidate_satisfaction,
            "offer_readiness": offer_readiness * 1.10
        }])

        try:
            opt_time = max(0, models["time_to_hire_days"].predict(df_opt)[0])
            opt_cost = max(0, models["cost_per_hire"].predict(df_opt)[0])
            opt_offer = np.clip(models["offer_acceptance_rate"].predict(df_opt)[0] * 100, 0, 100)

            delta_t = baseline["time"] - opt_time
            delta_c = baseline["cost"] - opt_cost
            delta_o = opt_offer - baseline["offer"]

            st.write("**Baseline vs Optimized Comparison:**")
            impact_data = pd.DataFrame({
                "Metric": ["Hiring Duration (days)", "Cost per Hire ($)", "Offer Acceptance (%)"],
                "Baseline": [baseline["time"], baseline["cost"], baseline["offer"]],
                "Optimized": [opt_time, opt_cost, opt_offer],
                "Change": [delta_t, delta_c, delta_o]
            }).round(2)

            st.dataframe(impact_data, use_container_width=True)
            st.caption("💡 The optimized scenario indicates measurable improvements in time and cost efficiency.")

        except Exception as e:
            st.error(f"Simulation failed: {e}")

# ----------------------------------------------------------
# TAB 3: DEPARTMENT & SOURCE ANALYSIS
# ----------------------------------------------------------
with tabs[2]:
    st.subheader("Department & Source Efficiency Analysis")

    try:
        df = pd.read_csv("deployment_retrain_summary_FEv3.csv")

        dept_summary = df.groupby("department")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate"]].mean().round(2)
        src_summary = df.groupby("source")[["time_to_hire_days", "cost_per_hire", "offer_acceptance_rate"]].mean().round(2)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📂 Efficiency by Department")
            st.dataframe(dept_summary.style.background_gradient(cmap="Greens"))
        with col2:
            st.markdown("### 🌐 Efficiency by Source")
            st.dataframe(src_summary.style.background_gradient(cmap="Blues"))

        st.success(f"Most efficient department: **{dept_summary['time_to_hire_days'].idxmin()}**")
        st.success(f"Lowest cost source: **{src_summary['cost_per_hire'].idxmin()}**")
        st.success(f"Best acceptance source: **{src_summary['offer_acceptance_rate'].idxmax()}**")

    except Exception as e:
        st.warning(f"Unable to load group analysis: {e}")
