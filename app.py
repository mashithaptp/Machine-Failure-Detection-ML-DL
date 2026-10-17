"""Streamlit interface for the saved AI4I 2020 machine-failure classifier.

Run from the project directory:
    python -m streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Machine Failure Prediction", page_icon="⚙️", layout="centered")

MODEL_PATH = Path(__file__).resolve().parent / "machine_failure_detection_model.joblib"


@st.cache_resource
def load_artifact(path: str):
    return joblib.load(path)


st.title("⚙️ Machine Failure Prediction")
st.write("Enter machine operating measurements to estimate the likelihood of machine failure.")
st.caption("AI4I 2020 Predictive Maintenance Dataset | Machine Learning model")

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Place 'machine_failure_detection_model.joblib' "
        "in the same folder as app.py."
    )
    st.stop()

try:
    artifact = load_artifact(str(MODEL_PATH))
    model = artifact["model"]
    threshold = float(artifact["threshold"])
    feature_columns = artifact["feature_columns"]
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

with st.form("prediction_form"):
    st.subheader("Machine measurements")
    machine_type = st.selectbox("Product type", options=["L", "M", "H"], index=1)

    col1, col2 = st.columns(2)
    with col1:
        air_temp = st.number_input("Air temperature (K)", min_value=250.0, max_value=350.0, value=300.0, step=0.1)
    with col2:
        process_temp = st.number_input("Process temperature (K)", min_value=250.0, max_value=350.0, value=310.0, step=0.1)

    speed = st.slider("Rotational speed (rpm)", min_value=1000, max_value=3000, value=1500, step=10)
    torque = st.slider("Torque (Nm)", min_value=0.0, max_value=80.0, value=40.0, step=0.5)
    wear = st.slider("Tool wear (min)", min_value=0, max_value=300, value=120, step=1)
    submitted = st.form_submit_button("Predict machine failure", type="primary", use_container_width=True)

if submitted:
    # Use the exact raw input names and derived-feature formulas from the notebook.
    reading = {
        "Type": machine_type,
        "Air temperature [K]": float(air_temp),
        "Process temperature [K]": float(process_temp),
        "Rotational speed [rpm]": int(speed),
        "Torque [Nm]": float(torque),
        "Tool wear [min]": int(wear),
    }
    input_df = pd.DataFrame([reading])
    input_df["Temperature difference [K]"] = (
        input_df["Process temperature [K]"] - input_df["Air temperature [K]"]
    )
    input_df["Mechanical power proxy"] = (
        input_df["Torque [Nm]"] * input_df["Rotational speed [rpm]"]
    )
    input_df = input_df.reindex(columns=feature_columns)

    if input_df.isna().any().any():
        st.error("The saved model expects features that are missing from the input. Check the notebook's feature engineering.")
        st.stop()

    try:
        failure_probability = float(model.predict_proba(input_df)[0, 1])
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    predicted_failure = failure_probability >= threshold
    st.subheader("Prediction result")
    if predicted_failure:
        st.error("⚠️ Predicted class: Machine failure")
    else:
        st.success("✅ Predicted class: No failure")

    st.metric("Estimated failure probability", f"{failure_probability:.2%}")
    st.progress(min(max(failure_probability, 0.0), 1.0))
    st.caption(f"Decision threshold selected during validation: {threshold:.2%}")

    with st.expander("View input measurements"):
        # Convert the DISPLAY table to text so Streamlit/Arrow does not fail
        # on a column mixing product-type strings with numeric measurements.
        display_df = pd.DataFrame(
            [(name, str(value)) for name, value in reading.items()],
            columns=["Measurement", "Input value"],
        )
        st.dataframe(display_df, hide_index=True, use_container_width=True)

st.divider()
st.caption(
    "Educational prototype only. AI4I 2020 is a synthetic dataset; "
    "predictions should not be used as the sole basis for real maintenance decisions."
)
