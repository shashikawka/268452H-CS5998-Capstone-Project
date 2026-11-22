import streamlit as st

st.set_page_config(
    page_title="Railway APU Anomaly Detection",
    page_icon="🚆",
    layout="wide"
)

st.title("Railway APU Anomaly Detection Dashboard")

st.write(
    "Anomaly Detection for Predictive Maintenance "
    "of Railway Air Production Units"
)

st.info(
    "MetroPT-3 | Isolation Forest and Autoencoder"
)

st.subheader("Project Overview")

st.write(
    "This dashboard visualizes multivariate sensor data, "
    "anomaly detection results, and documented failure "
    "periods from the MetroPT-3 dataset."
)

st.success("Dashboard initialized successfully!")
