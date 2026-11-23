import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

# -----------------------------------------------------
# Load saved model evaluation results
# -----------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"

@st.cache_data
def load_results():
    window_results = pd.read_csv(
        DATA_DIR / "model_comparison_windows.csv"
    )

    failure_results = pd.read_csv(
        DATA_DIR / "if_ae_failure_comparison.csv"
    )

    three_method_results = pd.read_csv(
        DATA_DIR / "three_method_failure_comparison.csv"
    )

    return (
        window_results,
        failure_results,
        three_method_results
    )


window_results, failure_results, three_method_results = load_results()

# --------------------------------------------------
# Temporary data verification
# --------------------------------------------------

with st.expander("Data Verification"):
    st.write("Window Results:", window_results.shape)
    st.write("Window Columns:", window_results.columns.tolist())

    st.write("Failure Results:", failure_results.shape)
    st.write("Failure Columns:", failure_results.columns.tolist())

    st.write("Three-Method Results:", three_method_results.shape)
    st.write(
        "Three-Method Columns:",
        three_method_results.columns.tolist()
    )