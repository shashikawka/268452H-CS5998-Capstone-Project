import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Railway APU Anomaly Detection",
    page_icon="🚆",
    layout="wide"
)

# --------------------------------------------------
# 2. Load saved evaluation results
# --------------------------------------------------

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


try:
    (
        window_results,
        failure_results,
        three_method_results
    ) = load_results()

except (FileNotFoundError, pd.errors.EmptyDataError) as error:
    st.error(f"Unable to load dashboard data: {error}")
    st.stop()

# --------------------------------------------------
# 3. Dashboard title
# --------------------------------------------------

st.title("Railway APU Anomaly Detection Dashboard")

st.write(
    "Anomaly Detection for Predictive Maintenance "
    "of Railway Air Production Units Using "
    "Multivariate Sensor Data"
)

st.caption(
    "Dataset: MetroPT-3 | "
    "Models: Isolation Forest and Autoencoder"
)

st.divider()

# --------------------------------------------------
# 4. Sidebar navigation
# --------------------------------------------------

st.sidebar.title("Dashboard Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Model Comparison",
        "Anomaly Timeline",
        "Failure Analysis",
        "Sensor Analysis"
    ]
)

# --------------------------------------------------
# 5. Verified evaluation summary
# --------------------------------------------------

# These are the verified results from Notebook 04.
# We will replace them with CSV-derived calculations
# in the next implementation step.

total_windows = 94307

if_anomalies = 3205
ae_anomalies = 4376

if_failure_rate = 66.49
ae_failure_rate = 94.61

# --------------------------------------------------
# 6. Overview
# --------------------------------------------------

if page == "Overview":

    st.header("Project Overview")

    st.write(
        "This project investigates anomaly detection "
        "in railway Air Production Units using "
        "multivariate sensor measurements."
    )

    st.write(
        "Two unsupervised machine learning methods "
        "are evaluated: Isolation Forest and "
        "a reconstruction-based Autoencoder."
    )

    st.subheader("Evaluation Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Evaluation Windows",
        f"{total_windows:,}"
    )

    col2.metric(
        "Isolation Forest Alerts",
        f"{if_anomalies:,}"
    )

    col3.metric(
        "Autoencoder Alerts",
        f"{ae_anomalies:,}"
    )

    st.info(
        "Results are based on 60-observation windows "
        "with a stride of 10 observations."
    )

# --------------------------------------------------
# 7. Model Comparison
# --------------------------------------------------

elif page == "Model Comparison":

    st.header("Model Comparison")

    comparison_df = pd.DataFrame({
        "Model": [
            "Isolation Forest",
            "Autoencoder"
        ],
        "Flagged Windows": [
            if_anomalies,
            ae_anomalies
        ],
        "Inside-Failure Flagging Rate (%)": [
            if_failure_rate,
            ae_failure_rate
        ]
    })

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Anomaly Flagging During Documented Failures"
    )

    st.bar_chart(
        comparison_df.set_index("Model")[
            "Inside-Failure Flagging Rate (%)"
        ]
    )

    st.warning(
        "These percentages describe anomaly flagging "
        "during documented failure intervals. "
        "They do not establish early failure prediction "
        "accuracy or supervised classification recall."
    )

# --------------------------------------------------
# 8. Interactive Anomaly Timeline
# --------------------------------------------------

elif page == "Anomaly Timeline":

    st.header("Interactive Anomaly Timeline")

    timeline_df = window_results.copy()

    timeline_df["timestamp"] = pd.to_datetime(
        timeline_df["timestamp"]
    )

    selected_model = st.selectbox(
        "Select Model",
        ["Isolation Forest", "Autoencoder"]
    )

    if selected_model == "Isolation Forest":
        score_col = "if_score"
        anomaly_col = "if_anomaly"
    else:
        score_col = "ae_score"
        anomaly_col = "ae_anomaly"

    period_options = [
        str(period)
        for period in timeline_df["failure_period"]
        .dropna()
        .unique()
    ]

    selected_period = st.selectbox(
        "Select Time Period",
        ["All Evaluation Data"] + period_options
    )

    if selected_period != "All Evaluation Data":
        timeline_df = timeline_df[
            timeline_df["failure_period"].astype(str)
            == selected_period
        ]

    timeline_df = timeline_df.sort_values("timestamp")

    total = len(timeline_df)
    anomalies = int(timeline_df[anomaly_col].sum())

    col1, col2 = st.columns(2)

    col1.metric(
        "Total Windows",
        f"{total:,}"
    )

    col2.metric(
        "Flagged Windows",
        f"{anomalies:,}"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=timeline_df["timestamp"],
            y=timeline_df[score_col],
            mode="lines",
            name="Anomaly Score",
            line=dict(
                color="steelblue",
                width=1
            )
        )
    )

    anomalous_df = timeline_df[
        timeline_df[anomaly_col] == 1
    ]

    fig.add_trace(
        go.Scattergl(
            x=anomalous_df["timestamp"],
            y=anomalous_df[score_col],
            mode="markers",
            name="Flagged Windows",
            marker=dict(
                color="red",
                size=4
            )
        )
    )

    fig.update_layout(
        title=f"{selected_model} Anomaly Scores",
        xaxis_title="Timestamp",
        yaxis_title="Anomaly Score",
        hovermode="closest",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Red markers indicate windows flagged as anomalous "
        "using the saved model predictions. "
        "These are not necessarily confirmed failures."
    )

# --------------------------------------------------
# 9. Failure Analysis
# --------------------------------------------------

elif page == "Failure Analysis":

    st.header("Failure Period Analysis")

    st.info(
        "Detailed failure-period comparisons "
        "will be implemented in the next step."
    )

# --------------------------------------------------
# 10. Sensor Analysis
# --------------------------------------------------

elif page == "Sensor Analysis":

    st.header("Sensor Analysis")

    st.info(
        "Interactive sensor visualizations "
        "will be added after connecting "
        "the processed sensor data."
    )

# --------------------------------------------------
# 11. Data verification
# --------------------------------------------------

with st.expander("Data Verification"):

    st.write(
        "Window Results:",
        window_results.shape
    )

    st.write(
        "Window Columns:",
        window_results.columns.tolist()
    )

    st.write(
        "Failure Results:",
        failure_results.shape
    )

    st.write(
        "Failure Columns:",
        failure_results.columns.tolist()
    )

    st.write(
        "Three-Method Results:",
        three_method_results.shape
    )

    st.write(
        "Three-Method Columns:",
        three_method_results.columns.tolist()
    )

# --------------------------------------------------
# 12. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "CS5998 Capstone Project | "
    "Master of Data Science and AI"
)
