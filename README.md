# In26-S3-CS5998-Capstone-Project

# Anomaly Detection for Predictive Maintenance of Railway Air Production Units Using Multivariate Sensor Data

## Overview

This project investigates anomaly detection in multivariate time-series sensor data from a railway Air Production Unit (APU) to support predictive maintenance.

The project uses the MetroPT-3 dataset, which contains sensor measurements collected from the Air Production Unit of a metro train. The study investigates unsupervised machine learning and deep learning approaches for identifying abnormal operating behaviour and evaluates detected anomalies against documented failure periods.

## Project Objectives

- Explore and preprocess multivariate railway sensor data.
- Analyse normal operating behaviour and documented failure periods.
- Develop an Isolation Forest baseline anomaly detection model.
- Develop and evaluate an Autoencoder-based anomaly detection model.
- Compare the approaches using anomaly detection rates and failure-period analysis.
- Investigate sensor-level reconstruction errors to understand model behaviour.
- Provide a lightweight visualization interface for sensor and anomaly analysis.

## Dataset

The project uses the **MetroPT-3 dataset**, available from the UCI Machine Learning Repository.

The dataset contains multivariate time-series measurements from a railway Air Production Unit, including pressure, temperature, motor current, and operational state variables.

The dataset includes four documented high-stress air-leak failure intervals used to evaluate anomaly detection behaviour.

The original dataset is not stored directly in this repository.

## Methodology

The project follows the main stages below:

1. **Exploratory Data Analysis (EDA)**
   - Examine sensor distributions, operational behaviour, and timestamp continuity.
   - Analyse documented failure intervals and sensor behaviour.

2. **Data Preprocessing**
   - Apply chronological training, validation, and evaluation splits.
   - Standardize continuous sensor features using training data.
   - Preserve binary operational features.
   - Generate sliding windows of 60 observations with a stride of 10, without crossing recording gaps.

3. **Isolation Forest Baseline**
   - Train an Isolation Forest model to identify unusual sensor observations.
   - Calculate anomaly scores and select a threshold using validation data.
   - Aggregate observation-level anomaly scores for window-level comparison.

4. **Autoencoder Anomaly Detection**
   - Develop a dense Autoencoder using multivariate sensor windows.
   - Train the model to reconstruct sensor observations.
   - Calculate reconstruction errors and identify anomalous windows using a validation-derived threshold.

5. **Model Evaluation and Analysis**
   - Compare anomaly detection rates within and outside documented failure intervals.
   - Investigate detection behaviour across the four failure periods.
   - Analyse sensor-level reconstruction errors during Failure 4.
   - Evaluate an alternative sensor-aware Autoencoder scoring method.
   - Compare anomaly prediction agreement between Isolation Forest and the Autoencoder.

6. **Visualization Dashboard (Planned)**
   - Develop a lightweight interface to visualize sensor behaviour, anomaly scores, and detected anomalies.

## Experimental Results

The models were evaluated using **94,307 evaluation windows**, including 2,966 windows within documented failure intervals.

Anomaly thresholds were selected using the 99th percentile of validation scores.

| Model | Anomaly Rate Within Failure Intervals | Anomaly Rate Outside Failure Intervals |
|---|---:|---:|
| Isolation Forest | 66.49% | 1.35% |
| Original Autoencoder | 94.61% | 1.72% |
| Sensor-Aware Autoencoder | 94.57% | 1.71% |

### Key Findings

- The original Autoencoder achieved the highest overall anomaly-flagging rate within documented failure intervals (94.61%).
- The Autoencoder detected almost all windows during Failures 1, 2, and 3 but did not flag any windows during Failure 4.
- Isolation Forest flagged 19.87% of Failure 4 windows, demonstrating different detection behaviour.
- The sensor-aware Autoencoder produced results similar to the original Autoencoder and did not improve Failure 4 detection.
- Isolation Forest and the original Autoencoder shared 3,142 flagged windows, with a Jaccard similarity of 0.7078.
- The original Autoencoder was selected as the primary anomaly detection approach based on its higher overall anomaly-flagging coverage during documented failure intervals.

**Evaluation Note:** The reported rates represent the proportion of windows flagged within or outside documented failure intervals. They should not be interpreted as supervised classification accuracy, confirmed false-positive rates, or verified early failure prediction performance.

## Current Progress

- [x] Project topic selected
- [x] MetroPT-3 dataset selected
- [x] Initial data understanding completed
- [x] Exploratory data analysis completed
- [x] Documented failure periods verified against the dataset
- [x] Project Definition
- [x] Data preprocessing and feature engineering
- [x] Isolation Forest baseline anomaly detection model
- [x] Autoencoder anomaly detection model
- [x] Model evaluation and failure-period comparison
- [x] Sensor-level reconstruction error analysis
- [x] Sensor-aware Autoencoder scoring experiment
- [x] Final experimental findings and limitations
- [x] Saved model and numerical results verification
- [ ] Visualization dashboard
- [ ] Final project documentation and submission

## Next Steps

- Develop a lightweight dashboard to visualize sensor measurements and anomaly detection results.
- Integrate saved model outputs into the visualization interface.
- Complete final project documentation and evaluation discussion.
- Prepare the final capstone submission.

## Tools and Technologies

- Python
- Google Colab / Jupyter Notebook
- Pandas and NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Git and GitHub

## Project Status

**In Progress**

Exploratory data analysis, preprocessing, Isolation Forest baseline development, Autoencoder development, and model evaluation have been completed.

The next development stage is the visualization dashboard.
