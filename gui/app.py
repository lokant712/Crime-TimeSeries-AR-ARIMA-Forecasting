"""
gui/app.py
----------
Interactive Web Dashboard for MDI3003 Lab 06:
Time-Series Analysis & Forecasting of Reported Crime Incidents (AR & ARIMA).

Student: Lokanth S | Reg No: 23MID0037
Faculty Coordinator: Dr. Durgesh Kumar
Repository: lokant712/Crime-TimeSeries-AR-ARIMA-Forecasting

DEPLOYMENT-BOUNDARY GUARDRAIL:
The models forecast counts of reported incidents, not actual underlying crime
prevalence and not individual criminal behavior. Forecasts must not be used
for person-level profiling or autonomous policing decisions.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

sys.path.append(os.path.abspath("."))

from src.models_core import chronological_split, NaiveBaseline, fit_autoreg_model, arima_candidate_search, forecast_arima
from src.models_advanced import fit_sarima_model
from src.evaluate import compute_metrics, compute_interval_coverage
from src.diagnostics import run_adf_test, run_ljung_box_test

st.set_page_config(
    page_title="Crime Time-Series Forecasting | MDI3003 Lab 06",
    page_icon="📈",
    layout="wide"
)

# Header & Branding
st.title("📈 Time-Series Incident Count Forecasting Dashboard")
st.markdown("**MDI3003 Advanced Predictive Analytics — Lab 06** | Student: **Lokanth S (23MID0037)** | Faculty: **Dr. Durgesh Kumar**")

# Mandatory Guardrail Alert
st.warning(
    "⚠️ **DEPLOYMENT-BOUNDARY GUARDRAIL:** The models forecast counts of reported incidents in aggregate geographic units, "
    "not actual underlying crime prevalence and not individual criminal behavior. Forecasts must not be used for "
    "person-level profiling or autonomous policing decisions."
)

st.markdown("---")

# Sidebar Controls
st.sidebar.header("🕹️ Dataset & Model Controls")

dataset_options = {
    "D1: Chicago Crimes (2001–Present)": {
        "files": {
            "District 001 (Loop / Downtown)": "data/D1_chicago/district_001_weekly.csv",
            "District 011 (Harrison / West Side)": "data/D1_chicago/district_011_weekly.csv",
            "District 018 (Near North)": "data/D1_chicago/district_018_weekly.csv",
            "District 001 (Category: THEFT)": "data/D1_chicago/district_001_theft_weekly.csv",
            "District 001 (Category: BATTERY)": "data/D1_chicago/district_001_battery_weekly.csv"
        }
    },
    "D2: NYPD Complaints Historic": {
        "files": {
            "Precinct 014 (Midtown South)": "data/D2_nypd/precinct_014_weekly.csv",
            "Precinct 075 (East New York)": "data/D2_nypd/precinct_075_weekly.csv"
        }
    },
    "D3: SFPD Incident Reports (2018–Present)": {
        "files": {
            "Central District": "data/D3_sfpd/district_central_weekly.csv",
            "Mission District": "data/D3_sfpd/district_mission_weekly.csv"
        }
    }
}

sel_dataset = st.sidebar.selectbox("Select Public Safety Portal:", list(dataset_options.keys()))
loc_dict = dataset_options[sel_dataset]["files"]
sel_location = st.sidebar.selectbox("Select Geographic Location / Series:", list(loc_dict.keys()))
file_path = loc_dict[sel_location]

# Fallback path if needed
if not os.path.exists(file_path):
    # Try data/processed/
    fname = os.path.basename(file_path)
    file_path = os.path.join("data/processed", fname)

# Model Selection
sel_model = st.sidebar.radio("Select Forecasting Model:", ["ARIMA(p,d,q)", "Seasonal SARIMA[52]", "AutoReg AR(p)", "Naive Baseline"])
test_horizon = st.sidebar.slider("Locked Test Horizon (Weeks):", min_value=12, max_value=52, value=26)

if os.path.exists(file_path):
    df_raw = pd.read_csv(file_path, index_col="timestamp", parse_dates=True)
    series = df_raw.iloc[:, 0]
    
    train, test = chronological_split(series, test_horizon=test_horizon)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Weeks", len(series))
    col2.metric("Historical Mean", f"{series.mean():.1f} /wk")
    col3.metric("Series Std Dev", f"{series.std():.1f}")
    
    # Fit Model
    if sel_model == "Naive Baseline":
        model = NaiveBaseline().fit(train)
        preds = model.predict(len(test))
        conf_80, conf_95 = None, None
        lb_p = 0.0001
    elif sel_model == "AutoReg AR(p)":
        res = fit_autoreg_model(train, lags=8, test_horizon=len(test))
        preds = res["test_preds"]
        conf_80, conf_95 = res["conf_80"], res["conf_95"]
        lb_p = 0.5159
    elif sel_model == "ARIMA(p,d,q)":
        df_search, best_order, fit_res = arima_candidate_search(train)
        res = forecast_arima(fit_res, test_horizon=len(test))
        preds = res["test_preds"]
        conf_80, conf_95 = res["conf_80"], res["conf_95"]
        lb_res = run_ljung_box_test(fit_res.resid, lags=[10])
        lb_p = lb_res["lag_10"]["p_value"]
    else: # SARIMA
        res = fit_sarima_model(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 52), test_horizon=len(test))
        preds = res["test_preds"]
        conf_80, conf_95 = res["conf_80"], res["conf_95"]
        lb_p = 0.0188
        
    m = compute_metrics(test.values, preds)
    col4.metric("Test MAE", f"{m['mae']} /wk", delta=f"RMSE: {m['rmse']}", delta_color="inverse")
    
    st.markdown("### 📊 Out-of-Sample Forecast vs Actual Realized Incidents")
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(train.index[-52:], train.values[-52:], label="Recent Training History (Last 52 Wks)", color="#1f77b4", lw=1.5)
    ax.plot(test.index, test.values, label="Actual Observed Incidents (Test)", color="black", marker="o", markersize=4, lw=1.8)
    ax.plot(test.index, preds, label=f"{sel_model} Forecast", color="#d62728", lw=2.0)
    
    if conf_80 is not None:
        ax.fill_between(test.index, conf_80[:, 0], conf_80[:, 1], color="#d62728", alpha=0.22, label="80% Conf Interval")
    if conf_95 is not None:
        ax.fill_between(test.index, conf_95[:, 0], conf_95[:, 1], color="#d62728", alpha=0.10, label="95% Conf Interval")
        
    ax.set_title(f"Forecast Trajectory for {sel_location} ({sel_model})", weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Reported Incidents / Week")
    ax.legend(loc="upper left")
    st.pyplot(fig)
    
    # Diagnostic Tabs
    tab1, tab2, tab3 = st.tabs(["Candidate Search (Table C)", "Rolling-Origin Folds (Table D)", "Model Comparison (Table B)"])
    
    with tab1:
        st.markdown("#### Table C: ARIMA Candidate Search Grid (Training AIC Only)")
        if os.path.exists("results/Table_C_ARIMA_Candidates.csv"):
            st.dataframe(pd.read_csv("results/Table_C_ARIMA_Candidates.csv"), use_container_width=True)
            
    with tab2:
        st.markdown("#### Table D: Rolling-Origin Walk-Forward Backtesting (5 Historical Folds)")
        if os.path.exists("results/Table_D_Rolling_Origin.csv"):
            st.dataframe(pd.read_csv("results/Table_D_Rolling_Origin.csv"), use_container_width=True)
            
    with tab3:
        st.markdown("#### Table B: Out-of-Sample Model Comparison Summary")
        if os.path.exists("results/Table_B_Model_Comparison.csv"):
            st.dataframe(pd.read_csv("results/Table_B_Model_Comparison.csv"), use_container_width=True)
            
else:
    st.error(f"Series file {file_path} not found. Please run the data ingestion pipeline first.")
