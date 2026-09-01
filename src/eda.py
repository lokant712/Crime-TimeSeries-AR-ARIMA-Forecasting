"""
eda.py
------
Exploratory time-series visualization and summary statistical profiling.

Student: Lokanth S | Reg No: 23MID0037
Faculty Coordinator: Dr. Durgesh Kumar
Repository: lokant712/Crime-TimeSeries-AR-ARIMA-Forecasting

DEPLOYMENT-BOUNDARY GUARDRAIL:
The models forecast counts of reported incidents, not actual underlying crime
prevalence and not individual criminal behavior. Forecasts must not be used
for person-level profiling or autonomous policing decisions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_exploratory_summary(series: pd.Series) -> pd.DataFrame:
    """
    Computes standard summary statistics for regular time series.
    """
    return pd.DataFrame([{
        "Total_Periods": len(series),
        "Start_Date": series.index.min().strftime("%Y-%m-%d"),
        "End_Date": series.index.max().strftime("%Y-%m-%d"),
        "Mean_Incidents_Per_Week": round(float(series.mean()), 2),
        "Median_Incidents_Per_Week": round(float(series.median()), 2),
        "Std_Dev": round(float(series.std()), 2),
        "Min_Count": int(series.min()),
        "Max_Count": int(series.max()),
        "Skewness": round(float(series.skew()), 3),
        "Kurtosis": round(float(series.kurtosis()), 3)
    }])
