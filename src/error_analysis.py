"""
error_analysis.py
-----------------
Qualitative and quantitative review of forecast residuals, outlier weeks,
and regime shifts across municipal crime incident forecasts.

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


def identify_worst_forecast_periods(test_series: pd.Series, pred_series: np.ndarray, top_k: int = 5) -> pd.DataFrame:
    """
    Identifies the top-k time intervals with the highest absolute forecast error,
    annotating calendar attributes, seasonal context, and administrative explanations.
    """
    y_true = test_series.values
    errors = np.abs(y_true - pred_series)
    
    df_err = pd.DataFrame({
        "timestamp": test_series.index.strftime("%Y-%m-%d"),
        "actual_reported_incidents": y_true,
        "predicted_incidents": np.round(pred_series, 2),
        "absolute_error": np.round(errors, 2),
        "percentage_error_pct": np.round((errors / np.clip(y_true, 1e-3, None)) * 100.0, 2),
        "month": test_series.index.month_name(),
        "is_month_start": test_series.index.is_month_start
    })
    
    df_worst = df_err.sort_values(by="absolute_error", ascending=False).head(top_k).reset_index(drop=True)
    
    # Qualitative contextual diagnosis
    annotations = []
    for _, row in df_worst.iterrows():
        month = row["month"]
        if month in ["June", "July", "August"]:
            annotations.append("Early summer surge: accelerated outdoor mobility exceeding linear autoregressive dampening.")
        elif month in ["December", "January"]:
            annotations.append("Holiday seasonality / end-of-year administrative reporting batching.")
        elif month in ["March", "April"]:
            annotations.append("Spring transition: volatile weather shifts influencing citizen reporting cadence.")
        else:
            annotations.append("Irregular short-term variation within expected confidence bounds.")
            
    df_worst["qualitative_context"] = annotations
    return df_worst


def analyze_structural_break_residuals(residuals: pd.Series, break_date: str = "2020-03-15") -> dict:
    """
    Computes split-regime variance and mean error before and after a structural break.
    """
    break_ts = pd.Timestamp(break_date)
    pre_break = residuals.loc[:break_ts].dropna()
    post_break = residuals.loc[break_ts:].dropna()
    
    return {
        "pre_break_mean_error": float(np.mean(pre_break)),
        "pre_break_std_error": float(np.std(pre_break)),
        "post_break_mean_error": float(np.mean(post_break)),
        "post_break_std_error": float(np.std(post_break)),
        "variance_ratio_post_vs_pre": float(np.var(post_break) / max(np.var(pre_break), 1e-6))
    }
