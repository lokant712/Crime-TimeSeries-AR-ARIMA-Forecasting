"""
diagnostics.py
--------------
Statistical time-series diagnostics (ADF, KPSS, ACF/PACF, Ljung-Box, Decomposition).

Student: Lokanth S | Reg No: 23MID0037
Faculty Coordinator: Dr. Durgesh Kumar
Repository: lokant712/Crime-TimeSeries-AR-ARIMA-Forecasting

DEPLOYMENT-BOUNDARY GUARDRAIL:
The models forecast counts of reported incidents, not actual underlying crime
prevalence and not individual criminal behavior. Forecasts must not be used
for person-level profiling or autonomous policing decisions.
"""

import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_adf_test(series: pd.Series, maxlag: int = 12) -> dict:
    """
    Performs the Augmented Dickey-Fuller (ADF) unit-root test on training data.
    
    H0: The series possesses a unit root (non-stationary).
    H1: The series is stationary.
    """
    res = adfuller(series.dropna(), maxlag=maxlag, autolag="AIC")
    stat, pval, usedlag, nobs, crit, icbest = res
    is_stationary = pval < 0.05
    
    summary = {
        "test_statistic": float(stat),
        "p_value": float(pval),
        "used_lag": int(usedlag),
        "n_obs": int(nobs),
        "critical_values": {k: float(v) for k, v in crit.items()},
        "is_stationary": bool(is_stationary),
        "conclusion": "Reject H0 (Stationary at 5% alpha)" if is_stationary else "Fail to reject H0 (Non-stationary, unit root present)"
    }
    return summary


def run_kpss_test(series: pd.Series, regression: str = "c") -> dict:
    """
    Performs the KPSS stationarity test on training data.
    
    H0: The series is trend-stationary / level-stationary.
    H1: The series possesses a unit root.
    """
    stat, pval, lags, crit = kpss(series.dropna(), regression=regression, nlags="auto")
    is_stationary = pval >= 0.05
    return {
        "test_statistic": float(stat),
        "p_value": float(pval),
        "used_lags": int(lags),
        "critical_values": {k: float(v) for k, v in crit.items()},
        "is_stationary": bool(is_stationary),
        "conclusion": "Fail to reject H0 (Stationary)" if is_stationary else "Reject H0 (Non-stationary)"
    }


def compute_acf_pacf(series: pd.Series, nlags: int = 30) -> dict:
    """
    Computes sample Autocorrelation Function (ACF) and Partial Autocorrelation
    Function (PACF) up to nlags along with 95% Bartlett significance bounds.
    """
    acf_vals, acf_conf = acf(series.dropna(), nlags=nlags, alpha=0.05)
    pacf_vals, pacf_conf = pacf(series.dropna(), nlags=nlags, alpha=0.05, method="yw")
    
    # 95% standard error bound under null of white noise: +- 1.96 / sqrt(N)
    n = len(series.dropna())
    bound = 1.96 / np.sqrt(n)
    
    # Suggest AR lag order from PACF (highest lag with significant PACF within first 8 lags)
    sig_lags = [i for i in range(1, min(9, nlags)) if abs(pacf_vals[i]) > bound]
    suggested_ar_p = max(sig_lags) if sig_lags else 1
    
    return {
        "acf": acf_vals.tolist(),
        "pacf": pacf_vals.tolist(),
        "acf_conf": acf_conf.tolist(),
        "pacf_conf": pacf_conf.tolist(),
        "significance_bound": float(bound),
        "suggested_ar_p": int(suggested_ar_p)
    }


def run_ljung_box_test(residuals: pd.Series, lags: list = [10, 20]) -> dict:
    """
    Runs Ljung-Box test for autocorrelation in model residuals.
    
    H0: Residuals are independently distributed (white noise / no autocorrelation).
    H1: Residuals exhibit serial autocorrelation.
    """
    clean_res = residuals.dropna()
    lb_df = acorr_ljungbox(clean_res, lags=lags, return_df=True)
    
    results = {}
    for lag in lags:
        if lag in lb_df.index:
            results[f"lag_{lag}"] = {
                "stat": float(lb_df.loc[lag, "lb_stat"]),
                "p_value": float(lb_df.loc[lag, "lb_pvalue"]),
                "is_white_noise": bool(lb_df.loc[lag, "lb_pvalue"] > 0.05)
            }
    return results


def perform_seasonal_decomposition(series: pd.Series, period: int = 52, model: str = "additive"):
    """
    Performs classical additive seasonal decomposition (Trend, Seasonality, Residual).
    """
    decomp = seasonal_decompose(series, model=model, period=period, extrapolate_trend="freq")
    return decomp
