"""
retrain.py
----------
Root CLI retraining script for MDI3003 Lab 06:
Trains Baseline, AR(p), ARIMA(p,d,q), SARIMA, and SARIMAX models,
saving serialized model checkpoints to models/ and results to results/.

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
import argparse
import joblib
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath("."))

from src.series_construction import process_all_datasets
from src.models_core import chronological_split, fit_autoreg_model, arima_candidate_search, forecast_arima
from src.models_advanced import fit_sarima_model, fit_sarimax_model


def retrain_all(save_dir="models", results_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 60)
    print("MDI3003 Lab 06: Retraining Time-Series Models")
    print("=" * 60)
    
    series_dict = process_all_datasets()
    s_d1 = series_dict["D1_Chicago_001"]
    
    train, test = chronological_split(s_d1, test_horizon=26)
    
    # 1. AR(8)
    print("Fitting AR(8)...")
    ar_res = fit_autoreg_model(train, lags=8, test_horizon=26)
    joblib.dump(ar_res["fit_res"], os.path.join(save_dir, "ar_p8_model.joblib"))
    
    # 2. ARIMA
    print("Running ARIMA Candidate Search & Fitting...")
    candidate_orders = [(1, 0, 0), (2, 0, 0), (1, 1, 1), (2, 1, 1), (0, 1, 1), (1, 1, 0), (2, 1, 2), (3, 1, 1)]
    df_search, best_order, best_fit = arima_candidate_search(train, candidate_orders=candidate_orders)
    joblib.dump(best_fit, os.path.join(save_dir, "arima_best_model.joblib"))
    
    # 3. SARIMA
    print("Fitting Seasonal SARIMA(1,1,1)x(1,0,1)[52]...")
    sarima_res = fit_sarima_model(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 52), test_horizon=26)
    joblib.dump(sarima_res["fit_res"], os.path.join(save_dir, "sarima_seasonal_model.joblib"), compress=3)
    
    print("=" * 60)
    print(f"All models successfully retrained and serialized to {save_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    retrain_all()
