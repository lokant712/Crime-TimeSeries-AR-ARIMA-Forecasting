"""
inference.py
------------
Root CLI inference and forward prediction script for MDI3003 Lab 06:
Generates point forecasts and prediction intervals for specified horizons.

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


def generate_forecast(horizon_steps=26, model_path="models/arima_best_model.joblib"):
    print("=" * 60)
    print(f"MDI3003 Lab 06: Time-Series Incident Count Inference (H={horizon_steps})")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Running retrain.py...")
        from retrain import retrain_all
        retrain_all()
        
    model = joblib.load(model_path)
    forecast_obj = model.get_forecast(steps=horizon_steps)
    preds = forecast_obj.predicted_mean
    conf_80 = forecast_obj.conf_int(alpha=0.20)
    conf_95 = forecast_obj.conf_int(alpha=0.05)
    
    df_fc = pd.DataFrame({
        "step_ahead": np.arange(1, horizon_steps + 1),
        "predicted_mean_incidents": np.round(preds.values, 2),
        "conf_80_lower": np.round(conf_80.iloc[:, 0].values, 2),
        "conf_80_upper": np.round(conf_80.iloc[:, 1].values, 2),
        "conf_95_lower": np.round(conf_95.iloc[:, 0].values, 2),
        "conf_95_upper": np.round(conf_95.iloc[:, 1].values, 2)
    })
    
    print("\nForecast Results (First 10 Steps):")
    print(df_fc.head(10).to_string(index=False))
    print("\n" + "=" * 60)
    print("Inference completed successfully.")
    print("=" * 60)
    return df_fc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate incident forecasts")
    parser.add_argument("--steps", type=int, default=26, help="Forecast horizon steps")
    args = parser.parse_args()
    generate_forecast(args.steps)
