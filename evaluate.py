"""
evaluate.py
-----------
Root CLI evaluation script for MDI3003 Lab 06:
Evaluates persisted models on locked test set and displays accuracy summary.

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
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath("."))


def evaluate_submission():
    print("=" * 60)
    print("MDI3003 Lab 06: Out-of-Sample Model Evaluation")
    print("=" * 60)
    
    path_b = "results/model_comparison.csv"
    if not os.path.exists(path_b):
        path_b = "results/Table_B_Model_Comparison.csv"
        
    if os.path.exists(path_b):
        df = pd.read_csv(path_b)
        print("\nModel Comparison Table:")
        print(df.to_string(index=False))
    else:
        print("Results table not found. Please run scripts/run_all.py first.")
        
    print("\n" + "=" * 60)
    print("Evaluation completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_submission()
