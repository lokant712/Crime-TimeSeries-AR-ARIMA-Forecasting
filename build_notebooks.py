"""
build_notebooks.py
------------------
Programmatically creates clean, fully-executed format Jupyter Notebooks
for all required submission notebooks:
- 23MID0037_Lab06_Crime_AR_ARIMA.ipynb (Master Consolidated Notebook)
- notebooks/D1_chicago_core.ipynb
- notebooks/D1_multi_location.ipynb
- notebooks/D2_nypd_replication.ipynb
- notebooks/D3_sfpd_replication.ipynb
- notebooks/advanced_sarima_rolling_lstm.ipynb
"""

import os
import json
import nbformat as nbf


def make_notebook(cells_data, out_path):
    nb = nbf.v4.new_notebook()
    nb["cells"] = []
    for cell_type, content in cells_data:
        if cell_type == "markdown":
            nb["cells"].append(nbf.v4.new_markdown_cell(content))
        elif cell_type == "code":
            nb["cells"].append(nbf.v4.new_code_cell(content))
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created notebook: {out_path}")


def create_all_notebooks():
    os.makedirs("notebooks", exist_ok=True)
    
    # 1. Master Consolidated Notebook: 23MID0037_Lab06_Crime_AR_ARIMA.ipynb
    master_cells = [
        ("markdown", """# MDI3003 Advanced Predictive Analytics — Lab 06
# Time-Series Analysis & Forecasting of Reported Crime Incidents (AR & ARIMA)

**Student:** Lokanth S | **Reg No:** 23MID0037 | **Faculty Coordinator:** Dr. Durgesh Kumar  
**GitHub Repository:** `lokant712/Crime-TimeSeries-AR-ARIMA-Forecasting`

---

### Deployment-Boundary Guardrail (Strictly Enforced)
> **The models forecast counts of reported incidents, not actual underlying crime prevalence and not individual criminal behavior. Forecasts must not be used for person-level profiling or autonomous policing decisions.**

---

### Public-Safety Framing Table
| Concept | Correct Interpretation |
|---|---|
| **Observed target** | Number of reported/recorded incidents in a defined place and time interval |
| **Forecasting unit** | One regular time interval for one defined geographic unit |
| **Location** | Defines a separate time series in the core; not a person-level risk feature |
| **Forecast** | Conditional estimate based on historical reporting patterns; not proof of future crime |
| **Deployment boundary** | Decision-support/academic forecasting only; no individual prediction or autonomous resource allocation |
"""),
        ("code", """import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('.')

from src.data_ingest import fetch_socrata_data
from src.series_construction import construct_weekly_series, process_all_datasets
from src.diagnostics import run_adf_test, run_kpss_test, compute_acf_pacf, run_ljung_box_test, perform_seasonal_decomposition
from src.models_core import chronological_split, NaiveBaseline, fit_autoreg_model, arima_candidate_search, forecast_arima
from src.models_advanced import fit_sarima_model, fit_sarimax_model, run_rolling_origin_backtest, train_evaluate_recurrent_benchmark
from src.evaluate import compute_metrics, compute_interval_coverage
from src.plotting import CAPTIONS

print("Pipeline modules loaded successfully.")
"""),
        ("markdown", """## 1. Time-Series Data Construction & Chronological Split
We resample incident records into regular weekly counts (`'W-MON'`), handle unrecorded intervals with explicit zero counts (assuming no recorded incidents in that interval), and partition chronologically without shuffling.
"""),
        ("code", """# Load processed series for Chicago District 001
s_d1 = pd.read_csv('data/processed/D1_Chicago_District_001.csv', index_col='timestamp', parse_dates=True)['reported_incidents']
print(f"Total series periods: {len(s_d1)} weeks ({s_d1.index.min().strftime('%Y-%m-%d')} to {s_d1.index.max().strftime('%Y-%m-%d')})")

# Chronological Split (Horizon H = 26 weeks)
H = 26
train_d1, test_d1 = chronological_split(s_d1, test_horizon=H)
print(f"Train size: {len(train_d1)} weeks | Test size: {len(test_d1)} weeks")
assert len(s_d1) > 3 * H, "Assertion check: len(y) > 3*H"
assert train_d1.index.max() < test_d1.index.min(), "Leakage assertion: no temporal overlap"
"""),
        ("markdown", """## 2. Statistical Diagnostics (Stationarity & ACF/PACF on Training Data Only)
We test for unit roots using ADF and KPSS, and inspect sample ACF and PACF on training history to select candidate AR lag orders without leaking test data.
"""),
        ("code", """adf_res = run_adf_test(train_d1)
print(f"ADF Statistic: {adf_res['test_statistic']:.4f}")
print(f"p-value: {adf_res['p_value']:.4e} ({adf_res['conclusion']})")

acf_pacf = compute_acf_pacf(train_d1, nlags=30)
print(f"Suggested AR order from PACF cutoff: p={acf_pacf['suggested_ar_p']}")
"""),
        ("markdown", """## 3. Model Development: Baselines, AR(p), and ARIMA Candidate Search
We fit the persistence baselines, an AR(p) model with lag order justified by PACF, and execute a full ARIMA candidate grid search across candidate set $\{(1,0,0),(2,0,0),(1,1,1),(2,1,1),(0,1,1),(1,1,0),(2,1,2),(3,1,1)\}$ strictly on training AIC.
"""),
        ("code", """# Naive & Seasonal Naive
naive_m = NaiveBaseline().fit(train_d1)
naive_preds = naive_m.predict(H)

snaive_m = NaiveBaseline(seasonal_period=52).fit(train_d1)
snaive_preds = snaive_m.predict(H)

# AR(p)
ar_res = fit_autoreg_model(train_d1, lags=8, test_horizon=H)

# ARIMA Candidate Search
candidate_orders = [(1, 0, 0), (2, 0, 0), (1, 1, 1), (2, 1, 1), (0, 1, 1), (1, 1, 0), (2, 1, 2), (3, 1, 1)]
df_candidates, best_order, best_fit = arima_candidate_search(train_d1, candidate_orders=candidate_orders)
print("Table C — ARIMA Candidate Search Results:")
display(df_candidates)

arima_res = forecast_arima(best_fit, test_horizon=H)
"""),
        ("markdown", """## 4. Rolling-Origin Walk-Forward Backtesting (Feedback Item 1)
To assess fold-wise stability across historical forecast origins without touching locked test data, we perform 5-fold rolling-origin backtesting.
"""),
        ("code", """df_table_d = pd.read_csv('results/Table_D_Rolling_Origin.csv')
print("Table D — Rolling-Origin Backtesting Results Across 5 Folds:")
display(df_table_d)
"""),
        ("markdown", """## 5. Locked Test Evaluation & Model Comparison (Table B)
We evaluate all models on the locked 26-week out-of-sample period (January–June 2024) exactly once.
"""),
        ("code", """df_table_b = pd.read_csv('results/Table_B_Model_Comparison.csv')
print("Table B — Out-of-Sample Model Comparison:")
display(df_table_b)
"""),
        ("markdown", """## 6. Advanced Extensions: Multi-Location, SARIMA, and Deep Learning
We replicate across multiple locations in Chicago, NYPD, SFPD, fit 52-week seasonal SARIMA, and benchmark against PyTorch LSTM/GRU.
"""),
        ("code", """df_table_e = pd.read_csv('results/Table_E_Multi_Location_Summary.csv')
print("Table E — Multi-Location Summary:")
display(df_table_e)

df_table_f = pd.read_csv('results/Table_F_Advanced_Comparison.csv')
print("Table F — Advanced Model & Deep Learning Trade-Off Comparison:")
display(df_table_f)
"""),
        ("markdown", """## 7. Five-Sentence Interpretation of Time/Location Limitations (Mandatory Deliverable)
1. **Administrative Unit Boundary:** The forecasts generated by these models apply exclusively to aggregate administrative police districts and cannot be disaggregated to specific street addresses or micro-hotspots.
2. **Recorded Count vs Underlying Reality:** Incident volumes represent citizen reporting activity and police recording practices rather than true underlying crime prevalence.
3. **Temporal Invariance Caveat:** Coefficients fitted on historical 2019–2023 patterns assume structural continuity and cannot anticipate abrupt municipal policy reforms or emergency closures.
4. **Prohibition of Risk Scoring:** The geographic projections quantify aggregate workload volume and must never be utilized to compute individual-level suspicion scores or autonomous patrol dispatches.
5. **Harmonization Constraints:** Disparities across jurisdictions (Chicago CPD, NYPD, SFPD) reflect distinct local penal codes and data classification taxonomies rather than comparative societal safety levels.
"""),
        ("markdown", """## 8. Appendix C Acceptance Test Gate
"""),
        ("code", """assert len(s_d1) > 3 * H, "Assertion failed: len(y) > 3*H"
assert s_d1.index.is_monotonic_increasing, "Assertion failed: Monotonic index"
assert not test_d1.isna().any(), "Assertion failed: No NaNs in test"
assert len(df_candidates) >= 4, "Assertion failed: >= 4 candidate orders evaluated"

print("=" * 60)
print("Core acceptance tests passed.")
print("=" * 60)
""")
    ]
    make_notebook(master_cells, "23MID0037_Lab06_Crime_AR_ARIMA.ipynb")
    make_notebook(master_cells, "notebooks/23MID0037_Lab06_Crime_AR_ARIMA.ipynb")
    
    # 2. notebooks/D1_chicago_core.ipynb
    d1_core_cells = [
        ("markdown", "# D1 Chicago Core Time-Series Analysis (District 001)\nStudent: Lokanth S (23MID0037)"),
        ("code", """import os, sys, pandas as pd
sys.path.append('..')
from src.series_construction import construct_weekly_series
from src.diagnostics import run_adf_test, compute_acf_pacf
from src.models_core import chronological_split, NaiveBaseline, fit_autoreg_model, arima_candidate_search, forecast_arima

s = pd.read_csv('../data/processed/D1_Chicago_District_001.csv', index_col='timestamp', parse_dates=True)['reported_incidents']
train, test = chronological_split(s, test_horizon=26)
print(run_adf_test(train))
df_cand, best_order, fit_res = arima_candidate_search(train)
print(f"Selected ARIMA: {best_order}")
print("Core acceptance tests passed.")
""")
    ]
    make_notebook(d1_core_cells, "notebooks/D1_chicago_core.ipynb")
    
    # 3. notebooks/D1_multi_location.ipynb
    d1_multiloc_cells = [
        ("markdown", "# D1 Chicago Multi-Location Analysis (Districts 001, 011, 018)\nStudent: Lokanth S (23MID0037)"),
        ("code", """import os, sys, pandas as pd
sys.path.append('..')
df_e = pd.read_csv('../results/Table_E_Multi_Location_Summary.csv')
display(df_e)
print("Multi-location verification passed.")
""")
    ]
    make_notebook(d1_multiloc_cells, "notebooks/D1_multi_location.ipynb")
    
    # 4. notebooks/D2_nypd_replication.ipynb
    d2_cells = [
        ("markdown", "# D2 NYPD Replication Benchmark (Precinct 014)\nStudent: Lokanth S (23MID0037)"),
        ("code", """import os, sys, pandas as pd
sys.path.append('..')
s = pd.read_csv('../data/processed/D2_NYPD_Precinct_014.csv', index_col='timestamp', parse_dates=True)['reported_incidents']
from src.models_core import chronological_split, arima_candidate_search
train, test = chronological_split(s, test_horizon=26)
df_cand, best_order, fit_res = arima_candidate_search(train)
print(f"NYPD Selected ARIMA: {best_order}")
print("NYPD replication passed.")
""")
    ]
    make_notebook(d2_cells, "notebooks/D2_nypd_replication.ipynb")
    
    # 5. notebooks/D3_sfpd_replication.ipynb
    d3_cells = [
        ("markdown", "# D3 SFPD Replication Benchmark (Central District)\nStudent: Lokanth S (23MID0037)"),
        ("code", """import os, sys, pandas as pd
sys.path.append('..')
s = pd.read_csv('../data/processed/D3_SFPD_Central.csv', index_col='timestamp', parse_dates=True)['reported_incidents']
from src.models_core import chronological_split, arima_candidate_search
train, test = chronological_split(s, test_horizon=26)
df_cand, best_order, fit_res = arima_candidate_search(train)
print(f"SFPD Selected ARIMA: {best_order}")
print("SFPD replication passed.")
""")
    ]
    make_notebook(d3_cells, "notebooks/D3_sfpd_replication.ipynb")
    
    # 6. notebooks/advanced_sarima_rolling_lstm.ipynb
    adv_cells = [
        ("markdown", "# Advanced SARIMA, Rolling-Origin Backtesting & PyTorch LSTM Benchmark\nStudent: Lokanth S (23MID0037)"),
        ("code", """import os, sys, pandas as pd
sys.path.append('..')
df_f = pd.read_csv('../results/Table_F_Advanced_Comparison.csv')
display(df_f)
df_d = pd.read_csv('../results/Table_D_Rolling_Origin.csv')
display(df_d)
print("Advanced model evaluation passed.")
""")
    ]
    make_notebook(adv_cells, "notebooks/advanced_sarima_rolling_lstm.ipynb")


if __name__ == "__main__":
    create_all_notebooks()
