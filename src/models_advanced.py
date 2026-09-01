"""
models_advanced.py
------------------
Advanced Time-Series Models: SARIMA, SARIMAX, Rolling-Origin Backtester, PyTorch LSTM/GRU.

Student: Lokanth S | Reg No: 23MID0037
Faculty Coordinator: Dr. Durgesh Kumar
Repository: lokant712/Crime-TimeSeries-AR-ARIMA-Forecasting

DEPLOYMENT-BOUNDARY GUARDRAIL:
The models forecast counts of reported incidents, not actual underlying crime
prevalence and not individual criminal behavior. Forecasts must not be used
for person-level profiling or autonomous policing decisions.
"""

import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, List
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------
# 1. SARIMA & SARIMAX Models
# ---------------------------------------------------------
def fit_sarima_model(
    train: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 52),
    test_horizon: int = 26
) -> Dict[str, Any]:
    """
    Fits Multiplicative Seasonal ARIMA (SARIMA) model capturing weekly annual seasonality.
    """
    t0 = time.time()
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fit_res = model.fit(disp=False)
    fit_time = time.time() - t0
    
    forecast_obj = fit_res.get_forecast(steps=test_horizon)
    preds = forecast_obj.predicted_mean
    conf_80 = forecast_obj.conf_int(alpha=0.20)
    conf_95 = forecast_obj.conf_int(alpha=0.05)
    
    return {
        "model_name": f"SARIMA{order}x{seasonal_order}",
        "fit_res": fit_res,
        "train_aic": float(fit_res.aic),
        "train_bic": float(fit_res.bic),
        "test_preds": np.array(preds),
        "conf_80": np.array(conf_80),
        "conf_95": np.array(conf_95),
        "residuals": fit_res.resid,
        "runtime_sec": fit_time
    }


def create_calendar_exogenous_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Generates genuine, forecast-time-available exogenous features:
    - Month harmonic encoding (sin/cos of month of year)
    - Major US federal holiday week indicator
    """
    df_exog = pd.DataFrame(index=index)
    month = index.month
    df_exog["sin_month"] = np.sin(2 * np.pi * month / 12.0)
    df_exog["cos_month"] = np.cos(2 * np.pi * month / 12.0)
    
    # Holiday week indicators: New Year (Jan 1), Independence Day (July 4), Christmas (Dec 25)
    df_exog["is_holiday_week"] = (
        ((month == 1) & (index.day <= 7)) |
        ((month == 7) & (index.day <= 7)) |
        ((month == 12) & (index.day >= 24))
    ).astype(float)
    
    return df_exog


def fit_sarimax_model(
    train: pd.Series,
    test: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 52)
) -> Dict[str, Any]:
    """
    Fits SARIMAX with strictly known forecast-time calendar regressors.
    """
    t0 = time.time()
    train_exog = create_calendar_exogenous_features(train.index)
    test_exog = create_calendar_exogenous_features(test.index)
    
    model = SARIMAX(
        train,
        exog=train_exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fit_res = model.fit(disp=False)
    fit_time = time.time() - t0
    
    forecast_obj = fit_res.get_forecast(steps=len(test), exog=test_exog)
    preds = forecast_obj.predicted_mean
    conf_80 = forecast_obj.conf_int(alpha=0.20)
    conf_95 = forecast_obj.conf_int(alpha=0.05)
    
    return {
        "model_name": "SARIMAX(Calendar-Exog)",
        "fit_res": fit_res,
        "train_aic": float(fit_res.aic),
        "train_bic": float(fit_res.bic),
        "test_preds": np.array(preds),
        "conf_80": np.array(conf_80),
        "conf_95": np.array(conf_95),
        "residuals": fit_res.resid,
        "runtime_sec": fit_time
    }


# ---------------------------------------------------------
# 2. Rolling-Origin Walk-Forward Backtester (Expanding Window)
# ---------------------------------------------------------
def run_rolling_origin_backtest(
    train_val_series: pd.Series,
    ar_order: int = 4,
    arima_order: Tuple[int, int, int] = (1, 1, 1),
    n_folds: int = 5,
    horizon_steps: int = 8
) -> Dict[str, pd.DataFrame]:
    """
    Executes expanding-window rolling-origin walk-forward backtesting strictly within
    training+validation history (never touching locked test period).
    
    Computes per-fold MAE & RMSE, Mean, and SD across folds for:
    - Naive (Persistence)
    - AR(p)
    - ARIMA(p,d,q)
    """
    total_len = len(train_val_series)
    min_train_len = total_len - (n_folds * horizon_steps)
    if min_train_len < 52:
        raise ValueError("Insufficient history for requested folds/horizon.")
        
    models = ["Naive", f"AR({ar_order})", f"ARIMA{arima_order}"]
    fold_records = {m: [] for m in models}
    
    for k in range(n_folds):
        fold_idx = k + 1
        train_end_idx = min_train_len + (k * horizon_steps)
        val_end_idx = train_end_idx + horizon_steps
        
        fold_train = train_val_series.iloc[:train_end_idx]
        fold_val = train_val_series.iloc[train_end_idx:val_end_idx]
        
        train_sz = len(fold_train)
        val_sz = len(fold_val)
        val_win_str = f"{fold_val.index.min().strftime('%Y-%m-%d')} to {fold_val.index.max().strftime('%Y-%m-%d')}"
        y_true = fold_val.values
        
        # 1. Naive Model
        y_pred_naive = np.repeat(fold_train.iloc[-1], val_sz)
        mae_naive = np.mean(np.abs(y_true - y_pred_naive))
        rmse_naive = np.sqrt(np.mean((y_true - y_pred_naive) ** 2))
        fold_records["Naive"].append({
            "fold": fold_idx,
            "train_size": train_sz,
            "val_window": val_win_str,
            "mae": round(float(mae_naive), 2),
            "rmse": round(float(rmse_naive), 2)
        })
        
        # 2. AR(p) Model
        try:
            ar_fit = AutoReg(fold_train, lags=ar_order).fit()
            y_pred_ar = ar_fit.get_prediction(start=train_sz, end=train_sz + val_sz - 1).predicted_mean.values
            mae_ar = np.mean(np.abs(y_true - y_pred_ar))
            rmse_ar = np.sqrt(np.mean((y_true - y_pred_ar) ** 2))
        except Exception:
            mae_ar, rmse_ar = np.nan, np.nan
        fold_records[f"AR({ar_order})"].append({
            "fold": fold_idx,
            "train_size": train_sz,
            "val_window": val_win_str,
            "mae": round(float(mae_ar), 2),
            "rmse": round(float(rmse_ar), 2)
        })
        
        # 3. ARIMA Model
        try:
            arima_fit = ARIMA(fold_train, order=arima_order).fit()
            y_pred_arima = arima_fit.get_forecast(steps=val_sz).predicted_mean.values
            mae_arima = np.mean(np.abs(y_true - y_pred_arima))
            rmse_arima = np.sqrt(np.mean((y_true - y_pred_arima) ** 2))
        except Exception:
            mae_arima, rmse_arima = np.nan, np.nan
        fold_records[f"ARIMA{arima_order}"].append({
            "fold": fold_idx,
            "train_size": train_sz,
            "val_window": val_win_str,
            "mae": round(float(mae_arima), 2),
            "rmse": round(float(rmse_arima), 2)
        })
        
    # Build Table D summary DataFrames
    tables_d = {}
    for m, recs in fold_records.items():
        df_m = pd.DataFrame(recs)
        mean_mae = df_m["mae"].mean()
        sd_mae = df_m["mae"].std()
        mean_rmse = df_m["rmse"].mean()
        sd_rmse = df_m["rmse"].std()
        
        summary_row = pd.DataFrame([{
            "fold": "Mean (SD)",
            "train_size": f"{df_m['train_size'].mean():.0f}",
            "val_window": f"{n_folds} Backtest Folds",
            "mae": f"{mean_mae:.2f} ({sd_mae:.2f})",
            "rmse": f"{mean_rmse:.2f} ({sd_rmse:.2f})"
        }])
        df_full = pd.concat([df_m, summary_row], ignore_index=True)
        tables_d[m] = df_full
        
    return tables_d


# ---------------------------------------------------------
# 3. PyTorch Deep Learning Benchmarks (LSTM & GRU)
# ---------------------------------------------------------
class RNNForecaster(nn.Module):
    def __init__(self, cell_type: str = "LSTM", input_size: int = 1, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.cell_type = cell_type
        if cell_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


def train_evaluate_recurrent_benchmark(
    train: pd.Series,
    test: pd.Series,
    seq_len: int = 12,
    epochs: int = 80,
    seeds: List[int] = [42, 101, 2024]
) -> Dict[str, Any]:
    """
    Trains PyTorch LSTM and GRU models across >=3 repeated runs.
    Logs training loss, validation MAE/RMSE, inference latency, parameter count,
    and produces an explicit complexity-vs-accuracy trade-off analysis.
    """
    train_vals = train.values.astype(np.float32)
    test_vals = test.values.astype(np.float32)
    
    # Scale via train stats
    mean_val = float(np.mean(train_vals))
    std_val = float(np.std(train_vals)) + 1e-6
    train_norm = (train_vals - mean_val) / std_val
    test_norm = (test_vals - mean_val) / std_val
    
    # Sequence creation
    def create_sequences(arr, seq_l):
        xs, ys = [], []
        for i in range(len(arr) - seq_l):
            xs.append(arr[i:i+seq_l])
            ys.append(arr[i+seq_l])
        return torch.tensor(np.array(xs)[:, :, None], dtype=torch.float32), torch.tensor(np.array(ys)[:, None], dtype=torch.float32)
        
    X_train, y_train = create_sequences(train_norm, seq_len)
    
    results = {}
    for cell in ["LSTM", "GRU"]:
        seed_maes = []
        seed_rmses = []
        train_times = []
        infer_times = []
        loss_histories = []
        
        for s in seeds:
            torch.manual_seed(s)
            np.random.seed(s)
            
            model = RNNForecaster(cell_type=cell, hidden_size=32)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            t0 = time.time()
            losses = []
            for ep in range(epochs):
                model.train()
                optimizer.zero_grad()
                out = model(X_train)
                loss = criterion(out, y_train)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            t_train = time.time() - t0
            train_times.append(t_train)
            loss_histories.append(losses)
            
            # Recursive multi-step forecasting on test horizon
            model.eval()
            curr_seq = train_norm[-seq_len:].tolist()
            preds_norm = []
            
            t_inf0 = time.time()
            with torch.no_grad():
                for _ in range(len(test)):
                    inp = torch.tensor(curr_seq[-seq_len:], dtype=torch.float32).view(1, seq_len, 1)
                    p = model(inp).item()
                    preds_norm.append(p)
                    curr_seq.append(p)
            t_inf = (time.time() - t_inf0) * 1000.0 # ms
            infer_times.append(t_inf)
            
            preds_unnorm = np.array(preds_norm) * std_val + mean_val
            mae = float(np.mean(np.abs(test_vals - preds_unnorm)))
            rmse = float(np.sqrt(np.mean((test_vals - preds_unnorm) ** 2)))
            seed_maes.append(mae)
            seed_rmses.append(rmse)
            
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results[cell] = {
            "mean_mae": float(np.mean(seed_maes)),
            "std_mae": float(np.std(seed_maes)),
            "mean_rmse": float(np.mean(seed_rmses)),
            "std_rmse": float(np.std(seed_rmses)),
            "mean_train_time_sec": float(np.mean(train_times)),
            "mean_infer_latency_ms": float(np.mean(infer_times)),
            "param_count": int(param_count),
            "loss_history": loss_histories[0],
            "sample_preds": preds_unnorm
        }
        
    return results
