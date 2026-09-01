"""
models_core.py
--------------
Core Time-Series Forecasting Models: Baseline (Naive/S-Naive), AR(p), ARIMA(p,d,q).

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
import warnings
from typing import Tuple, Dict, Any, List
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def chronological_split(
    series: pd.Series,
    test_horizon: int = 26
) -> Tuple[pd.Series, pd.Series]:
    """
    Splits series chronologically without shuffling into train and locked test sets.
    
    Ensures:
    - No temporal leakage (train strictly precedes test).
    - Minimum length requirement: len(y) > 3 * H.
    """
    total_len = len(series)
    if total_len <= 3 * test_horizon:
        raise ValueError(f"Series length ({total_len}) must be > 3 * horizon ({3 * test_horizon}).")
        
    train = series.iloc[:-test_horizon]
    test = series.iloc[-test_horizon:]
    
    assert train.index.max() < test.index.min(), "Leakage violation: train timestamps overlap with test."
    assert len(test) == test_horizon, f"Test set must have length {test_horizon}."
    
    logging.info(
        f"Chronological split completed: Train periods={len(train)} "
        f"({train.index.min().strftime('%Y-%m-%d')} to {train.index.max().strftime('%Y-%m-%d')}) | "
        f"Test periods={len(test)} ({test.index.min().strftime('%Y-%m-%d')} to {test.index.max().strftime('%Y-%m-%d')})"
    )
    return train, test


class NaiveBaseline:
    """
    Persistence / Last-Value and Seasonal-Naive Baselines.
    """
    def __init__(self, seasonal_period: int = None):
        self.seasonal_period = seasonal_period
        self.last_value = None
        self.train_history = None
        
    def fit(self, train: pd.Series):
        self.train_history = train.copy()
        self.last_value = train.iloc[-1]
        return self
        
    def predict(self, steps: int) -> pd.Series:
        if self.seasonal_period is None or len(self.train_history) < self.seasonal_period:
            # Last-value persistence
            preds = np.repeat(self.last_value, steps)
        else:
            # Seasonal naive: repeat corresponding observations from previous cycle
            lag_vals = self.train_history.iloc[-self.seasonal_period:].values
            reps = int(np.ceil(steps / self.seasonal_period))
            preds = np.tile(lag_vals, reps)[:steps]
        return preds


def fit_autoreg_model(
    train: pd.Series,
    lags: int,
    test_horizon: int
) -> Dict[str, Any]:
    """
    Fits an AR(p) model using statsmodels AutoReg with specified lag order.
    Produces point forecasts and prediction intervals.
    """
    model = AutoReg(train, lags=lags)
    fit_res = model.fit()
    
    forecast_res = fit_res.get_prediction(start=len(train), end=len(train) + test_horizon - 1)
    preds = forecast_res.predicted_mean
    conf_80 = forecast_res.conf_int(alpha=0.20)
    conf_95 = forecast_res.conf_int(alpha=0.05)
    
    # Train in-sample fitted values & residuals
    in_sample_preds = fit_res.fittedvalues
    train_resids = train.loc[in_sample_preds.index] - in_sample_preds
    
    return {
        "model_name": f"AR({lags})",
        "order": lags,
        "fit_res": fit_res,
        "train_aic": float(fit_res.aic),
        "train_bic": float(fit_res.bic),
        "test_preds": np.array(preds),
        "conf_80": np.array(conf_80),
        "conf_95": np.array(conf_95),
        "residuals": train_resids,
        "converged": True
    }


def arima_candidate_search(
    train: pd.Series,
    candidate_orders: List[Tuple[int, int, int]] = None
) -> Tuple[pd.DataFrame, Tuple[int, int, int], Any]:
    """
    Conducts a rigorous candidate grid search across ARIMA(p,d,q) orders on training data only.
    Logs candidate parameters, AIC, BIC, Log-Likelihood, and convergence status.
    Selects the best order strictly by minimum training AIC.
    """
    if candidate_orders is None:
        candidate_orders = [
            (1, 0, 0),
            (2, 0, 0),
            (1, 1, 1),
            (2, 1, 1),
            (0, 1, 1),
            (1, 1, 0),
            (2, 1, 2),
            (3, 1, 1)
        ]
        
    records = []
    best_aic = float("inf")
    best_order = None
    best_fit = None
    
    for order in candidate_orders:
        try:
            model = ARIMA(train, order=order)
            fit_res = model.fit()
            aic_val = fit_res.aic
            bic_val = fit_res.bic
            converged = bool(fit_res.mle_retvals.get("converged", True)) if hasattr(fit_res, "mle_retvals") else True
            
            records.append({
                "candidate_order": str(order),
                "p": order[0],
                "d": order[1],
                "q": order[2],
                "aic": round(float(aic_val), 2),
                "bic": round(float(bic_val), 2),
                "log_likelihood": round(float(fit_res.llf), 2),
                "converged": converged
            })
            
            if converged and aic_val < best_aic:
                best_aic = aic_val
                best_order = order
                best_fit = fit_res
                
        except Exception as e:
            records.append({
                "candidate_order": str(order),
                "p": order[0],
                "d": order[1],
                "q": order[2],
                "aic": np.nan,
                "bic": np.nan,
                "log_likelihood": np.nan,
                "converged": False
            })
            
    df_search = pd.DataFrame(records)
    df_search["selected"] = df_search["candidate_order"].apply(lambda x: "Y" if x == str(best_order) else "N")
    
    logging.info(f"ARIMA search evaluated {len(df_search)} candidate orders. Best order selected: {best_order} (AIC={best_aic:.2f})")
    return df_search, best_order, best_fit


def forecast_arima(
    fit_res: Any,
    test_horizon: int
) -> Dict[str, Any]:
    """
    Generates point forecasts and 80% & 95% prediction intervals from fitted ARIMA model.
    """
    forecast_obj = fit_res.get_forecast(steps=test_horizon)
    preds = forecast_obj.predicted_mean
    conf_80 = forecast_obj.conf_int(alpha=0.20)
    conf_95 = forecast_obj.conf_int(alpha=0.05)
    
    return {
        "test_preds": np.array(preds),
        "conf_80": np.array(conf_80),
        "conf_95": np.array(conf_95),
        "residuals": fit_res.resid
    }
