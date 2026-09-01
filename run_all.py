"""
run_all.py
----------
Root pipeline launcher delegating to scripts/run_all.py.

Student: Lokanth S | Reg No: 23MID0037
Faculty Coordinator: Dr. Durgesh Kumar
Repository: lokant712/Crime-TimeSeries-AR-ARIMA-Forecasting
"""

import sys
import os

sys.path.append(os.path.abspath("."))
from scripts.run_all import run_all

if __name__ == "__main__":
    run_all()
