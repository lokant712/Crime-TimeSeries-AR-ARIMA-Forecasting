"""
data_ingest.py
--------------
Live Socrata Open Data Ingestion & Provenance Module for MDI3003 Lab 06:
Time-Series Analysis & Forecasting of Reported Crime Incidents (AR & ARIMA).

Student: Lokanth S | Reg No: 23MID0037
Faculty Coordinator: Dr. Durgesh Kumar
Repository: lokant712/Crime-TimeSeries-AR-ARIMA-Forecasting

DEPLOYMENT-BOUNDARY GUARDRAIL:
The models forecast counts of reported incidents, not actual underlying crime
prevalence and not individual criminal behavior. Forecasts must not be used
for person-level profiling or autonomous policing decisions.
"""

import os
import json
import logging
import hashlib
import requests
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Socrata Endpoints & Metadata
ENDPOINTS = {
    "D1_CHICAGO": {
        "url": "https://data.cityofchicago.org/resource/ijzp-q8t2.json",
        "agency": "City of Chicago Data Portal / Chicago Police Department",
        "dataset_name": "Crimes - 2001 to Present",
        "date_col": "day",
        "loc_col": "district",
        "type_col": "primary_type",
        "count_col": "count",
        "default_districts": ["001", "011", "018"],
        "time_span": "2019-01-01 to 2024-06-30",
        "soql_params": {
            "$select": "date_trunc_ymd(date) as day, district, primary_type, count(1) as count",
            "$where": "date >= '2019-01-01T00:00:00' AND date <= '2024-06-30T23:59:59' AND district in('1','11','18','001','011','018')",
            "$group": "day, district, primary_type",
            "$limit": 50000
        }
    },
    "D2_NYPD": {
        "url": "https://data.cityofnewyork.us/resource/qgea-i56i.json",
        "agency": "NYC OpenData / New York City Police Department",
        "dataset_name": "NYPD Complaint Data Historic",
        "date_col": "cmplnt_fr_dt",
        "loc_col": "addr_pct_cd",
        "type_col": "ofns_desc",
        "count_col": None,
        "default_precincts": ["014", "075"],
        "time_span": "2019-01-01 to 2023-12-31",
        "soql_params": {
            "$where": "cmplnt_fr_dt >= '2019-01-01' AND cmplnt_fr_dt <= '2023-12-31' AND addr_pct_cd in('14','75','014','075')",
            "$limit": 50000,
            "$order": "cmplnt_fr_dt ASC"
        }
    },
    "D3_SFPD": {
        "url": "https://data.sfgov.org/resource/wg3w-h783.json",
        "agency": "DataSF / San Francisco Police Department",
        "dataset_name": "Police Department Incident Reports: 2018 to Present",
        "date_col": "day",
        "loc_col": "police_district",
        "type_col": None,
        "count_col": "count",
        "default_districts": ["Central", "Mission"],
        "time_span": "2019-01-01 to 2024-06-30",
        "soql_params": {
            "$select": "date_trunc_ymd(incident_datetime) as day, police_district, count(1) as count",
            "$where": "incident_datetime >= '2019-01-01T00:00:00' AND incident_datetime <= '2024-06-30T23:59:59' AND police_district in('Central','Mission')",
            "$group": "day, police_district",
            "$limit": 50000
        }
    }
}


def fetch_live_socrata_dataset(dataset_key: str, raw_dir: str = "data/raw") -> tuple[pd.DataFrame, str, int]:
    """
    Fetches real incident records directly from official government Socrata Open Data portals.
    Saves the exact raw JSON response and computes a verified SHA-256 checksum.
    """
    os.makedirs(raw_dir, exist_ok=True)
    cfg = ENDPOINTS[dataset_key]
    logging.info(f"Querying live Socrata API for {dataset_key} ({cfg['dataset_name']})...")
    
    headers = {"User-Agent": "MDI3003-Academic-Predictive-Analytics/23MID0037 (Academic Research)"}
    
    resp = requests.get(cfg["url"], params=cfg["soql_params"], headers=headers, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Socrata API query failed for {dataset_key} with HTTP {resp.status_code}: {resp.text}")
        
    records = resp.json()
    if not records or len(records) == 0:
        raise ValueError(f"Socrata API returned empty payload for {dataset_key}.")
        
    raw_bytes = json.dumps(records, sort_keys=True).encode("utf-8")
    sha256_hash = hashlib.sha256(raw_bytes).hexdigest()
    
    raw_path = os.path.join(raw_dir, f"{dataset_key}_raw.json")
    with open(raw_path, "wb") as f:
        f.write(raw_bytes)
        
    df = pd.DataFrame(records)
    logging.info(f"Successfully retrieved {len(df)} live records for {dataset_key} (SHA-256: {sha256_hash[:16]}...)")
    return df, sha256_hash, len(df)
