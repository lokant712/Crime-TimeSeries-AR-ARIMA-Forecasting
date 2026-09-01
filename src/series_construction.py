"""
series_construction.py
----------------------
Aggregation and regular time series builder for reported crime incident counts.

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
import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def construct_weekly_series_from_daily(
    df: pd.DataFrame,
    date_col: str,
    location_col: str,
    location_val: str,
    count_col: str = "count",
    freq: str = "W-MON",
    category_col: str = None,
    category_val: str = None,
    series_name: str = "reported_incidents"
) -> pd.Series:
    """
    Constructs a regular, strictly monotonic weekly time series from live Socrata aggregate daily counts.
    """
    sub_df = df.copy()
    
    # 1. Normalize Location
    if location_col and location_val:
        sub_df[location_col] = sub_df[location_col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        target_loc = str(location_val).strip().replace(".0", "")
        # Match '001', '01', '1', etc.
        sub_df = sub_df[sub_df[location_col].isin([target_loc, target_loc.zfill(2), target_loc.zfill(3), target_loc.lstrip("0")])]
        
    # 2. Category Filtering (if applicable)
    if category_col and category_val:
        sub_df[category_col] = sub_df[category_col].astype(str).str.upper().str.strip()
        sub_df = sub_df[sub_df[category_col] == str(category_val).upper().strip()]
        
    if len(sub_df) == 0:
        raise ValueError(f"No records found for location={location_val}, category={category_val}")
        
    # 3. Robust Datetime Parsing
    sub_df["parsed_dt"] = pd.to_datetime(sub_df[date_col], errors="coerce")
    sub_df = sub_df.dropna(subset=["parsed_dt"])
    sub_df = sub_df.sort_values("parsed_dt")
    
    # 4. Convert Count Column
    if count_col in sub_df.columns:
        sub_df["numeric_count"] = pd.to_numeric(sub_df[count_col], errors="coerce").fillna(1)
    else:
        sub_df["numeric_count"] = 1
        
    # 5. Weekly Resampling (Sum counts)
    sub_df = sub_df.set_index("parsed_dt")
    weekly_counts = sub_df["numeric_count"].resample(freq).sum()
    
    # 6. Full Regular Range Reindexing
    full_idx = pd.date_range(
        start=weekly_counts.index.min(),
        end=weekly_counts.index.max(),
        freq=freq
    )
    # Reindex and fill missing weeks with 0 count
    # ASSUMPTION: An unrecorded week represents 0 reported incidents within that interval.
    regular_series = weekly_counts.reindex(full_idx, fill_value=0)
    regular_series.name = series_name
    regular_series.index.name = "timestamp"
    
    # Quality Assertions
    assert regular_series.index.is_monotonic_increasing, "Series index must be strictly increasing."
    assert not regular_series.isna().any(), "Series must contain no missing/NaN values."
    assert (regular_series >= 0).all(), "Incident counts must be strictly non-negative."
    
    logging.info(
        f"Built weekly series: {series_name} | Location: {location_val} | Category: {category_val or 'ALL'} | "
        f"Periods: {len(regular_series)} ({regular_series.index.min().strftime('%Y-%m-%d')} to {regular_series.index.max().strftime('%Y-%m-%d')}) | "
        f"Mean: {regular_series.mean():.1f} /wk | Std: {regular_series.std():.1f}"
    )
    return regular_series


def process_all_datasets(data_dir: str = "data/processed"):
    """
    Fetches real live government Socrata extracts and constructs regular weekly series CSVs for:
    - D1 Chicago: District 001, District 011, District 018, Category Theft, Category Battery
    - D2 NYPD: Precinct 014, Precinct 075
    - D3 SFPD: Central District, Mission District
    Updates data/DATASET_MANIFEST.json with verified SHA-256 hashes.
    """
    from src.data_ingest import fetch_live_socrata_dataset
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs("data/D1_chicago", exist_ok=True)
    os.makedirs("data/D2_nypd", exist_ok=True)
    os.makedirs("data/D3_sfpd", exist_ok=True)
    
    # 1. D1 Chicago Live Pull
    df_d1, sha_d1, rows_d1 = fetch_live_socrata_dataset("D1_CHICAGO")
    s_d1_001 = construct_weekly_series_from_daily(df_d1, "day", "district", "001", count_col="count", series_name="D1_Chicago_District_001")
    s_d1_011 = construct_weekly_series_from_daily(df_d1, "day", "district", "011", count_col="count", series_name="D1_Chicago_District_011")
    s_d1_018 = construct_weekly_series_from_daily(df_d1, "day", "district", "018", count_col="count", series_name="D1_Chicago_District_018")
    
    s_d1_theft = construct_weekly_series_from_daily(df_d1, "day", "district", "001", count_col="count", category_col="primary_type", category_val="THEFT", series_name="D1_Chicago_001_THEFT")
    s_d1_battery = construct_weekly_series_from_daily(df_d1, "day", "district", "001", count_col="count", category_col="primary_type", category_val="BATTERY", series_name="D1_Chicago_001_BATTERY")
    
    s_d1_001.to_csv(os.path.join(data_dir, "D1_Chicago_District_001.csv"))
    s_d1_011.to_csv(os.path.join(data_dir, "D1_Chicago_District_011.csv"))
    s_d1_018.to_csv(os.path.join(data_dir, "D1_Chicago_District_018.csv"))
    s_d1_theft.to_csv(os.path.join(data_dir, "D1_Chicago_001_THEFT.csv"))
    s_d1_battery.to_csv(os.path.join(data_dir, "D1_Chicago_001_BATTERY.csv"))
    
    s_d1_001.to_csv("data/D1_chicago/district_001_weekly.csv")
    s_d1_011.to_csv("data/D1_chicago/district_011_weekly.csv")
    s_d1_018.to_csv("data/D1_chicago/district_018_weekly.csv")
    s_d1_theft.to_csv("data/D1_chicago/district_001_theft_weekly.csv")
    s_d1_battery.to_csv("data/D1_chicago/district_001_battery_weekly.csv")
    
    # 2. D2 NYPD Live Pull
    df_d2, sha_d2, rows_d2 = fetch_live_socrata_dataset("D2_NYPD")
    s_d2_014 = construct_weekly_series_from_daily(df_d2, "cmplnt_fr_dt", "addr_pct_cd", "14", count_col=None, series_name="D2_NYPD_Precinct_014")
    s_d2_075 = construct_weekly_series_from_daily(df_d2, "cmplnt_fr_dt", "addr_pct_cd", "75", count_col=None, series_name="D2_NYPD_Precinct_075")
    s_d2_014.to_csv(os.path.join(data_dir, "D2_NYPD_Precinct_014.csv"))
    s_d2_075.to_csv(os.path.join(data_dir, "D2_NYPD_Precinct_075.csv"))
    s_d2_014.to_csv("data/D2_nypd/precinct_014_weekly.csv")
    s_d2_075.to_csv("data/D2_nypd/precinct_075_weekly.csv")
    
    # 3. D3 SFPD Live Pull
    df_d3, sha_d3, rows_d3 = fetch_live_socrata_dataset("D3_SFPD")
    s_d3_central = construct_weekly_series_from_daily(df_d3, "day", "police_district", "Central", count_col="count", series_name="D3_SFPD_Central")
    s_d3_mission = construct_weekly_series_from_daily(df_d3, "day", "police_district", "Mission", count_col="count", series_name="D3_SFPD_Mission")
    s_d3_central.to_csv(os.path.join(data_dir, "D3_SFPD_Central.csv"))
    s_d3_mission.to_csv(os.path.join(data_dir, "D3_SFPD_Mission.csv"))
    s_d3_central.to_csv("data/D3_sfpd/district_central_weekly.csv")
    s_d3_mission.to_csv("data/D3_sfpd/district_mission_weekly.csv")
    
    # Update DATASET_MANIFEST.json with verified real SHA-256 hashes
    manifest = {
        "manifest_version": "1.0.0",
        "provenance_audit_type": "GENUINE_LIVE_SOCRATA_OPEN_DATA_PULL",
        "student": {
            "name": "Lokanth S",
            "reg_no": "23MID0037",
            "faculty_coordinator": "Dr. Durgesh Kumar"
        },
        "deployment_boundary": "The models forecast counts of reported incidents, not actual underlying crime prevalence and not individual criminal behavior. Forecasts must not be used for person-level profiling or autonomous policing decisions.",
        "datasets": {
            "D1_CHICAGO": {
                "name": "Chicago Crimes (2001 to Present)",
                "agency": "City of Chicago Data Portal / Chicago Police Department",
                "source_url": "https://data.cityofchicago.org/resource/ijzp-q8t2.json",
                "access_date": datetime.now().strftime("%Y-%m-%d"),
                "query_filter": "date >= '2019-01-01T00:00:00' AND date <= '2024-06-30T23:59:59' AND district in('1','11','18','001','011','018')",
                "aggregation": "Weekly ('W-MON')",
                "observation_window": f"{s_d1_001.index.min().strftime('%Y-%m-%d')} to {s_d1_001.index.max().strftime('%Y-%m-%d')}",
                "total_series_periods": len(s_d1_001),
                "districts_modeled": ["001", "011", "018"],
                "categories_modeled": ["THEFT", "BATTERY"],
                "raw_records_pulled": rows_d1,
                "sha256": sha_d1
            },
            "D2_NYPD": {
                "name": "NYPD Complaint Data Historic",
                "agency": "NYC OpenData / New York City Police Department",
                "source_url": "https://data.cityofnewyork.us/resource/qgea-i56i.json",
                "access_date": datetime.now().strftime("%Y-%m-%d"),
                "query_filter": "cmplnt_fr_dt >= '2019-01-01T00:00:00' AND cmplnt_fr_dt <= '2023-12-31T23:59:59' AND addr_pct_cd in('14','75','014','075')",
                "aggregation": "Weekly ('W-MON')",
                "observation_window": f"{s_d2_014.index.min().strftime('%Y-%m-%d')} to {s_d2_014.index.max().strftime('%Y-%m-%d')}",
                "total_series_periods": len(s_d2_014),
                "precincts_modeled": ["014", "075"],
                "raw_records_pulled": rows_d2,
                "sha256": sha_d2
            },
            "D3_SFPD": {
                "name": "Police Department Incident Reports: 2018 to Present",
                "agency": "DataSF / San Francisco Police Department",
                "source_url": "https://data.sfgov.org/resource/wg3w-h783.json",
                "access_date": datetime.now().strftime("%Y-%m-%d"),
                "query_filter": "incident_datetime >= '2019-01-01T00:00:00' AND incident_datetime <= '2024-06-30T23:59:59' AND police_district in('Central','Mission')",
                "aggregation": "Weekly ('W-MON')",
                "observation_window": f"{s_d3_central.index.min().strftime('%Y-%m-%d')} to {s_d3_central.index.max().strftime('%Y-%m-%d')}",
                "total_series_periods": len(s_d3_central),
                "districts_modeled": ["Central", "Mission"],
                "raw_records_pulled": rows_d3,
                "sha256": sha_d3
            }
        }
    }
    with open("data/DATASET_MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=4)
        
    logging.info("All processed weekly series saved successfully from live Socrata extracts.")
    return {
        "D1_Chicago_001": s_d1_001,
        "D1_Chicago_011": s_d1_011,
        "D1_Chicago_018": s_d1_018,
        "D1_Theft": s_d1_theft,
        "D1_Battery": s_d1_battery,
        "D2_NYPD_014": s_d2_014,
        "D3_SFPD_Central": s_d3_central
    }
