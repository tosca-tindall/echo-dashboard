"""
Sentinel-1 SAR Soil Moisture Downloader
========================================
Pulls Sentinel-1 backscatter data for East Anglia from Google Earth Engine
and derives a soil moisture proxy, saved as a CSV file ready for the dashboard.

Run with:   python3 download_soil_moisture.py
"""

import ee
import pandas as pd
import json
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these if needed
# ─────────────────────────────────────────────────────────────────────────────

PROJECT    = "echo-soil-moisture"   # Your GEE project name
START_DATE = "2020-01-01"
END_DATE   = "2023-12-31"
OUTPUT_CSV = "sentinel1_soil_moisture.csv"

# East Anglia bounding box
BBOX = [0.35, 52.40, 0.65, 52.60]  # [min_lon, min_lat, max_lon, max_lat]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Connect to Google Earth Engine
# ─────────────────────────────────────────────────────────────────────────────

print("Connecting to Google Earth Engine...")
ee.Initialize(project=PROJECT)
print("Connected.\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Define the study area
# ─────────────────────────────────────────────────────────────────────────────

region = ee.Geometry.Rectangle(BBOX)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Load Sentinel-1 backscatter data
#
# Sentinel-1 measures radar backscatter in decibels (dB).
# VV polarisation (vertical transmit, vertical receive) is most sensitive
# to soil moisture. Lower backscatter = drier soil.
# ─────────────────────────────────────────────────────────────────────────────

print("Loading Sentinel-1 data from Earth Engine...")

s1 = (
    ee.ImageCollection("COPERNICUS/S1_GRD")
    .filterBounds(region)
    .filterDate(START_DATE, END_DATE)
    .filter(ee.Filter.eq("instrumentMode", "IW"))           # Interferometric Wide swath
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))  # More consistent
    .select("VV")                                           # VV polarisation only
)

print(f"Found images. Extracting monthly means over bounding box...")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Compute monthly mean backscatter over the region
#
# Rather than downloading every pixel, we compute a single spatial mean
# for the region per month — this is fast and gives us the time series
# we need for the dashboard.
# ─────────────────────────────────────────────────────────────────────────────

def extract_monthly_mean(year, month):
    """Get the mean VV backscatter for one month over the study region."""
    start = f"{year}-{month:02d}-01"
    # Last day of month
    next_month = month + 1 if month < 12 else 1
    next_year  = year if month < 12 else year + 1
    end = f"{next_year}-{next_month:02d}-01"

    monthly_mean = (
        s1.filterDate(start, end)
        .mean()
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=100,        # 100m resolution for speed; S1 native is ~10m
            maxPixels=1e9,
        )
    )
    return {"year": year, "month": month, "vv_db": monthly_mean.get("VV")}


records = []
for year in range(2020, 2024):
    for month in range(1, 13):
        result = extract_monthly_mean(year, month)
        # .getInfo() sends the computation to GEE and retrieves the result
        vv_value = ee.Number(result["vv_db"]).getInfo()
        date_str = f"{year}-{month:02d}"
        records.append({"date": date_str, "vv_db": vv_value})
        print(f"  {date_str}: VV = {vv_value:.2f} dB" if vv_value else f"  {date_str}: no data")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Convert backscatter (dB) to a soil moisture proxy
#
# True physical inversion requires a radiative transfer model.
# For a prototype, we use a simple linear normalisation:
#   - Wetter soils → higher (less negative) VV backscatter
#   - We scale VV dB to a 0–1 soil moisture index
#
# In production this would be replaced by a full SAR inversion algorithm
# (e.g. the TU Wien change detection method or a trained ML model).
# ─────────────────────────────────────────────────────────────────────────────

df = pd.DataFrame(records).dropna()
df["date"] = pd.to_datetime(df["date"])

# Typical S1 VV backscatter range over agricultural land: -20 dB (dry) to -5 dB (wet)
VV_DRY = -20.0
VV_WET =  -5.0

df["sm_proxy"] = (df["vv_db"] - VV_DRY) / (VV_WET - VV_DRY)
df["sm_proxy"] = df["sm_proxy"].clip(0.0, 1.0)

# Scale to volumetric soil moisture range (m³/m³) comparable to ERA5-Land
# Typical range for East Anglian arable land: 0.10 – 0.42
SM_MIN = 0.10
SM_MAX = 0.42
df["sm_m3m3"] = SM_MIN + df["sm_proxy"] * (SM_MAX - SM_MIN)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Save to CSV
# ─────────────────────────────────────────────────────────────────────────────

df[["date", "vv_db", "sm_proxy", "sm_m3m3"]].to_csv(OUTPUT_CSV, index=False)

print(f"\n✓ Done. Saved {len(df)} monthly records to '{OUTPUT_CSV}'")
print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"  VV backscatter range: {df['vv_db'].min():.1f} to {df['vv_db'].max():.1f} dB")
print(f"  Soil moisture proxy range: {df['sm_m3m3'].min():.3f} to {df['sm_m3m3'].max():.3f} m³/m³")
print(f"\nNext step: run 'streamlit run app.py' to see the dashboard with real data.")
