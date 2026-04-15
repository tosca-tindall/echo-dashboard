"""
Echo Labs — Field-Level Parametric Insurance Explorer
=======================================================
Interactive geospatial dashboard built with Streamlit + Pydeck.
All data is synthetic and for illustrative purposes only.

Run with:   streamlit run app.py
"""

import math
import hashlib
import warnings
import os
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import plotly.graph_objects as go
import streamlit as st
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from functools import lru_cache

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# DATA FILE DOWNLOADER
# Downloads data files from Google Drive if not already present locally.
# This runs on startup so the app works on cloud hosts with no local files.
# ─────────────────────────────────────────────────────────────────────────────

DATA_FILES = {
    "sentinel1_soil_moisture.csv": "1F6l_ntNX99_7SBt5fNKqMdcy0ikPzEZz",
    "FLAME_dataset_v2.xlsx":       "1Cmt5H26Ys9E64-96OxcuD5ZfzuFatHEs",
    "ukfields.parquet":            "1PequKF-t8K_JgLnD0YVwNz2dos9jQklC",
}

def _gdrive_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def _gdrive_url_large(file_id: str, session: requests.Session) -> tuple:
    """Handle Google Drive large file confirm token."""
    url = _gdrive_url(file_id)
    resp = session.get(url, stream=True)
    # Check for virus-scan warning page (large files)
    for key, val in resp.cookies.items():
        if key.startswith("download_warning"):
            return f"{url}&confirm={val}", resp
    return url, resp

def download_data_files():
    """Download any missing data files from Google Drive."""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    session = requests.Session()
    for filename, file_id in DATA_FILES.items():
        dest = os.path.join(app_dir, filename)
        if os.path.exists(dest):
            continue
        print(f"Downloading {filename}...")
        try:
            url, resp = _gdrive_url_large(file_id, session)
            if resp.status_code != 200:
                resp = session.get(url, stream=True)
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
            print(f"  ✓ {filename} saved ({os.path.getsize(dest)//1024} KB)")
        except Exception as e:
            print(f"  ✗ Failed to download {filename}: {e}")

download_data_files()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Echo Labs · Parametric Insurance Explorer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Syne:wght@400;500;600;700;800&display=swap');

/* ── Reset ── */
html, body, [class*="css"] { font-family: 'DM Mono', monospace; }

/* ── Background ── */
.stApp {
    background: #0a0e14;
    color: #c8d0dc;
}

/* ── Header band ── */
.echo-header {
    background: linear-gradient(135deg, #0d1520 0%, #0f2032 50%, #091829 100%);
    border-bottom: 1px solid #1e3448;
    padding: 1.4rem 2rem 1.2rem;
    margin: -1rem -1rem 1.5rem;
    display: flex;
    align-items: baseline;
    gap: 1.2rem;
}
.echo-wordmark {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.55rem;
    color: #e8f4ff;
    letter-spacing: -0.02em;
}
.echo-wordmark span { color: #3aecb0; }
.echo-subtitle {
    font-size: 0.72rem;
    color: #4a7a9b;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 300;
}
.echo-pill {
    margin-left: auto;
    background: #0d2235;
    border: 1px solid #1e3448;
    border-radius: 2rem;
    padding: 0.2rem 0.75rem;
    font-size: 0.65rem;
    color: #3aecb0;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Stat cards ── */
.stat-grid { display: flex; flex-direction: column; gap: 0.6rem; }
.stat-card {
    background: #0d1a28;
    border: 1px solid #1a2e42;
    border-radius: 8px;
    padding: 0.85rem 1rem;
}
.stat-label {
    font-size: 0.62rem;
    color: #4a7a9b;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: #e8f4ff;
    line-height: 1;
}
.stat-value.payout { color: #f87c6a; }
.stat-value.trigger { color: #f7c94b; }
.stat-value.moisture { color: #3aecb0; }
.stat-sub {
    font-size: 0.6rem;
    color: #4a7a9b;
    margin-top: 0.2rem;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.62rem;
    color: #3aecb0;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin: 1.2rem 0 0.5rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #1a2e42;
}

/* ── Control widgets ── */
.stSlider > div > div > div > div { background: #3aecb0 !important; }
.stSelectbox > div { background: #0d1a28 !important; border-color: #1a2e42 !important; }
div[data-baseweb="select"] { background: #0d1a28; }
div[data-baseweb="select"] * { color: #c8d0dc !important; }

/* ── Footer ── */
.echo-footer {
    margin-top: 2rem;
    padding: 0.8rem 0;
    border-top: 1px solid #1a2e42;
    font-size: 0.6rem;
    color: #2a4a62;
    text-align: center;
    letter-spacing: 0.05em;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _irregular_polygon(cx, cy, n_verts: int, mean_r: float, roughness: float, rng) -> Polygon:
    """
    Generate a random irregular convex-ish polygon centred at (cx, cy).
    mean_r is in degrees (~111 km/°). roughness controls shape irregularity.
    """
    angles = np.sort(rng.uniform(0, 2 * math.pi, n_verts))
    radii  = mean_r * rng.uniform(1 - roughness, 1 + roughness, n_verts)
    coords = [(cx + r * math.cos(a), cy + r * math.sin(a))
              for r, a in zip(radii, angles)]
    coords.append(coords[0])
    poly = Polygon(coords)
    # Random rotation for realism
    angle_deg = rng.uniform(0, 90)
    return rotate(poly, angle_deg, origin='centroid')


@st.cache_data(show_spinner=False)
def generate_synthetic_dataset():
    """
    Build 20 synthetic East Anglian fields with correlated daily soil moisture
    time series and monthly payout calculations.

    Returns
    -------
    gdf        : GeoDataFrame  — field polygons with metadata
    monthly_df : DataFrame     — monthly aggregated (payout, sm, triggered)
    daily_df   : DataFrame     — daily soil moisture per field
    """
    rng = np.random.default_rng(2024)

    # ── 1. Field layout ──────────────────────────────────────────────────────
    # Load real FiBOA-UK field boundaries if the parquet file exists,
    # otherwise fall back to 20 synthetic polygons.
    import os
    FIBOA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "ukfields.parquet")

    # East Anglia bounding box in British National Grid (EPSG:27700) metres
    # derived from: lon 0.1–1.0, lat 52.0–53.0
    EA_WEST, EA_EAST   = 541000, 607000   # Easting range
    EA_SOUTH, EA_NORTH = 235000, 350000   # Northing range
    MAX_FIELDS = 150   # cap for dashboard performance

    if os.path.exists(FIBOA_PATH):
        # Load and clip to East Anglia bounding box
        raw = gpd.read_parquet(FIBOA_PATH)   # CRS: EPSG:27700
        bbox_mask = (
            (raw.geometry.bounds["minx"] >= EA_WEST)  &
            (raw.geometry.bounds["maxx"] <= EA_EAST)  &
            (raw.geometry.bounds["miny"] >= EA_SOUTH) &
            (raw.geometry.bounds["maxy"] <= EA_NORTH)
        )
        ea_fields = raw[bbox_mask].copy()

        # area column is already in hectares in this dataset
        # Filter to sensible field sizes (0.5–200 ha) and sample for performance
        ea_fields["area_ha"] = ea_fields["area"].round(1)
        ea_fields = ea_fields[
            (ea_fields["area_ha"] >= 0.5) & (ea_fields["area_ha"] <= 200)
        ]
        if len(ea_fields) > MAX_FIELDS:
            ea_fields = ea_fields.sample(MAX_FIELDS, random_state=42)

        # Reproject to WGS84 for the map
        ea_fields = ea_fields.to_crs("EPSG:4326")
        ea_fields["field_id"] = [f"F{i+1:03d}" for i in range(len(ea_fields))]
        ea_fields["centroid_lat"] = ea_fields.geometry.centroid.y
        ea_fields["centroid_lon"] = ea_fields.geometry.centroid.x
        gdf = ea_fields[["field_id", "geometry", "area_ha",
                          "centroid_lat", "centroid_lon"]].reset_index(drop=True)
    else:
        # ── Synthetic fallback: 20 irregular polygons ────────────────────
        CENTRE_LAT, CENTRE_LON = 52.50, 0.50
        N_FIELDS = 20
        grid_cols, grid_rows = 5, 4
        lat_step, lon_step = 0.025, 0.030

        field_records = []
        field_id = 0
        for row in range(grid_rows):
            for col in range(grid_cols):
                if field_id >= N_FIELDS:
                    break
                base_lat = CENTRE_LAT + (row - grid_rows / 2) * lat_step
                base_lon = CENTRE_LON + (col - grid_cols / 2) * lon_step
                lat = base_lat + rng.uniform(-0.008, 0.008)
                lon = base_lon + rng.uniform(-0.010, 0.010)
                target_ha = rng.uniform(10, 50)
                deg_r = math.sqrt(target_ha * 1e4) / 111_300 * 0.6
                n_verts = rng.integers(5, 9)
                poly = _irregular_polygon(lon, lat, n_verts, deg_r, 0.35, rng)
                gpoly = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326").to_crs("EPSG:27700")
                area_ha = gpoly.geometry.area.iloc[0] / 1e4
                field_records.append({
                    "field_id":     f"F{field_id+1:02d}",
                    "geometry":     poly,
                    "area_ha":      round(area_ha, 1),
                    "centroid_lat": poly.centroid.y,
                    "centroid_lon": poly.centroid.x,
                })
                field_id += 1
        gdf = gpd.GeoDataFrame(field_records, crs="EPSG:4326")

    # ── 2. Daily soil moisture time series ──────────────────────────────────
    # Load real Sentinel-1 derived soil moisture if the CSV exists,
    # otherwise fall back to a fully synthetic time series.
    SM_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "sentinel1_soil_moisture.csv")

    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    n_days = len(dates)
    n_fields = len(gdf)

    # Spatial correlation matrix (used in both paths below)
    lons_f = gdf["centroid_lon"].values
    lats_f = gdf["centroid_lat"].values
    dist_scale = 0.08
    corr_matrix = np.zeros((n_fields, n_fields))
    for i in range(n_fields):
        for j in range(n_fields):
            d = math.sqrt((lons_f[i]-lons_f[j])**2 + (lats_f[i]-lats_f[j])**2)
            corr_matrix[i, j] = math.exp(-d / dist_scale)
    L = np.linalg.cholesky(corr_matrix + 1e-6 * np.eye(n_fields))

    if os.path.exists(SM_CSV):
        # ── Real Sentinel-1 data path ────────────────────────────────────
        s1 = pd.read_csv(SM_CSV, parse_dates=["date"])
        s1 = s1.set_index("date")["sm_m3m3"]
        s1_monthly = s1.reindex(pd.date_range(s1.index.min(), s1.index.max(), freq="MS"))
        s1_daily = s1_monthly.reindex(dates).interpolate(method="linear")
        regional_signal = s1_daily.values

        # Small per-field variation: soil type bias + spatially correlated noise
        field_offsets = rng.normal(0, 0.018, n_fields)
        innovations   = rng.normal(0, 0.012, (n_days, n_fields))
        corr_noise    = (L @ innovations.T).T

        sm = np.zeros((n_days, n_fields))
        for t in range(n_days):
            sm[t, :] = np.clip(
                regional_signal[t] + field_offsets + corr_noise[t], 0.05, 0.50
            )
    else:
        # ── Fully synthetic fallback ─────────────────────────────────────
        doy = dates.day_of_year.values
        seasonal = 0.28 + 0.10 * np.cos(2 * np.pi * (doy - 15) / 365)
        phi = 0.93
        innovations = rng.normal(0, 0.038, (n_days, n_fields))
        corr_innovations = (L @ innovations.T).T
        drought_episodes = [
            ("2020-04-01", "2020-06-15", -0.055),
            ("2021-07-01", "2021-09-30", -0.075),
            ("2022-06-01", "2022-09-15", -0.090),
            ("2023-07-15", "2023-09-30", -0.065),
        ]
        drought_signal = np.zeros(n_days)
        for s, e, mag in drought_episodes:
            si = np.searchsorted(dates, pd.Timestamp(s))
            ei = np.searchsorted(dates, pd.Timestamp(e))
            ep = min(15, (ei - si) // 4)
            ramp = np.ones(ei - si)
            ramp[:ep] = np.linspace(0, 1, ep)
            ramp[-ep:] = np.linspace(1, 0, ep)
            drought_signal[si:ei] += mag * ramp
        sm = np.zeros((n_days, n_fields))
        for t in range(1, n_days):
            base = seasonal[t] + drought_signal[t]
            noise = phi * (sm[t-1, :] - seasonal[t-1] - drought_signal[t-1]) + corr_innovations[t]
            sm[t, :] = np.clip(base + noise, 0.05, 0.50)
        sm[0, :] = np.clip(seasonal[0] + rng.normal(0, 0.03, n_fields), 0.05, 0.50)

    daily_df = pd.DataFrame(sm, index=dates, columns=gdf["field_id"].values)

    # ── 3. Payout calculation ────────────────────────────────────────────────
    TRIGGER_PCTL = 20
    MAX_PAYOUT_PCTL = 5
    MAX_PAYOUT = 10_000.0

    records = []
    for fid in gdf["field_id"]:
        sm_s = daily_df[fid]
        roll30 = sm_s.rolling(30, min_periods=15).mean()

        # Monthly percentile thresholds (using full series)
        monthly_p20 = roll30.groupby(roll30.index.month).quantile(TRIGGER_PCTL / 100)
        monthly_p05 = roll30.groupby(roll30.index.month).quantile(MAX_PAYOUT_PCTL / 100)

        thr_p20 = roll30.index.month.map(monthly_p20.to_dict())
        thr_p05 = roll30.index.month.map(monthly_p05.to_dict())

        # Daily payout
        triggered = roll30 < thr_p20
        frac = np.where(
            roll30 >= thr_p20, 0.0,
            np.where(roll30 <= thr_p05, 1.0,
                     (thr_p20 - roll30) / np.maximum(thr_p20 - thr_p05, 1e-9))
        )
        daily_payout = pd.Series(np.clip(frac * MAX_PAYOUT, 0, MAX_PAYOUT),
                                 index=roll30.index)

        # Group to month, cap annual
        tmp = pd.DataFrame({
            "field_id":    fid,
            "date":        roll30.index,
            "sm":          sm_s.values,
            "roll30":      roll30.values,
            "triggered":   triggered.values,
            "daily_payout": daily_payout.values,
        })
        tmp["year"]  = tmp["date"].dt.year
        tmp["month"] = tmp["date"].dt.month
        tmp["ym"]    = tmp["date"].dt.to_period("M")

        # Annual cap per field
        annual_payouts = tmp.groupby("year")["daily_payout"].sum().clip(upper=MAX_PAYOUT)
        annual_scale   = annual_payouts / tmp.groupby("year")["daily_payout"].sum().replace(0, np.nan)

        # Monthly aggregation
        monthly = tmp.groupby("ym").agg(
            sm_mean=("sm", "mean"),
            triggered=("triggered", "any"),
            daily_payout_sum=("daily_payout", "sum"),
        ).reset_index()
        monthly["year"] = monthly["ym"].dt.year
        monthly["month_payout"] = (
            monthly["daily_payout_sum"]
            * monthly["year"].map(annual_scale).fillna(1.0)
        )
        monthly["field_id"] = fid
        records.append(monthly)

    monthly_df = pd.concat(records, ignore_index=True)
    monthly_df["ym_str"] = monthly_df["ym"].dt.strftime("%Y-%m")

    return gdf, monthly_df, daily_df


# ─────────────────────────────────────────────────────────────────────────────
# FLAME DATA LOADER
# Loads owner-beneficiary and farm data from FLAME_dataset_v2.xlsx and
# spatially joins the nearest farm to each field polygon centroid.
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_flame_data(gdf_json: str) -> pd.DataFrame:
    """
    Load FLAME dataset and join nearest farm + owner to each field.
    gdf_json is a JSON string of field centroids (for cache-key purposes).
    Returns a DataFrame indexed by field_id with owner/farm attributes.
    """
    import os, json
    from shapely.geometry import Point

    FLAME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "FLAME_dataset_v2.xlsx")
    if not os.path.exists(FLAME_PATH):
        return pd.DataFrame()

    xl = pd.ExcelFile(FLAME_PATH)

    # Load farms with coordinates
    farms = xl.parse("Farm list")
    farms = farms.dropna(subset=["Latitude", "Longitude"])
    farms = farms.rename(columns={
        "Farm ID":    "farm_id",
        "Latitude":   "lat",
        "Longitude":  "lon",
        "Address":    "address",
        "Postcode":   "postcode",
        "Farm type":  "farm_type",
        "Area (ha)":  "farm_area_ha",
        "Crop(s)":    "crops",
    })

    # Load owner-beneficiary list
    owners = xl.parse("Owner-beneficiary list")
    owners = owners.rename(columns={
        "Owner-beneficiary ID":           "owner_id",
        "Owner-beneficiary name":         "owner_name",
        "Company registration number":    "company_reg",
        "Registered address":             "owner_address",
        "Nature of business name (1)":    "business_type",
        "Area (ha)":                      "owner_area_ha",
        "Crop(s)":                        "owner_crops",
    })

    # Load owner crops with yields
    owner_crops = xl.parse("Owner-beneficiary crops")
    owner_crops = owner_crops.rename(columns={
        "Owner-beneficiary ID":  "owner_id",
        "Crop(s)":               "crop",
        "Area (ha)":             "crop_area_ha",
        "Total yield (tonnes)":  "yield_tonnes",
    })
    # Summarise top crop and total yield per owner
    owner_summary = (
        owner_crops.groupby("owner_id")
        .apply(lambda x: pd.Series({
            "top_crop":       x.loc[x["crop_area_ha"].idxmax(), "crop"] if len(x) else "Unknown",
            "total_yield_t":  round(x["yield_tonnes"].sum(), 1),
            "n_crops":        x["crop"].nunique(),
        }), include_groups=False)
        .reset_index()
    )

    # Merge owner data into farms using a hash-based assignment
    # (no direct Farm→Owner link in dataset; in production use Land Registry title)
    # We create a stable pseudo-random mapping so each farm consistently gets
    # a plausible owner with real company reg, address and business type
    import hashlib
    def _pick_owner(farm_id, owners_df):
        h = int(hashlib.md5(farm_id.encode()).hexdigest(), 16)
        idx = h % len(owners_df)
        return owners_df.iloc[idx]

    farms["owner_id"]     = farms["farm_id"].apply(lambda x: _pick_owner(x, owners)["owner_id"])
    farms["owner_name"]   = farms["farm_id"].apply(lambda x: _pick_owner(x, owners)["owner_name"])
    farms["company_reg"]  = farms["farm_id"].apply(lambda x: _pick_owner(x, owners)["company_reg"])
    farms["business_type"]= farms["farm_id"].apply(lambda x: _pick_owner(x, owners)["business_type"])

    farms = farms.merge(owner_summary, on="owner_id", how="left")

    # Build GeoDataFrame of farm points for spatial join
    farms_gdf = gpd.GeoDataFrame(
        farms,
        geometry=gpd.points_from_xy(farms["lon"], farms["lat"]),
        crs="EPSG:4326",
    )

    # Reconstruct field centroids from JSON
    centroids = pd.DataFrame(json.loads(gdf_json))
    centroids_gdf = gpd.GeoDataFrame(
        centroids,
        geometry=gpd.points_from_xy(centroids["centroid_lon"], centroids["centroid_lat"]),
        crs="EPSG:4326",
    )

    # Spatial join: each field gets the nearest farm (no distance cap —
    # dataset coverage is sparse so we always want the closest match)
    joined = gpd.sjoin_nearest(
        centroids_gdf[["field_id", "geometry"]],
        farms_gdf[["farm_id", "owner_name", "company_reg", "business_type",
                   "address", "postcode", "farm_type", "farm_area_ha",
                   "crops", "top_crop", "total_yield_t", "n_crops", "geometry"]],
        how="left",
        distance_col="dist_deg",
    )

    return joined.set_index("field_id").drop(columns=["geometry", "index_right"], errors="ignore")


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _lerp_color(t, c0, c1):
    """Linear interpolate between two RGB tuples."""
    return [int(c0[i] + t * (c1[i] - c0[i])) for i in range(3)]


def payout_color(amount: float) -> list:
    """Grey → yellow → red depending on payout amount (0–10000)."""
    if amount <= 0:
        return [45, 65, 85, 180]
    t = min(amount / 10_000, 1.0)
    if t < 0.5:
        rgb = _lerp_color(t * 2, [230, 190, 50], [240, 140, 40])
    else:
        rgb = _lerp_color((t - 0.5) * 2, [240, 140, 40], [210, 40, 30])
    return rgb + [210]


def trigger_color(triggered: bool) -> list:
    return [200, 45, 35, 210] if triggered else [45, 65, 85, 170]


def moisture_color(sm: float) -> list:
    """Light blue (dry) → dark blue (wet); sm roughly 0.05–0.45."""
    t = np.clip((sm - 0.05) / 0.40, 0, 1)
    rgb = _lerp_color(t, [120, 195, 230], [10, 60, 140])
    return rgb + [210]


# ─────────────────────────────────────────────────────────────────────────────
# POLYGON → PYDECK FORMAT
# ─────────────────────────────────────────────────────────────────────────────

def gdf_to_pydeck_rows(gdf, monthly_slice, colour_mode, flame_df=None):
    """
    Merge field geometries with monthly data and FLAME owner attributes,
    assign colours, return list of dicts for pdk.Layer("PolygonLayer", ...).
    """
    rows = []
    sm_lookup      = monthly_slice.set_index("field_id")["sm_mean"].to_dict()
    payout_lookup  = monthly_slice.set_index("field_id")["month_payout"].to_dict()
    trigger_lookup = monthly_slice.set_index("field_id")["triggered"].to_dict()

    for _, row in gdf.iterrows():
        fid  = row["field_id"]
        sm   = sm_lookup.get(fid, 0.25)
        pay  = payout_lookup.get(fid, 0.0)
        trig = trigger_lookup.get(fid, False)

        if colour_mode == "Payout Amount":
            colour = payout_color(pay)
        elif colour_mode == "Trigger Status":
            colour = trigger_color(trig)
        else:
            colour = moisture_color(sm)

        # FLAME owner/farm attributes (if available)
        flame = flame_df.loc[fid] if (flame_df is not None and fid in flame_df.index) else None
        owner_name   = str(flame["owner_name"])   if flame is not None and pd.notna(flame.get("owner_name"))   else "Unknown"
        company_reg  = str(flame["company_reg"])  if flame is not None and pd.notna(flame.get("company_reg"))  else "—"
        address      = str(flame["address"])      if flame is not None and pd.notna(flame.get("address"))      else "—"
        top_crop     = str(flame["top_crop"])     if flame is not None and pd.notna(flame.get("top_crop"))     else "—"
        total_yield  = f"{flame['total_yield_t']:,.0f}" if flame is not None and pd.notna(flame.get("total_yield_t")) else "—"
        biz_type     = str(flame["business_type"]) if flame is not None and pd.notna(flame.get("business_type")) else "—"

        exterior = list(row["geometry"].exterior.coords)
        rows.append({
            "field_id":    fid,
            "area_ha":     row["area_ha"],
            "sm":          round(sm, 4),
            "triggered":   "Yes" if trig else "No",
            "payout_gbp":  round(pay, 0),
            "polygon":     exterior,
            "fill_color":  colour,
            "owner_name":  owner_name,
            "company_reg": company_reg,
            "address":     address,
            "top_crop":    top_crop,
            "total_yield": total_yield,
            "biz_type":    biz_type,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("Loading field data…"):
    gdf, monthly_df, daily_df = generate_synthetic_dataset()

# Load FLAME owner/farm data and join to field centroids
_gdf_json = gdf[["field_id","centroid_lat","centroid_lon"]].to_json(orient="records")
with st.spinner("Loading FLAME owner data…"):
    flame_df = load_flame_data(_gdf_json)

all_periods = sorted(monthly_df["ym_str"].unique())
period_labels = pd.to_datetime([p + "-01" for p in all_periods]).strftime("%b %Y").tolist()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="echo-header">
  <div class="echo-wordmark">Echo<span>.</span>Labs</div>
  <div class="echo-subtitle">Field-Level Parametric Insurance Explorer</div>
  <div class="echo-pill">Sentinel-1 + FLAME</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: APP LAYOUT — two columns: map (left) + controls (right)
# ─────────────────────────────────────────────────────────────────────────────

col_map, col_ctrl = st.columns([3, 1], gap="medium")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: CONTROLS (right column)
# ─────────────────────────────────────────────────────────────────────────────

with col_ctrl:
    st.markdown('<div class="section-label">Time Period</div>', unsafe_allow_html=True)
    period_idx = st.slider(
        "Month",
        min_value=0,
        max_value=len(all_periods) - 1,
        value=24,                            # Jan 2022
        format="%d",
        label_visibility="collapsed",
    )
    selected_ym  = all_periods[period_idx]
    selected_label = period_labels[period_idx]
    st.markdown(f"<div style='text-align:center; font-family:Syne,sans-serif; font-size:1.1rem; font-weight:700; color:#e8f4ff; margin:-0.3rem 0 0.6rem;'>{selected_label}</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Colour Scheme</div>', unsafe_allow_html=True)
    colour_mode = st.selectbox(
        "Colour mode",
        ["Payout Amount", "Trigger Status", "Soil Moisture"],
        label_visibility="collapsed",
    )

    # ── Summary statistics ──────────────────────────────────────────────────
    month_data = monthly_df[monthly_df["ym_str"] == selected_ym]
    n_triggered     = int(month_data["triggered"].sum())
    total_payout    = float(month_data["month_payout"].sum())
    avg_sm          = float(month_data["sm_mean"].mean())
    pct_triggered   = 100 * n_triggered / len(gdf)

    st.markdown('<div class="section-label">Portfolio Snapshot</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-label">Fields Triggered</div>
        <div class="stat-value trigger">{n_triggered} <small style="font-size:0.7rem;color:#4a7a9b">/ {len(gdf)}</small></div>
        <div class="stat-sub">{pct_triggered:.0f}% of portfolio</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Total Payout</div>
        <div class="stat-value payout">£{total_payout:,.0f}</div>
        <div class="stat-sub">Monthly aggregate</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Avg Soil Moisture</div>
        <div class="stat-value moisture">{avg_sm:.3f}</div>
        <div class="stat-sub">m³/m³ across portfolio</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Colour Legend</div>', unsafe_allow_html=True)
    if colour_mode == "Payout Amount":
        st.markdown("""
        <div style="font-size:0.65rem; color:#8aa8c0; line-height:2;">
        <span style="color:#2d4155">■</span> No payout<br>
        <span style="color:#e6be32">■</span> Low (£1–£3,000)<br>
        <span style="color:#f08c28">■</span> Medium (£3–£7,000)<br>
        <span style="color:#d2281e">■</span> High (£7–£10,000)
        </div>""", unsafe_allow_html=True)
    elif colour_mode == "Trigger Status":
        st.markdown("""
        <div style="font-size:0.65rem; color:#8aa8c0; line-height:2;">
        <span style="color:#2d4155">■</span> No trigger<br>
        <span style="color:#c82d23">■</span> Trigger active
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size:0.65rem; color:#8aa8c0; line-height:2;">
        <span style="color:#78c3e6">■</span> Dry (0.05 m³/m³)<br>
        <span style="color:#3a7eb5">■</span> Moderate<br>
        <span style="color:#0a3c8c">■</span> Wet (0.45 m³/m³)
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: MAP RENDERING (left column)
# ─────────────────────────────────────────────────────────────────────────────

with col_map:
    pydeck_rows = gdf_to_pydeck_rows(gdf, month_data, colour_mode, flame_df=flame_df if not flame_df.empty else None)

    polygon_layer = pdk.Layer(
        "PolygonLayer",
        data=pydeck_rows,
        get_polygon="polygon",
        get_fill_color="fill_color",
        get_line_color=[180, 210, 240, 160],
        line_width_min_pixels=1,
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 255, 60],
    )

    view_state = pdk.ViewState(
        latitude=52.500,
        longitude=0.500,
        zoom=11.2,
        pitch=0,
        bearing=0,
    )

    tooltip = {
        "html": """
        <div style="font-family:'DM Mono',monospace; background:#0d1a28;
                    border:1px solid #1e3448; border-radius:6px;
                    padding:0.7rem 0.9rem; font-size:0.72rem; color:#c8d0dc;
                    max-width:280px;">
          <div style="font-family:Syne,sans-serif; font-size:0.85rem;
                      font-weight:700; color:#e8f4ff; margin-bottom:0.5rem;">
            {owner_name}
          </div>
          <div style="color:#4a7a9b; margin-bottom:0.1rem;">COMPANY REG</div>
          <div style="margin-bottom:0.4rem;">{company_reg}</div>
          <div style="color:#4a7a9b; margin-bottom:0.1rem;">ADDRESS</div>
          <div style="margin-bottom:0.4rem; font-size:0.65rem;">{address}</div>
          <div style="color:#4a7a9b; margin-bottom:0.1rem;">BUSINESS TYPE</div>
          <div style="margin-bottom:0.4rem;">{biz_type}</div>
          <div style="border-top:1px solid #1e3448; margin:0.4rem 0;"></div>
          <div style="color:#4a7a9b; margin-bottom:0.1rem;">PRIMARY CROP</div>
          <div style="margin-bottom:0.4rem;">{top_crop}</div>
          <div style="color:#4a7a9b; margin-bottom:0.1rem;">TOTAL YIELD</div>
          <div style="margin-bottom:0.4rem;">{total_yield} tonnes</div>
          <div style="border-top:1px solid #1e3448; margin:0.4rem 0;"></div>
          <div style="color:#4a7a9b; margin-bottom:0.1rem;">FIELD AREA</div>
          <div style="margin-bottom:0.4rem;">{area_ha} ha</div>
          <div style="color:#4a7a9b; margin-bottom:0.1rem;">SOIL MOISTURE</div>
          <div style="margin-bottom:0.4rem;">{sm} m³/m³</div>
          <div style="color:#4a7a9b; margin-bottom:0.1rem;">TRIGGERED</div>
          <div style="margin-bottom:0.4rem;">{triggered}</div>
          <div style="color:#4a7a9b; margin-bottom:0.1rem;">PAYOUT</div>
          <div style="color:#f87c6a; font-weight:500; font-size:0.85rem;">£{payout_gbp}</div>
        </div>
        """,
        "style": {"background": "transparent", "border": "none"},
    }

    deck = pdk.Deck(
        layers=[polygon_layer],
        initial_view_state=view_state,
        map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
        tooltip=tooltip,
    )

    st.pydeck_chart(deck, use_container_width=True, height=520)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: TIME SERIES CHART (below map, full width)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-label" style="margin-top:1.5rem;">Portfolio Time Series · 2020–2023</div>',
            unsafe_allow_html=True)

# Monthly portfolio aggregates
portfolio = (
    monthly_df.groupby("ym_str")
    .agg(total_payout=("month_payout", "sum"), avg_sm=("sm_mean", "mean"))
    .reset_index()
)
portfolio["date"] = pd.to_datetime(portfolio["ym_str"] + "-01")
portfolio = portfolio.sort_values("date")

fig = go.Figure()

# ── Payout bars (area fill for a cleaner look) ──────────────────────────────
fig.add_trace(go.Scatter(
    x=portfolio["date"],
    y=portfolio["total_payout"],
    name="Portfolio Payout",
    fill="tozeroy",
    fillcolor="rgba(210,40,30,0.15)",
    line=dict(color="#f87c6a", width=2),
    yaxis="y1",
    hovertemplate="<b>%{x|%b %Y}</b><br>Payout: £%{y:,.0f}<extra></extra>",
))

# ── Soil moisture line (right axis) ─────────────────────────────────────────
fig.add_trace(go.Scatter(
    x=portfolio["date"],
    y=portfolio["avg_sm"],
    name="Avg Soil Moisture",
    line=dict(color="#3aecb0", width=1.8, dash="dot"),
    yaxis="y2",
    hovertemplate="<b>%{x|%b %Y}</b><br>Soil moisture: %{y:.3f} m³/m³<extra></extra>",
))

# ── Selected month marker ────────────────────────────────────────────────────
sel_date = pd.Timestamp(selected_ym + "-01")
fig.add_vline(
    x=sel_date.timestamp() * 1000,
    line_color="#f7c94b",
    line_width=1.5,
    line_dash="dot",
    annotation_text=selected_label,
    annotation_font_color="#f7c94b",
    annotation_font_size=10,
    annotation_position="top right",
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0a0e14",
    plot_bgcolor="#0d1a28",
    height=240,
    margin=dict(l=10, r=10, t=20, b=30),
    font=dict(family="DM Mono", size=10, color="#8aa8c0"),
    legend=dict(
        orientation="h", x=0, y=1.12,
        font=dict(size=10), bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,0,0,0)",
    ),
    xaxis=dict(
        showgrid=False,
        color="#4a7a9b",
        tickformat="%b %Y",
    ),
    yaxis=dict(
        title=dict(text="Portfolio Payout (£)", font=dict(color="#f87c6a", size=10)),
        tickfont=dict(color="#f87c6a"),
        showgrid=True,
        gridcolor="#131f2e",
        tickformat="£,.0f",
    ),
    yaxis2=dict(
        title=dict(text="Avg Soil Moisture (m³/m³)", font=dict(color="#3aecb0", size=10)),
        tickfont=dict(color="#3aecb0"),
        overlaying="y",
        side="right",
        showgrid=False,
    ),
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="echo-footer">
  All data shown is entirely synthetic and generated for illustrative purposes only.
  Not for investment, insurance pricing, or regulatory decisions.
  Echo Labs · Field-Level Parametric Insurance Explorer
</div>
""", unsafe_allow_html=True)
