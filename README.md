# Echo Labs — Field-Level Parametric Insurance Explorer

Interactive geospatial dashboard visualising synthetic field-level parametric insurance payouts across a simulated East Anglian agricultural landscape.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Python 3.10+** is recommended. Use a virtual environment to avoid conflicts:
> ```bash
> python -m venv .venv && source .venv/bin/activate
> pip install -r requirements.txt
> ```

### 2. Run the app

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## What It Does

On load the app generates a synthetic dataset of **20 irregular field polygons** in East Anglia (centred ~52.5°N, 0.5°E) with:

- **Daily soil moisture** for Jan 2020 – Dec 2023, modelled as a spatially correlated AR(1) process with a realistic winter-peak seasonal cycle and injected drought episodes
- **Monthly payouts** per field using a parametric trigger: 30-day rolling mean below the monthly 20th percentile percentile triggers a linear payout up to £10,000/field/year

### Controls

| Control | Description |
|---|---|
| **Time slider** | Select any month from Jan 2020 to Dec 2023 |
| **Colour scheme** | Switch map colouring between Payout Amount, Trigger Status, or Soil Moisture |

### Panels

- **Map** — interactive Pydeck map with satellite basemap; hover any field for details
- **Summary cards** — fields triggered, total payout, and average soil moisture for the selected month
- **Time series chart** — full 2020–2023 portfolio payout (left axis) + average soil moisture (right axis); selected month shown as a vertical marker

---

## Switching to Real Data

The synthetic data generation block (`generate_synthetic_dataset()`) can be replaced with:

1. **ERA5-Land soil moisture** downloaded via `cdsapi` (see `parametric_insurance_model.py`)
2. **Sentinel-1 SAR** derived soil moisture at 10–20 m resolution for field-scale precision
3. **Real field boundaries** loaded from a GeoJSON file via `gpd.read_file()`

---

## Notes

All data is entirely synthetic and for illustrative purposes only. Not for investment, insurance pricing, or regulatory decisions.
