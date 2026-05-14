<div align="center">

# Crop Yield Prediction System

**Machine Learning · FastAPI · Random Forest · FAO + NASA Data**

[![Python](https://img.shields.io/badge/Python-11%25-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-89%25-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Active%20Development-22C55E?style=flat-square)]()

</div>

---

## Overview

An end-to-end machine learning system for predicting agricultural crop yields in Pakistan using 25 years of FAO crop production data merged with NASA POWER climate records. The system engineers quarterly seasonal weather features, lag-based historical signals, and a Random Forest regression pipeline — served via a production-grade FastAPI REST API.

Built to support farmer decision-making, government food security planning, and agricultural supply chain optimization.

📦 **Repo:** [github.com/Hamid-GenAI-Eng/crop_yield_project](https://github.com/Hamid-GenAI-Eng/crop_yield_project)

> ⚠️ **Active Development** — Model experimentation and API enhancements are ongoing.

---

## System Architecture
```
DATA SOURCES
├── FAO Crop Database        → Yield, Production, Area Harvested (2000–2024)
└── NASA POWER Climate Data  → Temperature, Rainfall, Humidity (Monthly)
ETL PIPELINE (build_production_model.py)
├── Step 1: Load & filter FAO data (Pakistan · 5 crops)
├── Step 2: Reshape crop data (Wide → Long)
├── Step 3: Aggregate weather to quarterly seasonal features
├── Step 4: Engineer lag features (Yield_Lag1)
└── Step 5: Merge datasets → 120-row training set
ML PIPELINE
├── Preprocessing
│   ├── StandardScaler     → 16 weather features + 1 lag feature
│   └── OneHotEncoder      → Crop type (5 binary columns)
├── Model: RandomForestRegressor (n_estimators=200, max_depth=12)
└── Evaluation: 5-Fold Cross Validation → R² > 0.85
SERVING LAYER (FastAPI)
├── POST /predict  → Crop yield forecast
└── GET  /health   → System status

```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python · Jupyter Notebook |
| **ML Framework** | scikit-learn · XGBoost |
| **Data Processing** | pandas · numpy |
| **Visualization** | matplotlib · plotly · seaborn |
| **API Framework** | FastAPI + Uvicorn |
| **Validation** | Pydantic v2 |
| **Model Serialization** | Joblib |
| **Data Sources** | FAO FAOSTAT · NASA POWER |

---

## Project Structure
```
crop_yield_project/
├── data/
│   └── raw/
│       ├── Production_Crops_Livestock_E_All_Data_NOFLAG.csv  # FAO dataset
│       └── POWER_NASA_DATA.csv                               # NASA climate data
│
├── src/
│   ├── build_production_model.py    # ETL pipeline + model training
│   ├── train.py                     # Stacking ensemble experiments
│   ├── preprocessing.py             # Feature engineering utilities
│   ├── eda_feature-engineering.py   # Exploratory analysis scripts
│   ├── data__prep.py                # Data preparation helpers
│   ├── preprocessing.ipynb          # Preprocessing notebook
│   └── model_analysis.ipynb         # Model evaluation notebook
│
├── models/
│   └── crop_yield_model.pkl         # Trained Random Forest pipeline
│
├── app/
│   └── main.py                      # FastAPI application
│
└── requirements.txt

```

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/Hamid-GenAI-Eng/crop_yield_project.git
cd crop_yield_project
pip install -r requirements.txt
```

### Train the Model

```bash
python src/build_production_model.py
# Outputs: models/crop_yield_model.pkl
```

### Run the API

```bash
# Development
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production (4 workers)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

API docs auto-generated at:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

## API Reference

### `POST /predict` — Yield Forecast

**Request Body:**

```json
{
  "item": "Wheat",
  "yield_lag1": 25000.0,
  "prectotcorr_q1": 45.5, "t2m_max_q1": 26.5, "t2m_min_q1": 10.2, "rh2m_q1": 55.0,
  "prectotcorr_q2": 12.0, "t2m_max_q2": 38.0, "t2m_min_q2": 22.0, "rh2m_q2": 30.0,
  "prectotcorr_q3": 150.0,"t2m_max_q3": 35.0, "t2m_min_q3": 25.0, "rh2m_q3": 65.0,
  "prectotcorr_q4": 10.0, "t2m_max_q4": 22.0, "t2m_min_q4": 8.0,  "rh2m_q4": 45.0
}
```

**Response:**

```json
{
  "status": "success",
  "crop": "Wheat",
  "predicted_yield_hg_ha": 26500.50,
  "predicted_yield_tonnes_ha": 2.65,
  "yield_change_percent": 6.02
}
```

### `GET /health` — System Status

```json
{
  "status": "active",
  "model_version": "v1.0.0_RandomForest"
}
```

---

## Supported Crops

| Crop | FAO Name | Typical Yield (hg/ha) | Season |
|---|---|---|---|
| **Wheat** | Wheat | 20,000 – 30,000 | Rabi (Oct–Mar) |
| **Rice** | Rice, paddy | 40,000 – 50,000 | Kharif (Jun–Nov) |
| **Maize** | Maize (corn) | 50,000 – 70,000 | Kharif |
| **Sugarcane** | Sugar cane | 50,000 – 70,000 | Year-round |
| **Cotton** | Seed cotton, unginned | 8,000 – 15,000 | Kharif |

---

## Feature Engineering

### Quarterly Weather Features (16 total)

Pakistan's agricultural calendar drives the seasonality design:

| Quarter | Months | Agricultural Context |
|---|---|---|
| Q1 | Jan – Mar | Rabi crop growth (Winter Wheat) |
| Q2 | Apr – Jun | Pre-Monsoon transition |
| Q3 | Jul – Sep | Kharif crop growth (Monsoon Rice/Cotton) |
| Q4 | Oct – Dec | Post-Monsoon / Wheat sowing prep |

Each quarter captures: `Rainfall (sum) · Max Temp · Min Temp · Humidity (mean)`

### Lag Feature

`Yield_Lag1` (previous year's yield) encodes soil health, farmer practice continuity, and state dependencies across seasons.

---

## Model Performance

| Metric | Value |
|---|---|
| **Algorithm** | Random Forest Regressor |
| **n_estimators** | 200 |
| **max_depth** | 12 |
| **Validation** | 5-Fold Cross Validation |
| **CV R² Score** | ~0.87 |
| **MAE** | ~1,500 – 2,000 hg/ha |
| **Training Samples** | ~120 rows (5 crops × 24 years) |

---

## Dataset Summary

| Source | Coverage | Key Variables |
|---|---|---|
| FAO FAOSTAT | 2000–2024 · Pakistan · 5 crops | Yield · Production · Area Harvested |
| NASA POWER | Monthly climate · Pakistan | Rainfall · Max/Min Temp · Humidity |

---

## Use Cases

- **Farmer Decision Support** — Predict yield from seasonal weather forecasts
- **Government Food Security** — National yield projections for stockpile planning
- **Agricultural Insurance** — Premium calculation and claims validation
- **Supply Chain Optimization** — Market price and logistics planning
- **Research & Breeding Programs** — Climate-yield relationship analysis

---

## Built By

**[Code Envision Technologies](https://codeenvisiontechnologies.com)**

Developed by **Hamid Saifullah** — Tech Lead at [Code Envision Technologies](https://codeenvisiontechnologies.com)

[![GitHub](https://img.shields.io/badge/GitHub-Hamid--GenAI--Eng-181717?style=flat-square&logo=github)](https://github.com/Hamid-GenAI-Eng)
[![Portfolio](https://img.shields.io/badge/Portfolio-hamid--saifullah-black?style=flat-square&logo=vercel)](https://hamid-saifullah-portfolio-nexus.vercel.app)
[![Code Envision Technologies](https://img.shields.io/badge/Company-Code%20Envision%20Technologies-0A66C2?style=flat-square)](https://codeenvisiontechnologies.com)
