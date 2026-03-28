# EpiWatch — AI-Powered Epidemic Spread Predictor

> **CodeCure 2026** · Track C: Epidemic Spread Prediction  
> SPIRIT 2026 · IIT (BHU) Varanasi

---

## Overview

**EpiWatch** is an AI-powered epidemic forecasting dashboard that predicts the spread of infectious diseases using real-world epidemiological data. It ingests the Johns Hopkins University (JHU) COVID-19 global time-series dataset, trains a Facebook Prophet forecasting model per country, computes outbreak risk scores, and presents everything through a clean, interactive Streamlit web application.

The goal is simple: give public health decision-makers a **30-day forward view** of outbreak trajectories — before a surge becomes a crisis.

---

## Live Demo

> Run locally using the instructions below.  
> The dashboard opens automatically at `http://localhost:8501`

---

## Screenshots

### Global Risk Map
Color intensity shows projected outbreak risk across all countries based on 14-day case growth rate and volume.

### 30-Day Forecast
Per-country Prophet forecast with actual vs predicted cases and 80% confidence intervals.

### Wave Detection
Automatic identification of distinct outbreak waves with peak markers.

### Top At-Risk Countries
Ranked table of countries by risk score, growth rate, and average daily cases.

---

## Features

- **Global risk map** — Interactive choropleth world map coloured by outbreak risk score
- **30-day case forecast** — Facebook Prophet model with confidence intervals per country
- **Wave detection** — Automatic identification of outbreak peaks using rolling average analysis
- **Risk ranking** — Top at-risk countries ranked by growth rate × case volume formula
- **Historical trends** — Multi-country case trend comparison since 2021
- **Fully interactive** — Country selector, forecast horizon slider, live KPI cards
- **201 countries tracked** — Full global coverage from JHU dataset

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| Data source | JHU CSSE COVID-19 Dataset (live from GitHub) |
| ML model | Facebook Prophet |
| Visualisation | Plotly Express + Graph Objects |
| Dashboard | Streamlit |
| Data processing | Pandas, NumPy |

---

## Project Structure

```
EpiWatch/
├── app.py              # Streamlit dashboard — UI, charts, map
├── model.py            # Data loading, preprocessing, forecasting, risk scoring
├── evaluate.py         # Model evaluation — MAE, RMSE, MAPE metrics
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/RV-Bhavika3/Epiwatch.git
cd Epiwatch
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> Prophet requires pystan under the hood. If installation fails, try:
> `pip install prophet --no-build-isolation`

### 4. Run the dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## How It Works

### Data Pipeline

```
JHU CSV (wide format: countries × dates)
    ↓
Melt → long format (date, country, cumulative_cases)
    ↓
.diff() → daily new cases
    ↓
7-day rolling average → smooth weekend reporting gaps
    ↓
Per-country time series ready for modelling
```

### Forecasting Model — Facebook Prophet

Prophet is a decomposable time-series model developed by Meta that handles:
- **Trend** — piecewise linear with automatic changepoint detection
- **Seasonality** — yearly and weekly patterns
- **Missing values and outliers** — natively without preprocessing

One model is trained per country on the **last 2 years** of data and forecasts daily new cases for the next N days with 80% confidence intervals.

### Risk Scoring Formula

```
growth_rate = (avg cases last 14 days − avg cases prior 14 days)
              ÷ (prior 14-day avg + 1) × 100

risk_score  = growth_rate × log(1 + avg_daily_cases) / 10
```

This formula identifies countries with both **high absolute case counts AND fast growth** — the most dangerous combination for public health systems.

### Wave Detection

Outbreak waves are detected using a local peak algorithm on the 7-day rolling average:
- A point is flagged as a peak if it is higher than both its neighbours
- Only peaks above the 75th percentile of all case values are marked
- This filters out noise and highlights true outbreak surges

---

## Model Evaluation

Evaluated on a 30-day held-out test set per country:

| Country | MAE (cases/day) | RMSE (cases/day) |
|---|---|---|
| India | 211 | 224 |
| United States | 36,723 | 36,812 |
| Brazil | 7,268 | 8,131 |
| France | 3,569 | 3,582 |
| Germany | 9,012 | 10,944 |

> MAE and RMSE are in absolute daily case counts. Lower is better.  
> Run `python evaluate.py` to regenerate these results.

---

## Datasets Used

| Dataset | Source | Usage |
|---|---|---|
| JHU COVID-19 Time Series | [CSSEGISandData/COVID-19](https://github.com/CSSEGISandData/COVID-19) | Primary — confirmed cases globally |
| Our World in Data COVID-19 | [owid/covid-19-data](https://github.com/owid/covid-19-data) | Optional enrichment |

---

## Judging Criteria Addressed

| Criterion | How EpiWatch addresses it |
|---|---|
| Functionality | Live dashboard with real data, working forecast, interactive map |
| Innovation | Wave detection + risk scoring formula not in standard solutions |
| Code quality | Modular structure: model.py separated from app.py, commented throughout |
| Scalability | Works for all 201 countries, forecast horizon is adjustable |
| Biological insight | Risk score combines epidemiological factors (growth rate + volume) |

---

## Team

Built for **CodeCure 2026**, Track C — Epidemic Spread Prediction  
Event: SPIRIT 2026, IIT (BHU) Varanasi

---

## License

MIT
