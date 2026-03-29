# EpiWatch — AI-Powered Epidemic Spread Predictor

> **CodeCure 2026** · Track C: Epidemic Spread Prediction  
> SPIRIT 2026 · IIT (BHU) Varanasi

---

## Live Demo

👉 **[https://epiwatch-codecure2026.streamlit.app/](https://epiwatch-codecure2026.streamlit.app/)**

> Full interactive dashboard — no installation required.  
> Also runs locally using the instructions below.

---

## Overview

**EpiWatch** is an AI-powered epidemic forecasting dashboard that predicts the spread of infectious diseases using real-world epidemiological data. It integrates two datasets — the Johns Hopkins University (JHU) COVID-19 global time-series dataset and the Our World in Data (OWID) COVID-19 dataset — trains a Facebook Prophet forecasting model per country, computes outbreak risk scores, and presents everything through a clean, interactive Streamlit web application.

The goal: give public health decision-makers a **30-day forward view** of outbreak trajectories — before a surge becomes a crisis.

---

## Features

- **Global risk map** — Interactive choropleth world map coloured by outbreak risk score across 201 countries
- **30-day case forecast** — Facebook Prophet model with actual vs predicted cases and 80% confidence intervals
- **Vaccination vs Cases chart** — Dual-axis chart showing how vaccination rollout correlated with case decline (OWID data)
- **Cases vs Deaths chart** — Visualises the 2–3 week lag between infections and deaths (JHU deaths data)
- **Wave detection** — Automatic identification of distinct outbreak peaks using rolling average analysis
- **Historical trends** — Multi-country case trend comparison for top 5 at-risk countries
- **Risk ranking table** — Top at-risk countries ranked by growth rate × case volume formula
- **Biological insight boxes** — Plain-language epidemiological interpretation under every chart
- **Fully interactive** — Country selector, forecast horizon slider (7–60 days), countries-to-rank slider
- **2 datasets integrated** — JHU (primary) + Our World in Data (secondary)

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| Primary dataset | JHU CSSE COVID-19 Dataset (live from GitHub) |
| Secondary dataset | Our World in Data COVID-19 Dataset (live from GitHub) |
| ML model | Facebook Prophet |
| Visualisation | Plotly Express + Graph Objects |
| Dashboard | Streamlit |
| Data processing | Pandas, NumPy |

---

## Project Structure

```
EpiWatch/
├── app.py              # Streamlit dashboard — all UI, charts, map
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

> If Prophet installation fails, try:
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
JHU Cases CSV (wide format: countries × dates)
    ↓
Melt → long format (date, country, cumulative_cases)
    ↓
.diff() → daily new cases
    ↓
7-day rolling average → smooth weekend reporting gaps
    ↓
Merged with JHU Deaths + OWID vaccination data
    ↓
Per-country time series ready for modelling
```

### Forecasting Model — Facebook Prophet

Prophet is a decomposable time-series model developed by Meta that handles:
- **Trend** — piecewise linear with automatic changepoint detection
- **Seasonality** — yearly and weekly patterns built in
- **Missing values and outliers** — handled natively without preprocessing

One model is trained per country on the **last 2 years** of data and forecasts daily new cases for the next N days with 80% confidence intervals.

### Risk Scoring Formula

```
growth_rate = (avg cases last 14 days − avg cases prior 14 days)
              ÷ (prior 14-day avg + 1) × 100

risk_score  = growth_rate × log(1 + avg_daily_cases) / 10
```

This formula identifies countries with both **high absolute case counts AND fast growth** — the most dangerous combination for public health systems.

### Wave Detection Algorithm

Outbreak waves are detected using a local peak algorithm on the 7-day rolling average:
- A point is a peak if it exceeds both its immediate neighbours
- Only peaks above the 75th percentile of all values are marked
- This filters noise and highlights true outbreak surges

### Biological Insights

Every chart includes a plain-language biological interpretation:
- **Vaccination vs Cases** — whether vaccine-induced herd immunity reduced transmission
- **Cases vs Deaths** — the 2–3 week disease progression lag from infection to severe outcome
- **Wave Detection** — variant emergence and breakdown of population immunity

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

> Run `python evaluate.py` to regenerate these results.

---

## Datasets Used

| Dataset | Source | Usage |
|---|---|---|
| JHU COVID-19 Time Series (Cases) | [CSSEGISandData/COVID-19](https://github.com/CSSEGISandData/COVID-19) | Primary — confirmed cases globally |
| JHU COVID-19 Time Series (Deaths) | [CSSEGISandData/COVID-19](https://github.com/CSSEGISandData/COVID-19) | Primary — death counts globally |
| Our World in Data COVID-19 | [owid/covid-19-data](https://github.com/owid/covid-19-data) | Secondary — vaccination, hospitalisation, R-value |

---

## Judging Criteria Addressed

| Criterion | How EpiWatch addresses it |
|---|---|
| Functionality | Live dashboard with real data, working forecast, interactive map, 6 chart sections |
| Innovation | Wave detection + risk scoring formula + vaccination correlation analysis |
| Code quality | Modular structure: model.py separated from app.py, fully commented |
| Scalability | Works for all 201 countries, forecast horizon fully adjustable |
| Biological insight | Vaccination vs cases, deaths lag, wave detection — all with plain-language interpretation boxes |
| Dataset usage | Both primary (JHU) and secondary (OWID) datasets fully integrated |

---

## Team

Built for **CodeCure 2026**, Track C — Epidemic Spread Prediction  
Event: SPIRIT 2026, IIT (BHU) Varanasi

---

## License

MIT
