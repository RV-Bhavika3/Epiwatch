"""
model.py — Core ML logic for EpiWatch
Handles: data loading, preprocessing, Prophet forecasting, risk scoring
Datasets: JHU COVID-19 (primary) + Our World in Data (secondary)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Dataset URLs ──────────────────────────────────────────────────────────────
JHU_CASES_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
    "time_series_covid19_confirmed_global.csv"
)

JHU_DEATHS_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
    "time_series_covid19_deaths_global.csv"
)

OWID_URL = (
    "https://raw.githubusercontent.com/owid/covid-19-data/master/"
    "public/data/owid-covid-data.csv"
)

COUNTRY_ALIASES = {
    "US": "United States",
    "Korea, South": "South Korea",
    "Taiwan*": "Taiwan",
    "Burma": "Myanmar",
    "West Bank and Gaza": "Palestine",
}

OWID_COUNTRY_ALIASES = {
    "United States": "United States",
    "South Korea": "South Korea",
}


def load_data():
    """Load and return JHU cases, JHU deaths, and OWID data."""
    # JHU Cases
    try:
        cases_raw = pd.read_csv(JHU_CASES_URL)
    except Exception as e:
        print(f"Warning: could not fetch JHU cases ({e}). Using fallback.")
        cases_raw = _synthetic_fallback()

    # JHU Deaths
    try:
        deaths_raw = pd.read_csv(JHU_DEATHS_URL)
    except Exception as e:
        print(f"Warning: could not fetch JHU deaths ({e}).")
        deaths_raw = None

    # OWID
    try:
        owid_raw = pd.read_csv(
            OWID_URL,
            usecols=[
                "location", "date",
                "new_vaccinations_smoothed",
                "people_fully_vaccinated_per_hundred",
                "hosp_patients",
                "icu_patients",
                "new_deaths_smoothed",
                "reproduction_rate",
                "stringency_index"
            ],
            low_memory=False
        )
        owid_raw["date"] = pd.to_datetime(owid_raw["date"])
        owid_raw = owid_raw[owid_raw["location"] != "World"]
        print("OWID dataset loaded successfully.")
    except Exception as e:
        print(f"Warning: could not fetch OWID data ({e}).")
        owid_raw = None

    return cases_raw, deaths_raw, owid_raw


def preprocess(cases_raw, deaths_raw=None):
    """
    Convert JHU wide-format CSV to long-format tidy DataFrame.
    Returns daily new cases with 7-day smoothing.
    """
    def _process_jhu(raw, value_name):
        raw = raw.rename(columns={
            "Country/Region": "country",
            "Province/State": "province",
            "Lat": "lat", "Long": "lon"
        })
        raw["country"] = raw["country"].replace(COUNTRY_ALIASES)
        date_cols = [c for c in raw.columns if c not in ("province", "country", "lat", "lon")]
        country_df = raw.groupby("country")[date_cols].sum().reset_index()
        long = country_df.melt(id_vars="country", var_name="date", value_name=value_name)
        long["date"] = pd.to_datetime(long["date"], format="%m/%d/%y")
        long = long.sort_values(["country", "date"]).reset_index(drop=True)
        long[f"daily_{value_name}"] = long.groupby("country")[value_name].diff().fillna(0)
        long[f"daily_{value_name}"] = long[f"daily_{value_name}"].clip(lower=0)
        long[f"{value_name}_smooth"] = (
            long.groupby("country")[f"daily_{value_name}"]
            .transform(lambda x: x.rolling(7, min_periods=1).mean())
        )
        return long[["date", "country", f"daily_{value_name}", f"{value_name}_smooth"]]

    cases_long = _process_jhu(cases_raw, "cases")

    if deaths_raw is not None:
        deaths_long = _process_jhu(deaths_raw, "deaths")
        df = cases_long.merge(deaths_long, on=["date", "country"], how="left")
    else:
        df = cases_long
        df["daily_deaths"] = 0
        df["deaths_smooth"] = 0

    df = df[df["date"] >= "2020-03-01"]
    return df


def get_owid_country(owid_raw, country):
    """Get OWID data for a specific country."""
    if owid_raw is None:
        return None
    country_data = owid_raw[owid_raw["location"] == country].copy()
    if len(country_data) == 0:
        return None
    return country_data.sort_values("date").reset_index(drop=True)


def train_and_forecast(df_long, country, forecast_days=30):
    """Train Prophet model and forecast for a country."""
    try:
        from prophet import Prophet
    except ImportError:
        return _simple_forecast_fallback(df_long, country, forecast_days)

    country_data = df_long[df_long["country"] == country].copy()
    country_data = country_data[["date", "cases_smooth"]].rename(
        columns={"date": "ds", "cases_smooth": "y"}
    ).dropna()

    if len(country_data) < 60:
        return None, None

    cutoff = country_data["ds"].max() - pd.Timedelta(days=730)
    train = country_data[country_data["ds"] >= cutoff].copy()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.15,
        seasonality_prior_scale=10,
        interval_width=0.80
    )
    model.fit(train)

    future = model.make_future_dataframe(periods=forecast_days, freq="D")
    forecast = model.predict(future)

    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[col] = forecast[col].clip(lower=0)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], train


def compute_risk_scores(df_long, top_n=20):
    """Compute risk scores for all countries."""
    latest = df_long["date"].max()
    cutoff_recent = latest - pd.Timedelta(days=14)
    cutoff_prior  = latest - pd.Timedelta(days=28)

    recent = (
        df_long[df_long["date"] > cutoff_recent]
        .groupby("country")["cases_smooth"].mean().rename("avg_recent")
    )
    prior = (
        df_long[(df_long["date"] > cutoff_prior) & (df_long["date"] <= cutoff_recent)]
        .groupby("country")["cases_smooth"].mean().rename("avg_prior")
    )

    risk = pd.concat([recent, prior], axis=1).dropna()
    risk["growth_rate"] = (
        (risk["avg_recent"] - risk["avg_prior"]) / (risk["avg_prior"] + 1)
    ) * 100
    risk["avg_daily_cases"] = risk["avg_recent"]
    risk["risk_score"] = (
        risk["growth_rate"].clip(lower=0) * np.log1p(risk["avg_daily_cases"]) / 10
    )

    risk = risk.reset_index()
    risk = risk.sort_values("risk_score", ascending=False).head(top_n)
    risk = risk[risk["avg_daily_cases"] > 10]
    return risk.reset_index(drop=True)


def compute_case_fatality_rate(df_long):
    """Compute case fatality rate (CFR) per country."""
    if "deaths_smooth" not in df_long.columns:
        return pd.DataFrame()

    latest = df_long["date"].max()
    cutoff = latest - pd.Timedelta(days=30)
    recent = df_long[df_long["date"] > cutoff]

    cfr = recent.groupby("country").agg(
        avg_cases=("cases_smooth", "mean"),
        avg_deaths=("deaths_smooth", "mean")
    ).reset_index()

    cfr = cfr[cfr["avg_cases"] > 10]
    cfr["cfr"] = (cfr["avg_deaths"] / (cfr["avg_cases"] + 1)) * 100
    cfr["cfr"] = cfr["cfr"].clip(0, 20)
    return cfr.sort_values("cfr", ascending=False).reset_index(drop=True)


def get_country_list(df_long):
    return sorted(df_long["country"].unique().tolist())


def _simple_forecast_fallback(df_long, country, forecast_days):
    country_data = df_long[df_long["country"] == country][["date", "cases_smooth"]].copy()
    country_data = country_data.rename(columns={"date": "ds", "cases_smooth": "y"})
    if len(country_data) < 30:
        return None, None
    last_val = country_data["y"].ewm(span=14).mean().iloc[-1]
    last_date = country_data["ds"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    forecast = pd.DataFrame({
        "ds": future_dates,
        "yhat": last_val,
        "yhat_lower": last_val * 0.7,
        "yhat_upper": last_val * 1.3
    })
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], country_data


def _synthetic_fallback():
    dates = pd.date_range("2022-01-01", periods=365, freq="D")
    date_strs = [d.strftime("%-m/%-d/%y") for d in dates]
    countries = ["India", "United States", "Brazil", "France", "Germany"]
    rows = []
    for country in countries:
        base = np.random.randint(10000, 100000)
        noise = np.cumsum(np.random.randn(len(dates)) * 500 + base * 0.001)
        cumulative = np.maximum(0, noise + base).astype(int)
        row = {"Province/State": "", "Country/Region": country, "Lat": 0, "Long": 0}
        for d, v in zip(date_strs, cumulative):
            row[d] = v
        rows.append(row)
    return pd.DataFrame(rows)
