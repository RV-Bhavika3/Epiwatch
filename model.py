"""
model.py — Core ML logic for EpiWatch
Handles: data loading, preprocessing, Prophet forecasting, risk scoring
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

JHU_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
    "time_series_covid19_confirmed_global.csv"
)

COUNTRY_ALIASES = {
    "US": "United States",
    "Korea, South": "South Korea",
    "Taiwan*": "Taiwan",
    "Burma": "Myanmar",
    "West Bank and Gaza": "Palestine",
}


def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(JHU_URL)
        return df
    except Exception as e:
        print(f"Warning: could not fetch live data ({e}). Using synthetic fallback.")
        return _synthetic_fallback()


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.rename(columns={
        "Country/Region": "country",
        "Province/State": "province",
        "Lat": "lat",
        "Long": "lon"
    })
    raw["country"] = raw["country"].replace(COUNTRY_ALIASES)

    date_cols = [c for c in raw.columns if c not in ("province", "country", "lat", "lon")]
    country_df = raw.groupby("country")[date_cols].sum().reset_index()

    long = country_df.melt(id_vars="country", var_name="date", value_name="cumulative")
    long["date"] = pd.to_datetime(long["date"], format="%m/%d/%y")
    long = long.sort_values(["country", "date"]).reset_index(drop=True)

    long["cases"] = long.groupby("country")["cumulative"].diff().fillna(0)
    long["cases"] = long["cases"].clip(lower=0).astype(int)

    long["cases_smooth"] = (
        long.groupby("country")["cases"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )

    long = long[long["date"] >= "2020-03-01"]
    return long[["date", "country", "cases", "cases_smooth"]]


def train_and_forecast(df_long, country, forecast_days=30):
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
    risk["growth_rate"] = ((risk["avg_recent"] - risk["avg_prior"]) / (risk["avg_prior"] + 1)) * 100
    risk["avg_daily_cases"] = risk["avg_recent"]
    risk["risk_score"] = (
        risk["growth_rate"].clip(lower=0) * np.log1p(risk["avg_daily_cases"]) / 10
    )

    risk = risk.reset_index()
    risk = risk.sort_values("risk_score", ascending=False).head(top_n)
    risk = risk[risk["avg_daily_cases"] > 10]
    return risk.reset_index(drop=True)


def get_country_list(df_long):
    return sorted(df_long["country"].unique().tolist())


def _simple_forecast_fallback(df_long, country, forecast_days):
    country_data = df_long[df_long["country"] == country][["date", "cases_smooth"]].copy()
    country_data = country_data.rename(columns={"date": "ds", "cases_smooth": "y"})
    if len(country_data) < 30:
        return None, None

    last_val  = country_data["y"].ewm(span=14).mean().iloc[-1]
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
