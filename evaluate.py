"""
evaluate.py — Model evaluation for EpiWatch
Run this to get MAE, RMSE, MAPE accuracy metrics.
Use these numbers in your slides and README.

Usage:
    python evaluate.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from model import load_data, preprocess


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred):
    mask = y_true > 10
    if mask.sum() == 0:
        return float("nan")
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_country(df_long, country, test_days=30):
    try:
        from prophet import Prophet
    except ImportError:
        print("Prophet not installed. Run: pip install prophet")
        return None

    data = df_long[df_long["country"] == country].copy()
    data = data[["date", "cases_smooth"]].rename(
        columns={"date": "ds", "cases_smooth": "y"}
    ).dropna()

    if len(data) < test_days + 60:
        print(f"  Skipping {country}: not enough data.")
        return None

    split_date = data["ds"].max() - pd.Timedelta(days=test_days)
    train = data[data["ds"] <= split_date]
    test  = data[data["ds"] >  split_date]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.15,
        interval_width=0.80
    )
    model.fit(train)

    future   = model.make_future_dataframe(periods=test_days, freq="D")
    forecast = model.predict(future)
    forecast["yhat"] = forecast["yhat"].clip(lower=0)

    preds   = forecast[forecast["ds"].isin(test["ds"])].set_index("ds")["yhat"]
    actuals = test.set_index("ds")["y"]
    aligned = pd.concat([actuals, preds], axis=1).dropna()
    aligned.columns = ["actual", "predicted"]

    return {
        "Country": country,
        "MAE":     mae(aligned["actual"].values,  aligned["predicted"].values),
        "RMSE":    rmse(aligned["actual"].values, aligned["predicted"].values),
        "MAPE (%)":mape(aligned["actual"].values, aligned["predicted"].values),
        "Test days": test_days
    }


if __name__ == "__main__":
    COUNTRIES = ["India", "United States", "Brazil", "France", "Germany"]

    print("=" * 55)
    print("EpiWatch — Model Evaluation")
    print("=" * 55)
    print("Loading data...")

    raw     = load_data()
    df_long = preprocess(raw)
    print(f"Data range: {df_long['date'].min().date()} to {df_long['date'].max().date()}\n")

    results = []
    for country in COUNTRIES:
        print(f"Evaluating {country}...")
        result = evaluate_country(df_long, country, test_days=30)
        if result:
            results.append(result)
            print(f"  MAE:  {result['MAE']:>10,.0f} cases/day")
            print(f"  RMSE: {result['RMSE']:>10,.0f} cases/day")
            print(f"  MAPE: {result['MAPE (%)']:>10.1f} %")
        print()

    if results:
        summary = pd.DataFrame(results)
        print("=" * 55)
        print("SUMMARY TABLE")
        print("=" * 55)
        print(summary.to_string(index=False))
        summary.to_csv("evaluation_results.csv", index=False)
        print("\nSaved → evaluation_results.csv")
