"""Train/evaluate a Random Forest baseline against a weekly seasonal-naive model."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed_daily_bookings.csv"
MODEL_PATH = ROOT / "src" / "baseline_model.joblib"
REPORTS = ROOT / "reports"

def mape_pct(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    nonzero = actual != 0
    if not nonzero.any():
        return None, 0
    value = np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100
    return float(value), int(nonzero.sum())

def add_features(frame):
    frame = frame.sort_values(["city", "service_type", "booking_date"]).copy()
    groups = frame.groupby(["city", "service_type"], sort=False)["bookings"]
    for lag in [1, 7, 14, 28]:
        frame[f"lag_{lag}"] = groups.shift(lag)
    group_bookings = frame.groupby(["city", "service_type"])["bookings"]
    frame["rolling_mean_7"] = group_bookings.transform(
        lambda s: s.shift(1).rolling(7, min_periods=7).mean()
    )
    frame["rolling_mean_28"] = group_bookings.transform(
        lambda s: s.shift(1).rolling(28, min_periods=28).mean()
    )
    frame["rolling_std_7"] = group_bookings.transform(
        lambda s: s.shift(1).rolling(7, min_periods=7).std()
    )
    frame["day_of_week_num"] = frame["booking_date"].dt.dayofweek
    frame["month"] = frame["booking_date"].dt.month
    frame["day_of_month"] = frame["booking_date"].dt.day
    frame["week_of_year"] = frame["booking_date"].dt.isocalendar().week.astype(int)
    frame["day_of_year"] = frame["booking_date"].dt.dayofyear
    return frame

def main():
    if not INPUT.exists():
        raise FileNotFoundError("Processed data missing. Run python src/data_pipeline.py first.")
    df = pd.read_csv(INPUT, parse_dates=["booking_date"])
    df = add_features(df)

    # Final 28 calendar days are held out; all model inputs are historical/calendar/weather fields.
    cutoff = df["booking_date"].max() - pd.Timedelta(days=27)
    required_lags = ["lag_1", "lag_7", "lag_14", "lag_28", "rolling_mean_7",
                     "rolling_mean_28", "rolling_std_7"]
    usable = df.dropna(subset=required_lags).copy()
    train = usable[usable.booking_date < cutoff].copy()
    test = usable[usable.booking_date >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Insufficient history for feature engineering and 28-day holdout.")

    categorical = ["city", "service_type", "weather_condition"]
    numeric = [
        "lag_1", "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_mean_28", "rolling_std_7",
        "day_of_week_num", "month", "day_of_month", "week_of_year", "day_of_year",
        "is_weekend", "is_holiday", "temperature_c", "rain_mm", "humidity_pct"
    ]
    features = categorical + numeric
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", "passthrough", numeric)
    ])
    model = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, max_features=0.9,
            random_state=42, n_jobs=-1
        ))
    ])
    model.fit(train[features], train["bookings"])
    rf_prediction = np.maximum(model.predict(test[features]), 0)
    naive_prediction = np.maximum(test["lag_7"].to_numpy(), 0)
    actual = test["bookings"].to_numpy()

    rf_mape, rf_nonzero = mape_pct(actual, rf_prediction)
    naive_mape, naive_nonzero = mape_pct(actual, naive_prediction)
    metrics = {
        "evaluation_protocol": "Chronological holdout: final 28 calendar days; daily rolling one-step predictions",
        "holdout_start": str(test.booking_date.min().date()),
        "holdout_end": str(test.booking_date.max().date()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "city_service_series": int(test[["city", "service_type"]].drop_duplicates().shape[0]),
        "random_forest": {
            "MAPE_percent": rf_mape, "MAPE_nonzero_actual_count": rf_nonzero,
            "MAE": float(mean_absolute_error(actual, rf_prediction)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, rf_prediction)))
        },
        "seasonal_naive_lag_7": {
            "MAPE_percent": naive_mape, "MAPE_nonzero_actual_count": naive_nonzero,
            "MAE": float(mean_absolute_error(actual, naive_prediction)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, naive_prediction)))
        },
        "MAPE_target_percent": 12.0,
        "random_forest_meets_MAPE_target": bool(rf_mape is not None and rf_mape <= 12.0),
        "weather_feature_caveat": "Evaluation uses recorded weather values for holdout dates. Production forecasts require weather forecasts or a weather-free fallback."
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    (REPORTS / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    predictions = test[["booking_date", "city", "service_type", "bookings"]].copy()
    predictions["seasonal_naive_prediction"] = naive_prediction
    predictions["random_forest_prediction"] = rf_prediction
    predictions.to_csv(REPORTS / "test_predictions.csv", index=False)

    # Feature importance from fitted RF pipeline.
    fitted_pre = model.named_steps["preprocess"]
    names = fitted_pre.get_feature_names_out()
    importances = model.named_steps["regressor"].feature_importances_
    pd.DataFrame({"feature": names, "importance": importances}).sort_values(
        "importance", ascending=False
    ).to_csv(REPORTS / "feature_importance.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print("Saved model:", MODEL_PATH)
    print("Saved metrics and predictions in reports/")

if __name__ == "__main__":
    main()
