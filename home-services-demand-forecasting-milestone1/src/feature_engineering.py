"""Feature engineering for city/service demand forecasting."""
import pandas as pd

def add_calendar_features(df, date_col="date"):
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["day_of_week"] = out[date_col].dt.dayofweek
    out["day_of_month"] = out[date_col].dt.day
    out["week_of_year"] = out[date_col].dt.isocalendar().week.astype(int)
    out["month"] = out[date_col].dt.month
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    return out

def add_lag_rolling_features(df, target="bookings", group_cols=("city","service_type")):
    out = df.copy()
    out = out.sort_values([*group_cols, "date"])
    grouped = out.groupby(list(group_cols), group_keys=False)[target]
    for lag in (1, 7, 14, 28):
        out[f"lag_{lag}"] = grouped.shift(lag)
    for window in (7, 14, 28):
        out[f"rolling_mean_{window}"] = grouped.transform(
            lambda s: s.shift(1).rolling(window).mean()
        )
    return out

def build_features(df):
    out = add_calendar_features(df)
    out = add_lag_rolling_features(out)
    return out
