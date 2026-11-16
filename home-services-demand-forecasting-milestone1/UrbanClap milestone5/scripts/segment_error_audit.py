"""
City/service segment error audit.

Expected input: reports/holdout_predictions.csv
Required columns: city, service_type, actual, predicted

Export actual and predicted values for the same holdout dates before running.
This is a segment-performance audit, not a demographic fairness audit.
"""
from pathlib import Path
import numpy as np
import pandas as pd

INPUT = Path("reports/holdout_predictions.csv")
REPORTS = Path("reports")
REQUIRED = {"city", "service_type", "actual", "predicted"}

def summarize(group):
    actual = group["actual"].astype(float)
    predicted = group["predicted"].astype(float)
    abs_error = (actual - predicted).abs()
    nonzero = actual != 0
    mape = ((abs_error[nonzero] / actual[nonzero].abs()) * 100).mean()
    return pd.Series({
        "observations": len(group),
        "zero_actual_rows": int((~nonzero).sum()),
        "MAE": abs_error.mean(),
        "RMSE": np.sqrt(((actual - predicted) ** 2).mean()),
        "MAPE_pct_nonzero_actuals": mape,
    })

def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"{INPUT} is missing. Export holdout actual/predicted values first."
        )

    df = pd.read_csv(INPUT)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=list(REQUIRED)).copy()
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    df["predicted"] = pd.to_numeric(df["predicted"], errors="coerce")
    df = df.dropna(subset=["actual", "predicted"])
    df = df[df["actual"] >= 0].copy()
    if df.empty:
        raise ValueError("No valid holdout rows remain after cleaning.")

    REPORTS.mkdir(parents=True, exist_ok=True)
    for filename, keys in [
        ("error_by_city.csv", ["city"]),
        ("error_by_service.csv", ["service_type"]),
        ("error_by_city_service.csv", ["city", "service_type"]),
    ]:
        result = df.groupby(keys).apply(summarize).reset_index()
        result.to_csv(REPORTS / filename, index=False)
        print(f"\n{filename}\n{result.to_string(index=False)}")

if __name__ == "__main__":
    main()
