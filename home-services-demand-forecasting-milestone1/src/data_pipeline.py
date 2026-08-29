"""Data loading and validation utilities for the demand forecasting pipeline."""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def load_booking_data(path=None):
    """Load the future booking dataset.

    The current supplied CSV is an intent-classification dataset, so this function
    intentionally validates expected forecasting columns before use.
    """
    path = Path(path) if path else DATA_DIR / "aiml_training_data.csv"
    df = pd.read_csv(path)
    required = {"date", "city", "service_type", "bookings"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Forecasting dataset is missing required columns: {sorted(missing)}. "
            "The supplied CSV is not a booking time series."
        )
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["city", "service_type", "date"]).reset_index(drop=True)

if __name__ == "__main__":
    try:
        print(load_booking_data().head())
    except ValueError as exc:
        print(exc)
