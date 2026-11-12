"""Validate, clean, aggregate and visualize the Urban Company bookings data."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "aiml_urbanclap_bookings.csv"
OUTPUT = ROOT / "data" / "processed_daily_bookings.csv"
REPORTS = ROOT / "reports"

REQUIRED = [
    "booking_date", "service_type", "city", "bookings",
    "weather_condition", "temperature_c", "is_weekend",
    "is_holiday", "rain_mm", "humidity_pct"
]

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT}")

    df = pd.read_csv(INPUT)
    missing_columns = [c for c in REQUIRED if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print("Raw shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Missing values by column:\\n", df.isna().sum().to_string())

    df["booking_date"] = pd.to_datetime(df["booking_date"], errors="coerce")
    df["bookings"] = pd.to_numeric(df["bookings"], errors="coerce")
    df = df.dropna(subset=["booking_date", "service_type", "city", "bookings"]).copy()
    df["city"] = df["city"].astype(str).str.strip()
    df["service_type"] = df["service_type"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["booking_date", "city", "service_type"])
    df["booking_date"] = df["booking_date"].dt.floor("D")

    # The source is already daily aggregated: preserve and sum its bookings field.
    daily = (
        df.groupby(["booking_date", "city", "service_type"], as_index=False)
          .agg(
              bookings=("bookings", "sum"),
              weather_condition=("weather_condition", "first"),
              temperature_c=("temperature_c", "mean"),
              is_weekend=("is_weekend", "max"),
              is_holiday=("is_holiday", "max"),
              rain_mm=("rain_mm", "mean"),
              humidity_pct=("humidity_pct", "mean"),
          )
          .sort_values(["city", "service_type", "booking_date"])
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUTPUT, index=False)

    print("\\nProcessed shape:", daily.shape)
    print("Date range:", daily.booking_date.min().date(), "to", daily.booking_date.max().date())
    print("Cities:", daily.city.nunique(), "| Services:", daily.service_type.nunique())
    print("Unique city-service-date keys:", daily[["booking_date", "city", "service_type"]].drop_duplicates().shape[0])
    print("Saved:", OUTPUT)

    # EDA figures
    daily_total = daily.groupby("booking_date")["bookings"].sum()
    plt.figure(figsize=(12, 5))
    daily_total.plot()
    plt.title("Daily Urban Company Bookings — All Cities and Services")
    plt.xlabel("Date"); plt.ylabel("Bookings"); plt.tight_layout()
    plt.savefig(REPORTS / "daily_booking_trend.png", dpi=160); plt.close()

    city_total = daily.groupby("city")["bookings"].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    city_total.plot(kind="bar")
    plt.title("Total Bookings by City")
    plt.xlabel("City"); plt.ylabel("Bookings"); plt.tight_layout()
    plt.savefig(REPORTS / "bookings_by_city.png", dpi=160); plt.close()

    service_total = daily.groupby("service_type")["bookings"].sum().sort_values(ascending=False)
    plt.figure(figsize=(11, 5))
    service_total.plot(kind="bar")
    plt.title("Total Bookings by Service")
    plt.xlabel("Service"); plt.ylabel("Bookings")
    plt.xticks(rotation=35, ha="right"); plt.tight_layout()
    plt.savefig(REPORTS / "bookings_by_service.png", dpi=160); plt.close()

    weather_total = daily.groupby("weather_condition")["bookings"].mean().sort_values(ascending=False)
    plt.figure(figsize=(9, 5))
    weather_total.plot(kind="bar")
    plt.title("Average Daily Bookings by Weather Condition")
    plt.xlabel("Weather condition"); plt.ylabel("Mean bookings")
    plt.xticks(rotation=25, ha="right"); plt.tight_layout()
    plt.savefig(REPORTS / "average_bookings_by_weather.png", dpi=160); plt.close()

    print("Saved EDA charts in reports/")

if __name__ == "__main__":
    main()
