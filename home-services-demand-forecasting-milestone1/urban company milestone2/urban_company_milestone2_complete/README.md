# Urban Company Demand Forecasting — Milestone 2

This repository contains the executed Milestone 2 data pipeline and baseline model for the supplied Urban Company bookings dataset.

## Dataset
- 29,200 daily city-service records
- 2025-10-01 through 2026-09-30
- 8 cities and 10 service categories
- Target: `bookings` (already aggregated daily)

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the workflow
From the project root:

```bash
python run_project.py
```

Or run each stage:

```bash
python src/data_pipeline.py
python src/train_baseline.py
```

## Main outputs
- `data/processed_daily_bookings.csv`
- `reports/daily_booking_trend.png`
- `reports/bookings_by_city.png`
- `reports/bookings_by_service.png`
- `reports/average_bookings_by_weather.png`
- `reports/model_metrics.json`
- `reports/test_predictions.csv`
- `reports/feature_importance.csv`
- `src/baseline_model.joblib`

## Baseline
The package compares a lag-7 seasonal-naive benchmark with a Random Forest regressor. Evaluation uses the final 28 calendar days as a chronological rolling one-step-ahead holdout. See `reports/baseline_report.md` for the actual results and limitations.

## Important scope note
This is the Milestone 2 baseline deliverable. The LSTM, Flask API, weekly retraining scheduler and drift alert are later project stages and are not represented as completed here.
