# Urban Company Demand Forecasting — Milestone 2 Report

## 1. Objective
Build the initial data ingestion, cleaning, exploratory analysis, feature engineering and baseline demand forecasting workflow for a home-services marketplace. The target variable is daily booking volume for each city and service type.

## 2. Dataset overview
- Source: `data/aiml_urbanclap_bookings.csv`
- Records: **29,200**
- Date range: **2025-10-01 to 2026-09-30** (365 calendar days)
- Cities: **8**
- Service categories: **10**
- Source granularity: daily aggregated bookings (the `hour` field is `Daily` throughout)
- Target: existing numeric `bookings` field
- Missing values in supplied CSV: **0** across the inspected columns
- Duplicate (date, city, service) keys: **0**
- Coverage: 80 city-service series, each with 365 daily records

The data is already aggregated at daily city-service level. Therefore, the pipeline preserves the supplied `bookings` values and does not count rows as individual bookings.

## 3. Data ingestion and preprocessing
Implemented in `src/data_pipeline.py`:
1. Load the supplied CSV.
2. Validate required columns.
3. Parse `booking_date` and coerce `bookings` to numeric.
4. Remove invalid rows and duplicate date-city-service keys.
5. Aggregate any duplicate keys defensively, summing booking counts and retaining weather/calendar fields.
6. Save `data/processed_daily_bookings.csv`.

No imputation was needed for the supplied source because the inspected columns contain no missing values.

## 4. Exploratory data analysis
The pipeline generates:
- `reports/daily_booking_trend.png`
- `reports/bookings_by_city.png`
- `reports/bookings_by_service.png`
- `reports/average_bookings_by_weather.png`

These plots are included in this package and can be used to discuss time trends, city/service differences and observed weather associations.

## 5. Feature engineering
Historical demand features are computed separately for each city-service series:
- Lag 1, 7, 14 and 28 days
- Previous 7-day rolling mean
- Previous 28-day rolling mean
- Previous 7-day rolling standard deviation
- Day of week, month, day of month, week of year and day of year
- City, service type and observed weather condition
- Weekend, holiday, temperature, rainfall and humidity

All lag and rolling features are shifted by one day so the current target is not used to construct its own features.

## 6. Baseline models
Two approaches are compared:
1. **Seasonal-naive benchmark:** today's forecast equals the same city-service series' bookings from seven days earlier.
2. **Random Forest Regressor:** 300 trees, `min_samples_leaf=2`, `max_features=0.9`, `random_state=42`; categorical variables are one-hot encoded in a scikit-learn pipeline.

## 7. Evaluation protocol
- Chronological holdout: final 28 calendar days.
- Holdout period: **2026-09-03 to 2026-09-30**.
- Training rows: **24,720**.
- Test rows: **2,240**.
- Test series: **80** city-service combinations.
- Metrics: MAPE, MAE and RMSE.
- MAPE is calculated over non-zero actual booking counts. The supplied dataset's booking counts are positive, so all test observations are included.

This is a daily rolling one-step-ahead backtest: lag features for each test date use earlier observed bookings. It is not a recursive 7-day forecast simulation.

## 8. Actual baseline results

| Model | MAPE | MAE | RMSE |
|---|---:|---:|---:|
| Seasonal naive (lag 7) | 10.62% | 34.81 | 64.54 |
| Random Forest | 3.69% | 14.15 | 20.52 |

**Acceptance-target comparison:** the Random Forest holdout MAPE is 3.69%, compared with the project target of 12%. On this evaluation split, it meets the numerical MAPE target.

The Random Forest improves on the seasonal-naive benchmark on this holdout. These results are specific to this dataset and split; they are not a guarantee of future production performance.

## 9. Limitations and interpretation
- The evaluation uses the recorded weather values for the holdout dates. A real future forecast cannot know actual future weather; production use should supply weather forecasts or test a fallback that excludes future-observed weather.
- The reported test is rolling one-step-ahead because actual bookings from earlier days in the holdout become available as the forecast advances. A separate recursive multi-step evaluation should be added when developing the 7-day serving workflow.
- The dataset contains daily aggregates; the `hour` field does not provide intraday variation.
- The model is a baseline, not the requested final LSTM system.

## 10. Milestone 2 deliverables
- Data ingestion and preprocessing script
- Processed daily dataset
- EDA charts
- Lag/rolling/calendar/weather feature engineering
- Seasonal-naive benchmark
- Trained Random Forest baseline model
- Holdout predictions, metrics and feature-importance file
- README and this report

## 11. Next steps (later milestones)
Implement the PyTorch LSTM and compare it against this benchmark; then develop the Flask API, 7-day city-service forecast endpoint, weekly retraining automation and drift alert. Those are outside the scope of this Milestone 2 baseline package.
