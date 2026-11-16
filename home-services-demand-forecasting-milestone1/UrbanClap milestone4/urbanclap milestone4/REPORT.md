# Week 9 Milestone Report — Integration + Testing

**Project:** Urban Company (UrbanClap) demand forecasting  
**Role:** AI/ML Intern  
**Milestone:** Week 9 — Integration + Testing

## 1. Objective

Integrate the trained demand forecasting model into a Flask serving layer, expose a seven-day forecast for a requested city and service category, define API integration tests, provide a 50-concurrent-request load-test procedure, and add a basic drift-monitoring and weekly-retraining workflow.

## 2. Data used

The processed daily dataset contains 29,200 rows, 8 cities, 10 service categories and 80 city/service time series. It spans 2025-10-01 to 2026-09-30. The inspected processed file contains no null values or duplicate rows. The raw booking file and processed file both contain 29,200 rows.

## 3. Serving architecture

1. The training pipeline reads and validates the processed daily CSV.
2. A global PyTorch LSTM is trained across all city/service series using a 28-day historical window.
3. The trained weights and normalization metadata are saved as `artifacts/demand_lstm.pt`.
4. Flask loads the artifact and historical series, then exposes `/health`, `/cities` and `/forecast`.
5. `/forecast` validates the city/service pair and recursively generates seven daily booking estimates.
6. The drift monitor compares recent booking volume with a historical reference window. A weekly cron example reruns training.

## 4. Model evaluation

A chronological 28-day holdout was used for each of the 80 city/service series. Evaluation was recursive: predictions were fed forward during the 28-day horizon. The Week 9 training run used 24,720 training sequences and 2,240 holdout observations.

| Metric | Week 9 global LSTM result |
|---|---:|
| MAPE | 11.20% |
| MAE | 36.67 bookings |
| RMSE | 60.62 bookings |
| Holdout horizon | 28 days |
| Holdout observations | 2,240 |

The client's stated target is MAPE ≤12%. The measured Week 9 global LSTM holdout result is below that threshold. This result is distinct from the supplied Week 6 experiment CSV, where LSTM_A and LSTM_B reported MAPE values of approximately 15.58%.

## 5. API contract

**POST `/forecast`**

Request:

```json
{"city":"Bangalore","service_type":"AC Repair"}
```

Successful response contains the city, service type, a seven-day horizon, generation timestamp and seven date/value forecast entries. Missing fields return HTTP 400; unknown city/service values return HTTP 404.

## 6. Testing and performance evidence

Automated integration tests are included in `tests/test_api.py` for readiness, catalogue retrieval, seven-day response shape, missing parameters and unknown cities. The 50-worker load-test runner is included in `scripts/load_test.py`; it records latency percentiles, throughput, errors and request counts.

**Execution status in the preparation environment:** the LSTM training and chronological holdout evaluation were executed successfully. Flask was not installed in the preparation runtime and package installation was unavailable, so the Flask integration tests and live HTTP load test could not be executed here. Consequently, no latency or throughput values are claimed in this report. Run the commands in the README in an environment with `requirements.txt` installed, then attach the generated `reports/load_test_results.json` and pytest output before marking those checks as executed.

## 7. Monitoring and retraining

The drift script compares the latest seven-day mean booking count with the preceding 28-day mean for each city/service pair. It raises an alert when the relative change exceeds 25%. The cron example schedules retraining every Monday at 02:00. For production, retraining should include validation gates, artifact versioning, rollback and service reload after successful validation.

## 8. Limitations and next steps

The historical dataset includes weather observations, but it does not provide future weather forecasts. The deployed LSTM therefore uses booking history and calendar features rather than assuming future weather is known. A production extension should connect a weather forecast source. The current implementation is a demonstration and does not include authentication, rate limiting, cloud deployment or a production database.

## 9. Deliverables

- Flask API source: `app.py`
- LSTM training and inference code: `src/`
- Saved trained model: `artifacts/demand_lstm.pt`
- API integration tests: `tests/test_api.py`
- 50-concurrency load test: `scripts/load_test.py`
- Drift monitoring: `src/drift_monitor.py`
- Weekly retraining example: `scripts/weekly_retrain.*`
- Setup and execution instructions: `README.md`
