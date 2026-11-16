# Urban Company (UrbanClap) â€” Week 9: Integration + Testing

## What is included

- `app.py` â€” Flask REST API with health, catalogue, and 7-day forecast endpoints.
- `src/model.py` â€” global PyTorch LSTM architecture and sequence feature preparation.
- `src/train.py` â€” reproducible training and chronological 28-day holdout evaluation; writes `artifacts/demand_lstm.pt` and metrics.
- `src/drift_monitor.py` â€” simple booking-volume drift alert (recent 7-day mean vs previous 28-day mean).
- `tests/test_api.py` â€” API integration tests.
- `scripts/load_test.py` â€” concurrent HTTP load test (default 50 workers, 200 requests); writes measured JSON and Markdown results.
- `scripts/run_week9_validation.py` â€” one-command runner that starts Flask, runs integration tests, then runs the 50-worker load test.
- `scripts/weekly_retrain.sh` and `scripts/weekly_retrain.cron` â€” weekly retraining schedule example.
- `data/` â€” supplied raw, processed and experiment CSVs.
- `reports/` â€” measured holdout results and project report.

## Dataset

The processed file contains 29,200 daily observations from 2025-10-01 through 2026-09-30, covering 8 cities, 10 service categories and 80 city/service series. The inspected processed dataset has no missing values or duplicate rows.

## Model and evaluation

A global LSTM is trained across the 80 city/service series. It uses a 28-day input window of log-transformed booking counts and weekday sine/cosine calendar signals. The 28-day holdout is chronological and evaluated recursively (each prediction is fed into the next forecast step), which is closer to a multi-step deployment setting than one-step teacher-forced evaluation.

The LSTM uses booking history and calendar features. Historical weather fields are present in the source dataset, but future weather observations are not supplied at inference time. To avoid using unavailable future information, weather is not included in the deployed model. A production extension can connect a weather forecast provider and retrain with forecast-available weather features.

The supplied Week 6 `experiment_results.csv` records MAPE values of approximately 15.58% for both earlier configurations. The Week 9 retrained global model in `reports/training_metrics.json` is a separate run and reports its own 28-day holdout result. Do not describe the Week 6 result as meeting the 12% target.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train the model

```bash
python src/train.py --epochs 4
```

This creates `artifacts/demand_lstm.pt`, `reports/training_metrics.json`, and `reports/week9_holdout_metrics.csv`.

## Start the Flask API

```bash
python app.py
```

The API listens on `http://127.0.0.1:5000`.

### Endpoints

- `GET /health` â€” model readiness.
- `GET /cities` â€” supported city and service values.
- `POST /forecast` â€” seven daily forecasts.

Example request:

```bash
curl -X POST http://127.0.0.1:5000/forecast \\
  -H "Content-Type: application/json" \\
  -d '{"city":"Bangalore","service_type":"AC Repair"}'
```

Expected response structure:

```json
{
  "city": "Bangalore",
  "service_type": "AC Repair",
  "forecast_horizon_days": 7,
  "generated_at_utc": "...",
  "forecast": [
    {"date": "YYYY-MM-DD", "predicted_bookings": 0.0}
  ]
}
```

## Integration tests

With the API dependencies installed and the model artifact generated:

```bash
pytest -q
```

The tests cover health, city/service catalogue, valid 7-day forecast, missing request fields, and unknown cities.

## Run all Week 9 validation (recommended)

After installing dependencies and ensuring the trained artifact exists, run this single command from the project root:

```bash
python scripts/run_week9_validation.py
```

The runner starts the Flask API, waits for `/health`, executes the integration suite with `pytest`, then sends **200 HTTP forecast requests using 50 concurrent worker threads**. It shuts the API down after the run. A successful run creates:

- `reports/load_test_results.json` â€” machine-readable measurements and environment details.
- `reports/load_test_results.md` â€” ready-to-attach measured latency and throughput summary.

The load-test report includes successful/failed requests, error rate, total elapsed time, overall and successful requests/second, and mean, p50, p95, p99, minimum and maximum latency. These figures are measured on the machine where the command is run and must not be replaced with estimates.

### Observed load-test results — 16 November 2026

The Week 9 validation was executed successfully with the following measured results:

| Metric | Observed result |
|---|---:|
| Concurrent workers | 50 |
| Total requests | 200 |
| Successful requests | 200 |
| Failed requests | 0 |
| Error rate | 0.0% |
| Total elapsed time | 6.646 seconds |
| Throughput | 30.09 requests/second |
| Mean latency | 1553.05 ms |
| P50 latency | 1648.24 ms |
| P95 latency | 1947.23 ms |
| P99 latency | 2062.23 ms |
| Minimum latency | 579.48 ms |
| Maximum latency | 2081.12 ms |

All 5 API integration tests passed, with 1 deprecation warning. The load test used the `/forecast` endpoint with Bangalore and AC Repair as the request input.

Detailed measurements are available in `reports/load_test_results.md` and `reports/load_test_results.json`. Results depend on the hardware and runtime environment.
To run only the load test, start `python app.py` in one terminal and run `python scripts/load_test.py --concurrency 50 --requests 200` in another.

## Drift check

```bash
python src/drift_monitor.py --data data/processed_daily_bookings.csv --threshold 0.25
```

The script checks whether the recent 7-day average differs by more than 25% from the preceding 28-day average for a city/service series. An alert is a signal for investigation, not proof of model degradation.

## Weekly automated retraining

Edit `scripts/weekly_retrain.cron` so its project path and virtual-environment Python path match the deployment machine, then install it with:

```bash
crontab scripts/weekly_retrain.cron
```

The example schedule runs every Monday at 02:00 server time. In production, add model validation, artifact versioning, rollback, logging, and a service restart/reload after a successful training run.

## Deployment notes / limitations

- This is a project demonstration, not a production-hardened service.
- The included dataset ends on 2026-09-30. Forecast dates therefore begin on 2026-10-01 when using the supplied data.
- No live weather feed, external database, authentication, rate limiting, or cloud deployment is configured.
- Load-test figures must be generated on the target machine because they depend on hardware and runtime conditions. Do not claim the load test is complete until `reports/load_test_results.md` and `.json` have been generated by a successful run.
- The Flask API and integration tests require the packages in `requirements.txt`.

