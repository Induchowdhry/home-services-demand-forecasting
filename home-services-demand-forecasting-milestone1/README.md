# Home Services Demand Forecasting

## Milestone 1 — Research + Architecture Design

### Project brief
Build an end-to-end demand forecasting pipeline for a home-services marketplace using 12 months of booking data, lag/rolling features, a PyTorch LSTM, a Flask API returning a 7-day city + service forecast, weekly retraining, and a simple drift alert.

### Important dataset validation note
The three files supplied for this submission were inspected before building the project:

- `aiml_training_data.csv`: 230 rows × 3 columns — `text`, `intent`, `confidence`
- `aiml_labelled_samples.json`: labelled Q&A samples across healthcare, agriculture, finance, NLP and ML topics
- `aiml_test_cases.txt`: 20 validation cases across multiple categories

The supplied CSV is **not booking/time-series data**: it contains text intent examples and has no city, service type, date/time, weather, or booking-demand target. Therefore, this Milestone 1 repository does **not** claim to train or validate a demand-forecasting LSTM, and it cannot honestly demonstrate the acceptance target of MAPE ≤ 12% until the correct booking dataset is provided.

This repository keeps the supplied files unchanged and provides the requested forecasting architecture and implementation skeleton so that the correct booking dataset can be plugged in later.

## Research reviewed

1. **Hochreiter & Schmidhuber (1997), “Long Short-Term Memory”**
   - Introduced LSTM units for learning long-term dependencies.
   - Relevance: demand forecasting contains temporal dependencies where recent and older observations can influence future demand.

2. **PyTorch documentation — `torch.nn.LSTM`**
   - Documents the LSTM module, input sequence shapes, hidden states, and configuration.
   - Relevance: the planned forecasting model will be implemented in PyTorch.

3. **Jason Brownlee / Machine Learning Mastery — LSTM for Time Series Forecasting**
   - Practical walkthrough of converting time-series observations into supervised sequences and evaluating forecasts.
   - Relevance: supports the sequence-building approach planned for the forecasting pipeline.

## Proposed architecture

```text
Booking + weather data
        |
        v
Data validation / cleaning
        |
        v
Time aggregation (city + service + day)
        |
        v
Calendar + weather features
        |
        v
Lag features + rolling statistics
        |
        v
Chronological train / validation / 4-week hold-out
        |
        v
Scaling fitted on training data only
        |
        v
Sliding-window sequences
        |
        v
PyTorch LSTM
        |
        v
Dense regression head
        |
        v
7-day demand forecast
        |
        +------> MAPE / MAE / RMSE
        |
        +------> Save model + scaler
        |
        v
Flask API: /forecast
        |
        +------> Drift monitoring
        |
        +------> Weekly retraining cron
```

## Planned features

### Time features
- day of week
- day of month
- week of year
- month
- weekend indicator
- hour/time slot if available

### Demand-history features
- lag 1
- lag 7
- lag 14
- lag 28
- 7-day rolling mean
- 14-day rolling mean
- 28-day rolling mean

### Context features
- city
- service type
- weather variables such as temperature, rainfall and weather condition

## Evaluation plan

The primary metric is **MAPE**. Additional metrics are **MAE** and **RMSE**.

The final evaluation must use a chronological **4-week hold-out**. No future observations may be used when creating training features.

Target acceptance criterion:

**MAPE ≤ 12% on the 4-week hold-out.**

Because the supplied CSV is not time-series booking data, this target has not been claimed as achieved in Milestone 1.

## API plan

Planned endpoint:

`GET /forecast?city=<city>&service_type=<service>`

Expected response:

```json
{
  "city": "Gurugram",
  "service_type": "Cleaning",
  "forecast": [
    {"date": "YYYY-MM-DD", "predicted_demand": 0.0}
  ]
}
```

The completed API will return 7 daily forecasts.

## Weekly retraining plan

A cron job will run `retraining/weekly_retrain.sh` once per week. It will:

1. Load the latest booking data.
2. Rebuild features.
3. Train the LSTM.
4. Evaluate the candidate model.
5. Save the model only if validation criteria are met.
6. Record the training timestamp and metrics.

## Drift monitoring plan

`monitoring/drift.py` provides the initial monitoring structure. The production version will compare incoming feature distributions against the training reference distribution and raise an alert when drift exceeds a configured threshold.

## Repository structure

```text
home-services-demand-forecasting/
├── api/
│   └── app.py
├── data/
│   ├── aiml_training_data.csv
│   ├── aiml_labelled_samples.json
│   └── aiml_test_cases.txt
├── models/
├── monitoring/
│   └── drift.py
├── notebooks/
│   └── 01_data_exploration.ipynb
├── retraining/
│   └── weekly_retrain.sh
├── src/
│   ├── data_pipeline.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

Python 3.14 may work for some packages, but the original project brief recommends Python 3.11 for compatibility.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Milestone 1 status

- [x] Project structure created
- [x] Supplied data files included
- [x] Data inspected
- [x] Research plan documented
- [x] Forecasting architecture designed
- [x] Evaluation metrics selected
- [x] API contract designed
- [x] Weekly retraining design documented
- [x] Drift monitoring skeleton added
- [ ] Correct booking/time-series dataset supplied
- [ ] LSTM trained on booking data
- [ ] MAPE ≤ 12% demonstrated
- [ ] 7-day API forecast demonstrated
- [ ] Weekly cron execution demonstrated

## Next milestone

Replace the current non-forecasting CSV with the correct booking dataset, then implement the feature engineering and LSTM pipeline without changing the overall architecture.
