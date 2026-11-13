# Urban Company Demand Forecasting — Week 6

## Objective
Train and compare two PyTorch LSTM configurations for daily booking demand across city-service combinations. The dataset contains 29,200 daily records, 8 cities and 10 service types over 2025-10-01 to 2026-09-30.

## Week 6 experiment design
- Lookback: 28 days
- Direct forecast horizon: 28 days
- Hold-out: final 28 days (2026-09-03 to 2026-09-30)
- Training windows: 7,520 (stride = 3 days)
- Evaluation: MAPE, MAE and RMSE
- Weather inputs: historical temperature, rain and humidity are standardized using training-period statistics. Future weather is not supplied to the decoder; this avoids using unknown future weather values.

## Results
| Model | Holdout MAPE |
|---|---:|
| Seasonal-naive baseline | 11.25% |
| LSTM_A (32 hidden, 1 layer, lr 0.001) | 15.58% |
| LSTM_B (48 hidden, 2 layers, dropout 0.2, lr 0.0005) | 15.58% |
| 80% seasonal-naive + 20% LSTM_A ensemble | 10.87% |

The ensemble is below the 12% MAPE target on this hold-out. The standalone LSTM configurations are above 12%; therefore the report does **not** claim that the LSTM alone meets the target. The blend was evaluated as an additional experiment and is included transparently.

## Reproduce
```bash
python -m venv .venv
# activate the environment
pip install -r requirements.txt
python train.py --data data/processed_daily_bookings.csv --epochs 12
```
MLflow tracking is configured to write local runs to `./mlruns`. Open the UI with:
```bash
mlflow ui --backend-store-uri ./mlruns
```

## Project structure
- `data/`: supplied raw and processed CSV data
- `src/model.py`: LSTM model definition
- `src/data_pipeline.py`: feature engineering and training-window builder
- `train.py`: two-configuration training and MLflow logging
- `models/best_model.pth`: selected LSTM checkpoint
- `experiment_results.csv`: measured comparison of the two LSTM experiments
- `experiment_logs.csv`: epoch-level training loss
- `benchmark_comparison.csv`: baseline, LSTM and ensemble metrics
- `blend_experiment.csv`: ensemble-weight sensitivity experiment
- `predictions.csv`: hold-out predictions by city and service
- `reports/week6_experiment_report.pdf`: report for milestone submission

## Limitations and next steps
1. Standalone LSTM MAPE is 15.58%, so improve it before claiming the LSTM meets the target.
2. Test rolling-origin validation across multiple cutoffs rather than relying on one hold-out.
3. Add explicit future-weather forecasts or scenario-based weather inputs when connecting the production API.
4. Add MLflow artifact logging for the final model and prediction files in the deployment environment.
