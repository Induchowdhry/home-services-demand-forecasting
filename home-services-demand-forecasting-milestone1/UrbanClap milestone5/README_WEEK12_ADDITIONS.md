# Week 12 final submission additions

## Demo video
Record the five-minute walkthrough using `demos/DEMO_RECORDING_SCRIPT.txt`.
Place the actual MP4 in `demos/` and replace this line with its accessible URL:
- Demo URL: [ADD YOUR VIDEO LINK AFTER RECORDING]

## Continuous integration
A starter GitHub Actions workflow is provided at `.github/workflows/ci.yml`.
After pushing, open the repository's **Actions** tab and verify the latest `main`
workflow run is green. Adjust the Python version if your project uses another version.

## City/service error audit
The script `scripts/segment_error_audit.py` expects
`reports/holdout_predictions.csv` with columns:
`city, service_type, actual, predicted`.

Export actual and predicted values from the same holdout period, then run:
```powershell
python scripts/segment_error_audit.py
```
It creates:
- `reports/error_by_city.csv`
- `reports/error_by_service.csv`
- `reports/error_by_city_service.csv`

Do not invent results. Review sample counts and error differences before writing
the audit conclusion. This is a segment performance audit, not a demographic
fairness assessment.

## Existing reported model metrics
- Holdout: 28 days / 2,240 observations
- MAPE: 11.20%
- MAE: 36.67
- RMSE: 60.62
- MAPE target: <= 12% (reported as achieved)

## Deployment note
The repository documents a local/demo Flask API unless you have separately deployed
and verified a public service. Do not claim a public production URL without testing it.
Keep secrets out of Git. Use a production WSGI server and HTTPS for a real deployment.
