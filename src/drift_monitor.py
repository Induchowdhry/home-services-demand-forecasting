"""Simple booking-volume drift alert; compares recent 7-day mean to prior 28 days."""
import argparse, json
import pandas as pd

def check_drift(path, threshold=0.25):
    df=pd.read_csv(path,parse_dates=['booking_date'])
    end=df.booking_date.max(); recent_start=end-pd.Timedelta(days=6); reference_end=recent_start-pd.Timedelta(days=1); reference_start=reference_end-pd.Timedelta(days=27)
    recent=df[df.booking_date.between(recent_start,end)].groupby(['city','service_type']).bookings.mean()
    reference=df[df.booking_date.between(reference_start,reference_end)].groupby(['city','service_type']).bookings.mean()
    joined=pd.concat([reference.rename('reference_mean'),recent.rename('recent_mean')],axis=1).dropna()
    joined['relative_change']=(joined.recent_mean-joined.reference_mean)/joined.reference_mean.clip(lower=1)
    alerts=joined[joined.relative_change.abs()>threshold].reset_index()
    return {'status':'ALERT' if len(alerts) else 'OK','threshold_fraction':threshold,'window':{'reference_days':28,'recent_days':7},'series_checked':len(joined),'alert_count':len(alerts),'alerts':alerts.round(4).to_dict(orient='records')}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/processed_daily_bookings.csv');p.add_argument('--threshold',type=float,default=.25);a=p.parse_args()
    result=check_drift(a.data,a.threshold);print(json.dumps(result,indent=2));raise SystemExit(1 if result['status']=='ALERT' else 0)
