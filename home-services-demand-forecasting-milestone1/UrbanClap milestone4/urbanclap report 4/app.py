"""Flask REST API: GET /health, GET /cities, POST /forecast."""
from pathlib import Path
from threading import Lock
import pandas as pd
import torch
from flask import Flask, jsonify, request
from src.model import DemandLSTM, LOOKBACK, calendar_features

ROOT=Path(__file__).resolve().parent
DATA_PATH=ROOT/'data/processed_daily_bookings.csv'
ARTIFACT_PATH=ROOT/'artifacts/demand_lstm.pt'
app=Flask(__name__)
_model=None; _artifact=None; _data=None
_runtime_lock = Lock()

def load_runtime():
    global _model,_artifact,_data
    # Guard first-load initialization so simultaneous requests do not race to load the artifact/data.
    with _runtime_lock:
        if _model is None:
            if not ARTIFACT_PATH.exists(): raise FileNotFoundError('Model artifact missing. Run python src/train.py first.')
            _artifact=torch.load(ARTIFACT_PATH,map_location='cpu',weights_only=False)
            _model=DemandLSTM(hidden_size=_artifact['hidden_size'],num_layers=_artifact['num_layers'],dropout=_artifact['dropout'])
            _model.load_state_dict(_artifact['state_dict']); _model.eval()
        if _data is None:
            _data=pd.read_csv(DATA_PATH,parse_dates=['booking_date'])
        return _model,_artifact,_data

def predict_series(model,artifact,history,start_date,days=7):
    hist=list(map(float,history)); rows=[]
    for i in range(days):
        date=pd.Timestamp(start_date)+pd.Timedelta(days=i)
        seq=[]
        for j in range(LOOKBACK):
            ix=len(hist)-LOOKBACK+j
            val=hist[ix] if ix>=0 else hist[0]
            prev=date-pd.Timedelta(days=LOOKBACK-j)
            z=(__import__('numpy').log1p(max(val,0))-artifact['mean'])/(artifact['std']+1e-8)
            seq.append([z,*calendar_features(prev)])
        with torch.no_grad():
            scaled=float(model(torch.tensor([seq],dtype=torch.float32)).item())
        pred=max(0.0,float(__import__('numpy').expm1(scaled*artifact['std']+artifact['mean'])))
        rows.append({'date':date.strftime('%Y-%m-%d'),'predicted_bookings':round(pred,2)})
        hist.append(pred)
    return rows

@app.get('/health')
def health():
    try:
        load_runtime(); return jsonify({'status':'ok','model_loaded':True}),200
    except Exception as exc: return jsonify({'status':'error','message':str(exc)}),503

@app.get('/cities')
def cities():
    try:
        _,artifact,_=load_runtime()
        return jsonify({'cities':artifact['cities'],'services':artifact['services']}),200
    except Exception as exc: return jsonify({'error':str(exc)}),503

@app.post('/forecast')
def forecast():
    payload=request.get_json(silent=True) or {}
    city=str(payload.get('city','')).strip()
    service=str(payload.get('service_type','')).strip()
    if not city or not service: return jsonify({'error':'city and service_type are required'}),400
    try:
        model,artifact,data=load_runtime()
        if city not in artifact['cities']: return jsonify({'error':f'Unknown city: {city}'}),404
        if service not in artifact['services']: return jsonify({'error':f'Unknown service_type: {service}'}),404
        group=data[(data.city==city)&(data.service_type==service)].sort_values('booking_date')
        if group.empty: return jsonify({'error':'No history available for this city/service pair'}),404
        last_date=group.booking_date.iloc[-1]
        result=predict_series(model,artifact,group.bookings.tail(LOOKBACK).tolist(),last_date+pd.Timedelta(days=1),7)
        return jsonify({'city':city,'service_type':service,'forecast_horizon_days':7,'generated_at_utc':pd.Timestamp.utcnow().isoformat(),'forecast':result}),200
    except FileNotFoundError as exc: return jsonify({'error':str(exc)}),503
    except Exception as exc: return jsonify({'error':f'Forecast failed: {exc}'}),500

if __name__=='__main__': app.run(host='0.0.0.0',port=5000,debug=False,threaded=True)
