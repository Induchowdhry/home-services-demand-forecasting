"""Train and evaluate a global LSTM; saves model artifact and honest holdout metrics."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import DemandLSTM, LOOKBACK, calendar_features

ROOT = Path(__file__).resolve().parents[1]

def load_data(path):
    df = pd.read_csv(path, parse_dates=['booking_date'])
    required = {'booking_date','city','service_type','bookings'}
    missing = required - set(df.columns)
    if missing: raise ValueError(f'Missing required columns: {sorted(missing)}')
    if df[list(required)].isna().any().any(): raise ValueError('Missing values in required columns')
    return df.sort_values(['city','service_type','booking_date'])

def make_examples(df, holdout_days=28):
    series = []
    for (city, service), g in df.groupby(['city','service_type']):
        g = g.sort_values('booking_date')
        values = g.bookings.to_numpy(dtype=np.float32)
        dates = g.booking_date.to_list()
        if len(values) <= LOOKBACK + holdout_days: continue
        series.append((city, service, values, dates))
    if not series: raise ValueError('Not enough history to create sequences')
    train_logs = np.concatenate([np.log1p(v[:-holdout_days]) for _,_,v,_ in series])
    mean, std = float(train_logs.mean()), float(train_logs.std() + 1e-8)
    X, y = [], []
    for _,_,values,dates in series:
        train_end = len(values) - holdout_days
        for t in range(LOOKBACK, train_end):
            hist = (np.log1p(values[t-LOOKBACK:t]) - mean) / std
            cal = np.stack([calendar_features(dates[j]) for j in range(t-LOOKBACK,t)])
            X.append(np.column_stack([hist, cal]))
            y.append((np.log1p(values[t]) - mean) / std)
    return series, np.asarray(X,dtype=np.float32), np.asarray(y,dtype=np.float32), mean, std

def recursive_forecast(model, history, start_date, mean, std, days, device):
    import pandas as pd
    hist = list(map(float, history))
    out=[]
    for i in range(days):
        d = pd.Timestamp(start_date) + pd.Timedelta(days=i)
        seq=[]
        for j in range(LOOKBACK):
            idx = len(hist)-LOOKBACK+j
            val = hist[idx] if idx >= 0 else hist[0]
            prev_date = d - pd.Timedelta(days=LOOKBACK-j)
            z=(np.log1p(max(val,0))-mean)/(std+1e-8)
            seq.append([z,*calendar_features(prev_date)])
        with torch.no_grad():
            pred=float(model(torch.tensor([seq],dtype=torch.float32,device=device)).cpu().item())
        value=max(0.0,float(np.expm1(pred*std+mean)))
        out.append(value); hist.append(value)
    return np.asarray(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default=str(ROOT/'data/processed_daily_bookings.csv'))
    ap.add_argument('--epochs',type=int,default=12)
    ap.add_argument('--hidden-size',type=int,default=48)
    ap.add_argument('--layers',type=int,default=2)
    ap.add_argument('--dropout',type=float,default=.2)
    ap.add_argument('--learning-rate',type=float,default=.0005)
    args=ap.parse_args()
    torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(1)
    df=load_data(args.data)
    series,X,y,mean,std=make_examples(df)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=DemandLSTM(hidden_size=args.hidden_size,num_layers=args.layers,dropout=args.dropout).to(device)
    loader=DataLoader(TensorDataset(torch.tensor(X),torch.tensor(y).reshape(-1,1)),batch_size=256,shuffle=True)
    opt=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    loss_fn=torch.nn.MSELoss()
    history=[]
    for epoch in range(args.epochs):
        model.train(); losses=[]
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device)
            opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            losses.append(float(loss.item()))
        history.append(float(np.mean(losses)))
        print(f'Epoch {epoch+1}/{args.epochs} - train_mse={history[-1]:.6f}')
    model.eval(); actual=[]; predicted=[]
    for city,service,values,dates in series:
        split=len(values)-28
        preds=recursive_forecast(model,values[:split],dates[split],mean,std,28,device)
        actual.extend(values[split:].tolist()); predicted.extend(preds.tolist())
    actual=np.asarray(actual); predicted=np.asarray(predicted)
    mape=float(np.mean(np.abs((actual-predicted)/np.maximum(actual,1e-8)))*100)
    mae=float(np.mean(np.abs(actual-predicted))); rmse=float(np.sqrt(np.mean((actual-predicted)**2)))
    artifact={'state_dict':model.cpu().state_dict(),'hidden_size':args.hidden_size,'num_layers':args.layers,'dropout':args.dropout,'mean':mean,'std':std,'lookback':LOOKBACK,'cities':sorted(df.city.unique().tolist()),'services':sorted(df.service_type.unique().tolist())}
    ROOT.joinpath('artifacts').mkdir(exist_ok=True)
    torch.save(artifact,ROOT/'artifacts/demand_lstm.pt')
    metrics={'model':'Global LSTM','holdout_days':28,'holdout_rows':len(actual),'city_service_series':len(series),'MAPE_percent':mape,'MAE':mae,'RMSE':rmse,'epochs':args.epochs,'training_examples':len(X),'device':device,'training_loss_by_epoch':history,'acceptance_target_MAPE_percent':12.0,'acceptance_target_met':bool(mape<=12.0)}
    (ROOT/'reports/training_metrics.json').write_text(json.dumps(metrics,indent=2))
    pd.DataFrame([metrics]).drop(columns=['training_loss_by_epoch']).to_csv(ROOT/'reports/week9_holdout_metrics.csv',index=False)
    print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
