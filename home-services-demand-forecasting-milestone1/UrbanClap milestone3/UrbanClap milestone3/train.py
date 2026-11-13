import argparse, json, os
import numpy as np, pandas as pd, torch, mlflow
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data_pipeline import load_daily_data, make_training_windows
from src.model import DirectLSTM

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="data/processed_daily_bookings.csv")
    ap.add_argument("--epochs",type=int,default=12)
    ap.add_argument("--tracking-uri",default="file:./mlruns")
    args=ap.parse_args()
    torch.set_num_threads(1); torch.manual_seed(42); np.random.seed(42)
    df=load_daily_data(args.data)
    X,Y,groups,cities,services,wmean,wstd=make_training_windows(df)
    X=torch.tensor(X,dtype=torch.float32); Y=torch.tensor(Y,dtype=torch.float32)
    configs=[
      {"name":"LSTM_A","hidden_size":32,"num_layers":1,"dropout":0.0,"lr":0.001},
      {"name":"LSTM_B","hidden_size":48,"num_layers":2,"dropout":0.2,"lr":0.0005}
    ]
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("UrbanClap-Demand-Forecasting")
    results=[]
    for cfg in configs:
        model=DirectLSTM(X.shape[-1],cfg["hidden_size"],cfg["num_layers"],cfg["dropout"],28)
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg["lr"])
        loader=DataLoader(TensorDataset(X,Y),batch_size=512,shuffle=True)
        with mlflow.start_run(run_name=cfg["name"]):
            mlflow.log_params({**cfg,"epochs":args.epochs,"lookback":28,"horizon":28})
            for epoch in range(1,args.epochs+1):
                model.train(); losses=[]
                for xb,yb in loader:
                    optimizer.zero_grad(); loss=nn.MSELoss()(model(xb),yb)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    optimizer.step(); losses.append(loss.item())
                mlflow.log_metric("train_mse",float(np.mean(losses)),step=epoch)
            preds=[]; actual=[]
            model.eval()
            with torch.no_grad():
                for city,service,g,features,scale in groups:
                    pr=model(torch.tensor(features[309:337][None],dtype=torch.float32)).numpy().ravel()*scale
                    preds.extend(np.maximum(pr,0))
                    actual.extend(g.bookings.iloc[337:365].to_numpy(float))
            preds=np.array(preds); actual=np.array(actual)
            mape=float(np.mean(np.abs(actual-preds)/np.maximum(actual,1))*100)
            mae=float(mean_absolute_error(actual,preds))
            rmse=float(np.sqrt(mean_squared_error(actual,preds)))
            mlflow.log_metrics({"holdout_mape_percent":mape,"holdout_mae":mae,"holdout_rmse":rmse})
            results.append({"experiment":cfg["name"],"MAPE_percent":mape,"MAE":mae,"RMSE":rmse})
            os.makedirs("models",exist_ok=True)
            torch.save(model.state_dict(),f"models/{cfg['name']}.pth")
    pd.DataFrame(results).to_csv("experiment_results_reproduced.csv",index=False)
    print(pd.DataFrame(results).to_string(index=False))
if __name__=="__main__": main()
