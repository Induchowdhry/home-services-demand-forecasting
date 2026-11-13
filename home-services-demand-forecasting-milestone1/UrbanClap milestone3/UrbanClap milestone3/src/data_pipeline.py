import numpy as np
import pandas as pd

def load_daily_data(path):
    df = pd.read_csv(path, parse_dates=["booking_date"])
    df = df.sort_values(["city", "service_type", "booking_date"]).reset_index(drop=True)
    df["dow_sin"] = np.sin(2*np.pi*df.booking_date.dt.dayofweek/7)
    df["dow_cos"] = np.cos(2*np.pi*df.booking_date.dt.dayofweek/7)
    df["doy_sin"] = np.sin(2*np.pi*df.booking_date.dt.dayofyear/365.25)
    df["doy_cos"] = np.cos(2*np.pi*df.booking_date.dt.dayofyear/365.25)
    return df

def make_training_windows(df, cutoff=337, lookback=28, horizon=28):
    cities = sorted(df.city.unique())
    services = sorted(df.service_type.unique())
    city_map = {v:i for i,v in enumerate(cities)}
    service_map = {v:i for i,v in enumerate(services)}
    train_rows = df.groupby(["city","service_type"], sort=False).head(cutoff)
    weather = ["temperature_c","rain_mm","humidity_pct"]
    means, stds = train_rows[weather].mean(), train_rows[weather].std().replace(0,1).fillna(1)
    cols = ["booking_scaled","dow_sin","dow_cos","doy_sin","doy_cos","is_weekend","is_holiday",
            "temperature_c","rain_mm","humidity_pct"]
    X, Y, groups = [], [], []
    for (city, service), g in df.groupby(["city","service_type"], sort=True):
        g = g.sort_values("booking_date").reset_index(drop=True).copy()
        scale = max(float(g.loc[:cutoff-1,"bookings"].mean()), 1.0)
        g["booking_scaled"] = g.bookings / scale
        for c in weather:
            g[c] = (g[c] - means[c]) / max(float(stds[c]), 1e-6)
        onehot = np.zeros(len(cities)+len(services), dtype=np.float32)
        onehot[city_map[city]] = 1
        onehot[len(cities)+service_map[service]] = 1
        features = np.concatenate([g[cols].to_numpy(np.float32),
                                   np.tile(onehot,(len(g),1))], axis=1)
        target = g.bookings.to_numpy(np.float32) / scale
        for start in range(0, cutoff-lookback-horizon+1, 3):
            X.append(features[start:start+lookback])
            Y.append(target[start+lookback:start+lookback+horizon])
        groups.append((city,service,g,features,scale))
    return np.stack(X), np.stack(Y), groups, cities, services, means, stds
