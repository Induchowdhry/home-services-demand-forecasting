"""Global LSTM demand forecaster for city/service daily booking series."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn

LOOKBACK = 28

class DemandLSTM(nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 48, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_size, 24), nn.ReLU(), nn.Linear(24, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

def calendar_features(date_value):
    dow = int(date_value.weekday())
    angle = 2.0 * np.pi * dow / 7.0
    return np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)

def make_input(history, next_date, mean, std):
    """Create a 28x3 sequence from booking history and calendar signals."""
    hist = np.asarray(history, dtype=np.float32)[-LOOKBACK:]
    if len(hist) < LOOKBACK:
        hist = np.pad(hist, (LOOKBACK-len(hist), 0), mode='edge')
    log_values = (np.log1p(np.maximum(hist, 0)) - mean) / (std + 1e-8)
    # Calendar signals correspond to the historical sequence's weekdays.
    import pandas as pd
    end = pd.Timestamp(next_date) - pd.Timedelta(days=1)
    dates = [end - pd.Timedelta(days=i) for i in range(LOOKBACK-1, -1, -1)]
    cal = np.stack([calendar_features(d) for d in dates])
    return np.column_stack([log_values, cal]).astype(np.float32)
