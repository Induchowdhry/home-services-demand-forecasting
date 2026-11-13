import torch
from torch import nn

class DirectLSTM(nn.Module):
    """Direct 28-day multi-horizon LSTM forecaster."""
    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.0, horizon=28):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.head(self.dropout(hidden[-1]))
