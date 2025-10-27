import torch
from torch import nn

class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=5, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)
