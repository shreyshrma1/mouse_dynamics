import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

class DynamicsClassifier(nn.Module):
    def __init__(self, input_size=39, hidden_one=64, hidden_two=32, dropout=0.5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_one),
            nn.BatchNorm1d(hidden_one),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_one, hidden_two),
            nn.BatchNorm1d(hidden_two),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_two, 1)
        )

    def forward(self, x):
        return self.backbone(x)  # raw logit, shape (B, 1); use BCEWithLogitsLoss during training

def preprocess_data(x_train, x_val):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_train_tensor = torch.FloatTensor(x_train)
    x_val_tensor = torch.FloatTensor(x_val)
    return scaler, x_train_tensor, x_val_tensor