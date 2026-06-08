import torch
import torch.nn as nn

class CDSVDDNetwork(nn.Module):
    def __init__(self, input_dim=48, hidden_dim=32, output_dim=8):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim, bias=False)
        )

    def forward(self, x):
        return self.backbone(x)