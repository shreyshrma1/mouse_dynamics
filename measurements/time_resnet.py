import torch
import torch.nn as nn
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=8, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm1d(out_channels)
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding="same"),
            nn.BatchNorm1d(out_channels)
        )
    def forward(self, x):
        initial_block = self.backbone(x)
        out = initial_block + self.shortcut(x)
        return torch.relu(out)

class ResNetTime(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.block1 = ResidualBlock(in_channels, 64)
        self.block2 = ResidualBlock(64, 128)
        self.block3 = ResidualBlock(128, 128)
        self.pooling = nn.AdaptiveAveragePool1d(1)
        self.dense = nn.Linear(128, 2)
    def forward(self, x):
        block1_out = self.block1(x)             # (B, 2, 128) -> (B, 64, 128)
        block2_out = self.block2(block1_out)    # (B, 64, 128) -> (B, 128, 128)
        block3_out = self.block3(block2_out)    # (B, 128, 128) -> (B, 128, 128)
        pool_out = self.pooling(block3_out)     # (B, 128, 128) -> (B, 128, 1)
        pool_out = pool_out.squeeze(-1)         # (B, 128, 1) -> (B, 128)
        out = self.dense(pool_out)              # (B, 128) -> (B, 2)
        return out
