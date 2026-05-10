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
    def __init__(self, in_channels=2, seq_len=128):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.Sequential(
            ResidualBlock(in_channels, 64),
            ResidualBlock(64, 128),
            ResidualBlock(128, 128)
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.expand = nn.Linear(128, 128 * seq_len)
        self.decoder = nn.Sequential(
            ResidualBlock(128, 128),
            ResidualBlock(128, 64),
            ResidualBlock(64, in_channels),
        )
        
    def encode(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)          # (B, 128, seq_len)
        x = self.pooling(x).squeeze(-1) # (B, 128)
        return x
    
    def decode(self, z):
        x = self.expand(z)                          # (B, 128 * seq_len)
        x = x.view(-1, 128, self.seq_len)           # (B, 128, seq_len)
        x = self.decoder(x)                         # (B, in_channels, seq_len)
        x = x.permute(0, 2, 1)                      # (B, seq_len, in_channels)
        return x
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
    
    def reconstruction_error(self, x):
        x_recon = self.forward(x)
        error = ((x - x_recon) ** 2).mean(dim=[1, 2])
        return error
