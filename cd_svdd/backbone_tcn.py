import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, bias=False
        )

    def forward(self, x):
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.relu  = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        return out + x


class TCNNetwork(nn.Module):
    def __init__(self, input_dim=12, channels=32, n_layers=4,
                 kernel_size=3, output_dim=8):
        super().__init__()

        self.input_proj = nn.Conv1d(input_dim, channels, kernel_size=1, bias=False)

        self.blocks = nn.Sequential(*[
            TCNBlock(channels, kernel_size, dilation=2**i)
            for i in range(n_layers)
        ])

        self.output_proj = nn.Linear(channels, output_dim, bias=False)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.blocks(x)
        x = x.mean(dim=2)
        return self.output_proj(x)