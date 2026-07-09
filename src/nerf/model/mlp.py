import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_channels=6, out_channels=4):
        super().__init__()

        hidden_dim  = 256
        hidden_layers_num = 7

        self.linear1 = nn.Linear(in_channels, hidden_dim)
        self.relu1 = nn.ReLU()
        self.hidden_layers = [
            (
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(hidden_layers_num)
        ]

        self.last_linear = nn.Linear(hidden_dim, out_channels)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        for linear_layer, relu in self.hidden_layers:
            x = linear_layer(x)
            x = relu(x)
        out = self.last_linear(x)
        return out
