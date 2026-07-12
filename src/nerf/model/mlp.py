import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_location_channels=3, in_direction_channels=3, out_color_channels=3):
        super().__init__()

        self.in_location_channels = in_location_channels
        hidden_dim  = 256
        hidden_layers_num = 8

        self.linear1 = nn.Linear(self.in_location_channels, hidden_dim)
        self.relu1 = nn.ReLU()
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
                )
                for _ in range(hidden_layers_num - 2)
            ]
        )
        self.last_linear = nn.Linear(hidden_dim, hidden_dim + 1)
        self.last_relu = nn.ReLU()
        self.additional_linear_layer = nn.Linear(hidden_dim + in_direction_channels, 128)
        self.additional_relu = nn.ReLU()
        self.output_linear_layer = nn.Linear(128, out_color_channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input):
        x, direction_vector = input[:, :self.in_location_channels], input[:, self.in_location_channels:]
        x = self.linear1(x)
        x = self.relu1(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.last_linear(x)
        x = self.last_relu(x)
        sigma, feature_vector = x[:, 0:1], x[:, 1:]
        x2 = self.additional_linear_layer(torch.cat([feature_vector, direction_vector], dim=1))
        x2 = self.additional_relu(x2)
        x2 = self.output_linear_layer(x2)
        colors = self.sigmoid(x2)
        out = torch.cat([colors, sigma], dim=1)
        return out

# SPRAWDZ CZY DOBRZE