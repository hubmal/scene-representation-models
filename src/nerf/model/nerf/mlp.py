import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_location_channels=60, in_direction_channels=24, out_color_channels=3):
        super().__init__()

        self.in_location_channels = in_location_channels
        hidden_dim  = 256

        self.linear1 = nn.Linear(self.in_location_channels, hidden_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.relu4 = nn.ReLU()
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.relu5 = nn.ReLU()
        self.linear6 = nn.Linear(self.in_location_channels + hidden_dim, hidden_dim)
        self.relu6 = nn.ReLU()
        self.linear7 = nn.Linear(hidden_dim, hidden_dim)
        self.relu7 = nn.ReLU()
        self.linear8 = nn.Linear(hidden_dim, hidden_dim)
        self.relu8 = nn.ReLU()
        self.linear9 = nn.Linear(hidden_dim, hidden_dim + 1)
        self.relu9 = nn.ReLU()
        self.linear10 = nn.Linear(hidden_dim + in_direction_channels, 128)
        self.relu10 = nn.ReLU()
        self.linear11 = nn.Linear(128, out_color_channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input):
        point, direction_vector = input[:, :self.in_location_channels], input[:, self.in_location_channels:]
        x = self.linear1(point)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.relu4(x)
        x = self.linear5(x)
        x = self.relu5(x)
        x = self.linear6(torch.cat([point, x], dim=1))
        x = self.relu6(x)
        x = self.linear7(x)
        x = self.relu7(x)
        x = self.linear8(x)
        x = self.relu8(x)
        x = self.linear9(x)
        sigma, feature_vector = x[:, 0:1], x[:, 1:]
        if self.training:
            sigma = sigma + torch.randn_like(sigma)
        sigma = self.relu9(sigma)
        x2 = self.linear10(torch.cat([feature_vector, direction_vector], dim=1))
        x2 = self.relu10(x2)
        x2 = self.linear11(x2)
        colors = self.sigmoid(x2)
        out = torch.cat([colors, sigma], dim=1)
        return out