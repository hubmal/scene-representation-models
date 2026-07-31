import torch


class PCD:
    def __init__(self):
        self.coords = torch.rand(3) * 2 - 1
        self.scaling_vector = torch.rand(3) * 2 - 1
        self.quaternion = torch.rand(4) * 2 - 1
        self.color = torch.rand(3)
        self.opacity = torch.rand(1)