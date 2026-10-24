import torch
from torch import nn

class DQN(nn.Module):
    """Classic CNN DQN: (batch, channels, height, width) -> 12 action values."""
    def __init__(self, input_shape, n_actions=12):
        super().__init__()
        c, h, w = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.features(torch.zeros(1, c, h, w)).shape[1]
        self.head = nn.Sequential(nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, n_actions))

    def forward(self, x):
        return self.head(self.features(x))
