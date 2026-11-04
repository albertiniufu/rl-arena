import torch
from torch import nn


class DuelingQNetwork(nn.Module):
    """Rede dueling: Q(s, a) = V(s) + A(s, a) − média_a A(s, a).

    A centralização das vantagens remove a ambiguidade entre o valor do
    estado e as vantagens; a saída continua tendo um Q por ação.
    """
    def __init__(self, input_shape, n_actions):
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
        self.v_head = nn.Sequential(
            nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, 1))
        self.a_head = nn.Sequential(
            nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, n_actions))

    def forward(self, x):
        feats = self.features(x)
        v = self.v_head(feats)
        a = self.a_head(feats)
        return v + a - a.mean(dim=1, keepdim=True)
