import torch
from torch import nn

class DQN(nn.Module):
    """Classic CNN DQN: (batch, channels, height, width) -> action values.

    Com `dueling=True`, a cabeça separa o valor do estado (`v_head`, saída 1)
    da vantagem de cada ação (`a_head`, saída `n_actions`) e devolve
    `V + (A - média(A))`, centralizando as vantagens para remover a
    ambiguidade entre os dois ramos. Sem dueling, a cabeça única histórica
    (`head`) é mantida idêntica.
    """
    def __init__(self, input_shape, n_actions=12, dueling=False):
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
        self.dueling = dueling
        if dueling:
            self.head = None
            self.v_head = nn.Sequential(nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, 1))
            self.a_head = nn.Sequential(nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, n_actions))
        else:
            self.head = nn.Sequential(nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, n_actions))
            self.v_head = None
            self.a_head = None

    def forward(self, x):
        feats = self.features(x)
        if self.dueling:
            v = self.v_head(feats)
            a = self.a_head(feats)
            return v + a - a.mean(dim=1, keepdim=True)
        return self.head(feats)
