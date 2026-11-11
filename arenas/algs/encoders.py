"""Encoders compartilhados pelos algoritmos.

Aceita dois formatos de observação:

- `(C, H, W)` — grade semântica do seek_coin; usa a CNN histórica (mesmos
  nomes/ordem de camadas, para manter compatibilidade com checkpoints).
- `(N,)` — vetor de features (risk/economy); usa um MLP.
"""
import torch
from torch import nn


def build_encoder(input_shape):
    """Sequência de features que mapeia a observação para um vetor latente."""
    if len(input_shape) == 1:
        n = int(input_shape[0])
        return nn.Sequential(
            nn.Linear(n, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
    if len(input_shape) == 3:
        c, h, w = input_shape
        return nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
    raise ValueError(
        f"observação deve ser (C, H, W) ou (N,); recebido {input_shape}")


def encoder_output_size(input_shape) -> int:
    """Número de features na saída do encoder (inferido com um forward)."""
    encoder = build_encoder(input_shape)
    with torch.no_grad():
        sample = torch.zeros(1, *input_shape)
        return int(encoder(sample).shape[1])
