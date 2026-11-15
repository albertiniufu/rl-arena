"""Rede actor-critic evidencial do EPPO.

O ator é categórico (mesma cabeça do PPO). O crítico não devolve um escalar:
devolve os quatro parâmetros da Normal-Inverse-Gamma (NIG) usada por
Akgül et al. (TMLR 2025) para modelar a incerteza do valor:

    (μ, σ²) | m ~ NIG(ω, ν, α, β)
    V | μ, σ²     ~ N(μ, σ²)

O encoder é compartilhado com o ator (como no PPO deste repo) para não
duplicar a CNN; o que muda em relação ao PPO é a cabeça do crítico e como
os momentos preditivos entram no GAE e na vantagem.
"""
import torch
from torch import nn

from ..encoders import build_encoder, encoder_output_size

# Limites dos log-parâmetros. O clamp evita `exp` estourando; os pisos
# mantêm ν, α−1 e β longe de zero, pois a variância da NIG explode quando
# α → 1 (o paper exige α > 1 para média finita).
_LOG_CLAMP = (-10.0, 10.0)
_NU_FLOOR = 1e-6
_ALPHA_MINUS_ONE_FLOOR = 1e-3
_BETA_FLOOR = 1e-6


class EvidentialActorCriticNetwork(nn.Module):
    """Encoder + ator categórico + crítico NIG.

    `forward` devolve `(logits, raw)`: `logits` de shape `(batch, n_actions)`
    e `raw` de shape `(batch, 4)` com ω e os log-parâmetros de ν, α−1 e β
    (use `evidential_parameters` para os valores positivos).
    """

    def __init__(self, input_shape, n_actions):
        super().__init__()
        input_shape = tuple(input_shape)
        self.features = build_encoder(input_shape)
        n = encoder_output_size(input_shape)
        self.actor = nn.Sequential(
            nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, n_actions))
        self.critic = nn.Sequential(
            nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, 4))

    def forward(self, x):
        feats = self.features(x)
        return self.actor(feats), self.critic(feats)


def evidential_parameters(raw):
    """Converte as 4 saídas cruas do crítico em `(ω, ν, α, β)`.

    ω fica livre (pode ser negativo); ν, α−1 e β saem de `exp` com piso,
    garantindo ν > 0, α > 1 e β > 0.
    """
    lo, hi = _LOG_CLAMP
    omega = raw[..., 0]
    nu = raw[..., 1].clamp(lo, hi).exp().clamp(min=_NU_FLOOR)
    alpha_minus_one = raw[..., 2].clamp(lo, hi).exp().clamp(
        min=_ALPHA_MINUS_ONE_FLOOR)
    beta = raw[..., 3].clamp(lo, hi).exp().clamp(min=_BETA_FLOOR)
    return omega, nu, alpha_minus_one + 1.0, beta


def evidential_moments(params):
    """Média e variância preditivas da NIG.

    `E[V] = ω`; `Var[V] = β/(α−1) · (1 + 1/ν)` — a primeira parcela é a
    incerteza aleatória e a segunda, a epistêmica (μ e σ² desconhecidos).
    """
    omega, nu, alpha, beta = params
    variance = beta / (alpha - 1.0) * (1.0 + 1.0 / nu)
    return omega, variance
