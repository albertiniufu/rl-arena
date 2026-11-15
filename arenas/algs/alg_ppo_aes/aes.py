"""Agendador AES — Adaptive Entropy Scheduling.

Wang, Xia, Chen e Liu, *Tracking Drift: Variation-Aware Entropy Scheduling
for Non-Stationary Reinforcement Learning* (ICML 2026), arXiv:2601.19624.

O agendador substitui o coeficiente de entropia fixo por um coeficiente
calculado online a partir de um proxy observável do drift: o quantil 0,9 do
erro de TD absoluto do lote. O proxy passa por uma EMA e é acumulado; a
regra de escala é a raiz quadrada da média acumulada,

    λ_t = κ · sqrt(Â_t / t),   Â_t = Σ_{s≤t} EMA(|δ_s|_0.9)

recortada em `[lambda_min, lambda_max]`. No PPO o coeficiente entra no
lugar de `c_ent` na perda: `loss = pg_loss - λ_t·H(π) + vf·v_loss`.

Este é o mesmo contrato da implementação oficial (`cleanrl`): a atualização
acontece a cada minibatch do PPO, com resíduos de TD de 1 passo.
"""
import torch


class AESScheduler:
    """Coeficiente de entropia adaptativo com proxy de drift em quantil.

    Estado: `ema` (None até o primeiro update), `accumulated`, `step` e
    `value`. Não há reset explícito; um agendador novo recomeça do zero.
    """

    def __init__(self, q=0.9, beta=0.95, kappa=1.0,
                 lambda_min=1e-4, lambda_max=0.1):
        if not (0.0 <= q <= 1.0):
            raise ValueError("AES: q deve estar em [0, 1]")
        if not (0.0 <= beta < 1.0):
            raise ValueError("AES: beta deve estar em [0, 1)")
        if not (kappa > 0.0 and lambda_min > 0.0
                and lambda_min <= lambda_max):
            raise ValueError("AES: requer kappa > 0 e 0 < min <= max")
        self.q = float(q)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.ema = None
        self.accumulated = 0.0
        self.step = 0
        self.value = self.lambda_min

    def update(self, residuals):
        """Consome os resíduos de TD do minibatch e devolve o novo λ."""
        with torch.no_grad():
            residuals = torch.as_tensor(residuals, dtype=torch.float32)
            if residuals.numel() == 0:
                raise ValueError("AES requer resíduos de TD não vazios")
            if not torch.isfinite(residuals).all():
                raise ValueError("AES requer resíduos de TD finitos")
            raw = torch.quantile(
                residuals.abs().flatten(), self.q).item()
        if self.ema is None:
            self.ema = raw
        else:
            self.ema = self.beta * self.ema + (1.0 - self.beta) * raw
        self.step += 1
        self.accumulated += self.ema
        scheduled = self.kappa * (max(self.accumulated, 0.0)
                                  / self.step) ** 0.5
        self.value = float(min(max(scheduled, self.lambda_min),
                               self.lambda_max))
        return self.value

    def state_dict(self):
        return {
            "q": self.q, "beta": self.beta, "kappa": self.kappa,
            "lambda_min": self.lambda_min, "lambda_max": self.lambda_max,
            "ema": self.ema, "accumulated": self.accumulated,
            "step": self.step, "value": self.value,
        }

    def load_state_dict(self, state):
        for key in ("q", "beta", "kappa", "lambda_min", "lambda_max"):
            if state[key] != getattr(self, key):
                raise ValueError("AES: configuração do checkpoint difere")
        self.ema = state["ema"]
        self.accumulated = state["accumulated"]
        self.step = state["step"]
        self.value = state["value"]

    def scalars(self):
        """Diagnósticos para log (estilo TensorBoard da implementação oficial)."""
        return {
            "aes/raw_proxy_ema": 0.0 if self.ema is None else float(self.ema),
            "aes/accumulated_proxy": float(self.accumulated),
            "aes/lambda": float(self.value),
        }
