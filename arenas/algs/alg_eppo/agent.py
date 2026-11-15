"""EPPO — Evidential Proximal Policy Optimization.

Akgül, Baykal, Haußmann e Kandemir, *Overcoming Non-stationary Dynamics
with Evidential Proximal Policy Optimization* (TMLR 2025),
arXiv:2503.01468.

Adaptação para o seek_coin (ações discretas e uma CNN pequena):

* o crítico modela `V(s)` com uma Normal-Inverse-Gamma e devolve
  `(ω, ν, α, β)` em vez de um escalar; é treinado por máxima
  verossimilhança do tipo II com as hiperpriors do paper;
* o GAE usa `E[V] = ω` para a média e propaga a variância de `V` na mesma
  recursão exponencialmente ponderada ("EPPO_cor" ou "EPPO_ind");
* a política otimiza a vantagem otimista `A_UCB = E[A] + κ·σ[A]` no
  surrogate recortado do PPO — a incerteza do crítico vira exploração
  dirigida.

O paper usa ator e crítico MLPs separados para controle contínuo; aqui o
encoder é compartilhado com as duas cabeças e o ator continua categórico,
como no restante do repo.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Gamma, Normal

from .model import (
    EvidentialActorCriticNetwork,
    evidential_moments,
    evidential_parameters,
)
from .rollout_buffer import RolloutBuffer


def compute_gae_with_uncertainty(rewards, values, next_values, values_var,
                                 next_values_var, terminated, episode_ends,
                                 gamma, gae_lambda, variant="cor"):
    """GAE(λ) com média e variância evidenciais.

    A média é o GAE usual, com `E[V] = ω` no lugar de V. A variância segue
    o paper, com `c_next = ((1−λ)/λ)²` e `c_acc = (γλ)²`:

    * `cor` (erros correlacionados):
      `Var[A_t] = Var[V_t] + c_next · C_t`
    * `ind` (estimadores k-passos independentes):
      `Var[A_t] = c_var · Var[V_t] + c_next · C_t`, com
      `c_var = (1−λ)/(1+λ)`

    e `C_t = c_acc · (C_{t+1}·propaga_t + Var[V_{t+1}])`. Devolve
    `(vantagens_médias, variâncias, retornos)`.
    """
    if not 0.0 < gae_lambda < 1.0:
        raise ValueError("gae_lambda deve estar em (0, 1) para o EPPO")
    c_next = ((1.0 - gae_lambda) / gae_lambda) ** 2
    c_acc = (gamma * gae_lambda) ** 2
    c_var = (1.0 - gae_lambda) / (1.0 + gae_lambda)
    advantages = torch.zeros_like(rewards)
    variances = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    var_acc = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(rewards.shape[0])):
        nonterminal = 1.0 - terminated[t].float()
        delta = (rewards[t] + gamma * nonterminal * next_values[t]
                 - values[t])
        # Fronteira de episódio (terminação ou truncamento): a vantagem e a
        # variância não atravessam para o episódio seguinte.
        propagates = (1.0 - episode_ends[t].float()) * nonterminal
        gae = delta + gamma * gae_lambda * propagates * gae
        var_acc = c_acc * (propagates * var_acc + next_values_var[t])
        advantages[t] = gae
        if variant == "cor":
            variances[t] = values_var[t] + c_next * var_acc
        elif variant == "ind":
            variances[t] = c_var * values_var[t] + c_next * var_acc
        else:
            raise ValueError(
                f"variante de variância desconhecida: {variant!r} "
                "(use 'cor' ou 'ind')")
    return advantages, variances, advantages + values


def evidential_nll(targets, omega, nu, alpha, beta):
    """NLL da NIG (máxima verossimilhança tipo II), média no lote.

    `Ω = 2β(1+ν)`; a expressão é a do paper (o termo constante `½log π`
    não altera o gradiente, mas fica explícito para casar com a fórmula).
    """
    omega_scale = 2.0 * beta * (1.0 + nu)
    nll = (0.5 * math.log(math.pi) - 0.5 * torch.log(nu)
           - alpha * torch.log(omega_scale)
           + (alpha + 0.5) * torch.log(
               nu * (targets - omega) ** 2 + omega_scale)
           + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5))
    return nll.mean()


def evidential_prior_log_prob(omega, nu, alpha, beta):
    """Log-probabilidade média das hiperpriors do paper.

    ω ~ N(0, 100²); ν ~ Gam(5,1); α−1 ~ Gam(5,1); β ~ Gam(5,1). É o
    `log p(m)` do regularizador `−ξ·log p(m)`.
    """
    zeros = torch.zeros_like(omega)
    p_omega = Normal(zeros, torch.full_like(omega, 100.0)).log_prob(omega)
    ones_nu = torch.ones_like(nu)
    p_nu = Gamma(torch.full_like(nu, 5.0), ones_nu).log_prob(nu)
    p_alpha = Gamma(torch.full_like(alpha, 5.0), ones_nu).log_prob(alpha - 1.0)
    p_beta = Gamma(torch.full_like(beta, 5.0), ones_nu).log_prob(beta)
    return (p_omega + p_nu + p_alpha + p_beta).mean()


class EPPOAgent:
    algorithm = "eppo"
    arch_version = "alg-eppo-v1"
    on_policy = True

    def __init__(self, obs_shape, n_actions, lr=3e-4, gamma=.99,
                 batch_size=64, rollout_steps=2048, epochs=10, clip=0.2,
                 gae_lambda=.95, entropy_coef=0.0, max_grad_norm=0.5,
                 ucb_radius=0.01, evidential_coef=0.01,
                 gae_variant="cor", device=None, n_agents=None,
                 team_ids=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.obs_shape = tuple(obs_shape)
        self.n_actions = int(n_actions)
        self.gamma = gamma
        self.batch_size = int(batch_size)
        self.rollout_steps = int(rollout_steps)
        self.epochs = int(epochs)
        self.clip = float(clip)
        self.gae_lambda = float(gae_lambda)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.ucb_radius = float(ucb_radius)
        self.evidential_coef = float(evidential_coef)
        self.gae_variant = str(gae_variant)
        # Aceitos para checkpoints/avaliação multiagente com política
        # compartilhada; o EPPO em si não usa esses campos.
        self.n_agents = n_agents
        self.team_ids = tuple(team_ids) if team_ids else ()
        self.online = EvidentialActorCriticNetwork(
            self.obs_shape, self.n_actions).to(self.device)
        # O contrato de checkpoint pede uma rede `target`; não há rede-alvo,
        # então o alias evita duplicar parâmetros.
        self.target = self.online
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = RolloutBuffer()
        self.steps = 0
        self.epsilon = 0.0
        self.last_grad_norm = 0.0
        self.last_uncertainty = 0.0

    @torch.no_grad()
    def act(self, state, greedy=False):
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        logits, _ = self.online(x)
        if greedy:
            return int(logits.argmax(dim=1).item())
        return int(Categorical(logits=logits).sample().item())

    @torch.no_grad()
    def act_batch(self, states, greedy=False):
        """Ações para um lote de estados, uma por linha (shape `(n,)`)."""
        x = torch.as_tensor(np.asarray(states), dtype=torch.float32,
                            device=self.device)
        logits, _ = self.online(x)
        if greedy:
            actions = logits.argmax(dim=1)
        else:
            actions = Categorical(logits=logits).sample()
        return actions.cpu().numpy().astype(np.int64)

    @torch.no_grad()
    def q_stats(self, state):
        """Média e máximo de `E[V] = ω` para o estado (contrato de stats)."""
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        _, raw = self.online(x)
        omega = raw[..., 0]
        return float(omega.mean()), float(omega.max())

    def head_weights(self):
        """Pesos da última camada do ator, para o histograma do TensorBoard."""
        return self.online.actor[-1].weight

    def end_episode(self, info=None):
        """Marca a fronteira de episódio no rollout; o update vem depois."""
        self.buffer.mark_episode_end()
        return None

    def finish_training(self):
        """Consome o rollout parcial que sobrou no fim do treino."""
        if len(self.buffer) == 0:
            return None
        return self._update()

    def learn(self):
        if len(self.buffer) < self.rollout_steps:
            return None
        return self._update()

    def _update(self):
        states, actions, rewards, next_states, terminated, episode_ends = \
            self.buffer.as_arrays()
        behavior_log_probs = self.buffer.log_probs_array()
        s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        ns = torch.as_tensor(next_states, dtype=torch.float32,
                             device=self.device)
        # GAE (média e variância) são recorrências escalares: em CUDA cada
        # passo sincroniza. Recompensas, médias e variâncias vão para a CPU,
        # como no PPO deste repo; a GPU só recebe as vantagens prontas.
        r = torch.as_tensor(rewards, dtype=torch.float32)
        term = torch.as_tensor(terminated, dtype=torch.float32)
        ends = torch.as_tensor(episode_ends, dtype=torch.float32)
        with torch.no_grad():
            logits_old, raw = self.online(s)
            _, raw_next = self.online(ns)
            values, values_var = evidential_moments(
                evidential_parameters(raw))
            next_values, next_values_var = evidential_moments(
                evidential_parameters(raw_next))
            if behavior_log_probs is None:
                # Coleta e update usam a mesma política: recalcular é exato.
                old_log_probs = torch.log_softmax(logits_old, dim=1).gather(
                    1, a.unsqueeze(1)).squeeze(1)
            else:
                # Coleta defasada (`--overlap`): o ratio do PPO precisa da
                # política de comportamento que gerou o rollout.
                old_log_probs = torch.as_tensor(
                    behavior_log_probs, dtype=torch.float32, device=self.device)
            mean_adv, adv_var, returns = compute_gae_with_uncertainty(
                r, values.cpu(), next_values.cpu(), values_var.cpu(),
                next_values_var.cpu(), term, ends,
                self.gamma, self.gae_lambda, self.gae_variant)
            mean_adv = mean_adv.to(self.device)
            adv_var = adv_var.to(self.device)
            returns = returns.to(self.device)
            self.last_uncertainty = float(
                torch.sqrt(adv_var.clamp_min(0.0)).mean().item())
            # Exploração dirigida: bônus de otimismo proporcional ao desvio
            # da vantagem, depois normalizado no lote (como no paper).
            advantages = mean_adv + self.ucb_radius * torch.sqrt(
                adv_var.clamp_min(0.0) + 1e-12)
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8)
        n = s.shape[0]
        last_loss = 0.0
        for _ in range(self.epochs):
            order = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                logits, raw = self.online(s[idx])
                omega, nu, alpha, beta = evidential_parameters(raw)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(a[idx])
                ratio = torch.exp(log_probs - old_log_probs[idx])
                adv = advantages[idx]
                unclipped = ratio * adv
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
                policy_loss = -torch.min(unclipped, clipped).mean()
                critic_loss = evidential_nll(
                    returns[idx], omega, nu, alpha, beta)
                if self.evidential_coef > 0.0:
                    critic_loss = critic_loss - self.evidential_coef * (
                        evidential_prior_log_prob(omega, nu, alpha, beta))
                entropy = dist.entropy().mean()
                loss = (policy_loss + critic_loss
                        - self.entropy_coef * entropy)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.last_grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    self.online.parameters(), self.max_grad_norm))
                self.optimizer.step()
                last_loss = float(loss.item())
        self.steps += 1
        self.buffer.clear()
        return last_loss


Agent = EPPOAgent
