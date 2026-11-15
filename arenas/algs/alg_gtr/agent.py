"""GTR — Gaussian-Reshaped Trust Region Policy Optimization.

Liu et al., *Local Guidance, Global Impact: Gaussian-Reshaped Trust Region
Unlocks Behavior Transitions* (2026), arXiv:2606.03382.

O GTR é um plug-in para o PPO: mantém o surrogate recortado e soma uma
penalidade de confiança em forma gaussiana. Para a razão `r = πθ/π_ref`,

    L_Gaussian(r) = ½·(1 − e^{−(r−1)²})

é limitada (≤ ½) e não monotônica no gradiente: perto de r=1 reproduz a
região de confiança quadrática, mas o gradiente desaparece para r→0
("esquecimento") e para r grande ("liberação" para trocar de modo). O
coeficiente é adaptativo por amostra, `η = |Â|/ε`.

O "Mixture Gaussian Anchor" troca a referência única por K políticas de
épocas recentes (π₁ = μ, a política de comportamento):

    L_MixGaussian = (1/K) Σ_i [π_i/μ · ½(1 − e^{−(πθ/π_i − 1)²})]

Guardamos apenas os log-probs das ações amostradas em cada início de época
(vetores de tamanho do rollout), o que evita manter K redes inteiras.
K não é numérico no paper; o default é 4 (o mesmo número de épocas do PPO).
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .model import ActorCriticNetwork
from .rollout_buffer import RolloutBuffer


def compute_gae(rewards, values, next_values, terminated, episode_ends,
                gamma, gae_lambda):
    """Vantagens e retornos GAE(λ), respeitando fronteiras de episódio."""
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(rewards.shape[0])):
        nonterminal = 1.0 - terminated[t].float()
        delta = (rewards[t] + gamma * nonterminal * next_values[t]
                 - values[t])
        propagates = (1.0 - episode_ends[t].float()) * nonterminal
        gae = delta + gamma * gae_lambda * propagates * gae
        advantages[t] = gae
    return advantages, advantages + values


def gaussian_penalty(current_log_probs, anchor_log_probs, behavior_log_probs,
                     log_clamp=20.0):
    """Penalidade L_MixGaussian por amostra.

    `current_log_probs`: log-prob da ação tomada sob πθ.
    `anchor_log_probs`: lista de vetores (um por política-âncora π_i).
    `behavior_log_probs`: log-prob da ação sob μ, que gerou o rollout.

    As razões são calculadas em espaço logarítmico e recortadas em
    `±log_clamp` por estabilidade numérica (as âncoras podem ter
    probabilidade muito menor que μ para a ação amostrada).
    """
    if not anchor_log_probs:
        raise ValueError("GTR requer pelo menos uma âncora")
    total = torch.zeros_like(current_log_probs)
    for anchor in anchor_log_probs:
        r_anchor = torch.exp(
            (anchor - behavior_log_probs).clamp(-log_clamp, log_clamp))
        r_current = torch.exp(
            (current_log_probs - anchor).clamp(-log_clamp, log_clamp))
        total = total + r_anchor * 0.5 * (
            1.0 - torch.exp(-(r_current - 1.0) ** 2))
    return total / len(anchor_log_probs)


class GTRAgent:
    algorithm = "gtr"
    arch_version = "alg-gtr-v1"
    on_policy = True

    def __init__(self, obs_shape, n_actions, lr=3e-4, gamma=.99,
                 batch_size=64, rollout_steps=1024, epochs=4, clip=0.2,
                 gae_lambda=.95, value_coef=.5, entropy_coef=.01,
                 max_grad_norm=10.0, anchor_count=4, device=None,
                 n_agents=None, team_ids=None):
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
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.anchor_count = int(anchor_count)
        self.n_agents = n_agents
        self.team_ids = tuple(team_ids) if team_ids else ()
        self.online = ActorCriticNetwork(
            self.obs_shape, self.n_actions).to(self.device)
        self.target = self.online
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = RolloutBuffer()
        self.steps = 0
        self.epsilon = 0.0
        self.last_grad_norm = 0.0
        self.last_penalty = 0.0

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
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        _, value = self.online(x)
        return float(value.mean()), float(value.max())

    def head_weights(self):
        return self.online.actor[-1].weight

    def end_episode(self, info=None):
        self.buffer.mark_episode_end()
        return None

    def finish_training(self):
        if len(self.buffer) == 0:
            return None
        return self._update()

    def learn(self):
        if len(self.buffer) < self.rollout_steps:
            return None
        return self._update()

    @torch.no_grad()
    def _action_log_probs(self, states, actions):
        logits, _ = self.online(states)
        return torch.log_softmax(logits, dim=1).gather(
            1, actions.unsqueeze(1)).squeeze(1)

    def _update(self):
        states, actions, rewards, next_states, terminated, episode_ends = \
            self.buffer.as_arrays()
        behavior_log_probs = self.buffer.log_probs_array()
        s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        ns = torch.as_tensor(next_states, dtype=torch.float32,
                             device=self.device)
        r = torch.as_tensor(rewards, dtype=torch.float32)
        term = torch.as_tensor(terminated, dtype=torch.float32)
        ends = torch.as_tensor(episode_ends, dtype=torch.float32)
        with torch.no_grad():
            logits_old, values = self.online(s)
            _, next_values = self.online(ns)
            if behavior_log_probs is None:
                old_log_probs = torch.log_softmax(logits_old, dim=1).gather(
                    1, a.unsqueeze(1)).squeeze(1)
            else:
                old_log_probs = torch.as_tensor(
                    behavior_log_probs, dtype=torch.float32, device=self.device)
            advantages, returns = compute_gae(
                r, values.cpu(), next_values.cpu(), term, ends,
                self.gamma, self.gae_lambda)
            advantages = advantages.to(self.device)
            returns = returns.to(self.device)
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8)
        # π₁ = μ (política de comportamento); as demais âncoras são os
        # snapshots no início de cada época, das mais antigas às mais novas.
        anchors = [old_log_probs.detach()]
        n = s.shape[0]
        last_loss = 0.0
        for epoch in range(self.epochs):
            if epoch > 0:
                anchors.append(self._action_log_probs(s, a))
                if len(anchors) > self.anchor_count:
                    anchors.pop(0)
            order = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                logits, values = self.online(s[idx])
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(a[idx])
                ratio = torch.exp(log_probs - old_log_probs[idx])
                adv = advantages[idx]
                unclipped = ratio * adv
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
                policy_loss = -torch.min(unclipped, clipped).mean()
                # Penalidade gaussiana misturada, com η = |Â|/ε por amostra.
                mix = gaussian_penalty(
                    log_probs, [anchor[idx] for anchor in anchors],
                    old_log_probs[idx])
                eta = adv.abs().detach() / self.clip
                gtr_loss = (eta * mix).mean()
                value_loss = nn.functional.mse_loss(values, returns[idx])
                entropy = dist.entropy().mean()
                loss = (policy_loss + gtr_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.last_grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    self.online.parameters(), self.max_grad_norm))
                self.optimizer.step()
                self.last_penalty = float(gtr_loss.item())
                last_loss = float(loss.item())
        self.steps += 1
        self.buffer.clear()
        return last_loss


Agent = GTRAgent
