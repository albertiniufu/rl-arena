"""Redes do HAPPO: atores heterogêneos + crítico centralizado.

O HAPPO (Heterogeneous-Agent PPO) existe para cenários em que os agentes não
são intercambiáveis — aqui, coletores e empurradores têm espaços de ação
diferentes. Isso quebra a premissa do IPPO/MAPPO de uma política
compartilhada, então cada agente tem o **seu próprio ator** (CNN + cabeça
categórica), enquanto o crítico continua centralizado no estilo CTDE:

* ``actors[i]`` vê apenas a observação local do agente ``i``;
* ``critic`` recebe as features de **todos** os agentes no mesmo passo mais
  um one-hot do agente consultado e devolve um V por agente. Ele só existe no
  treino.

A propriedade ``features`` aponta para o extrator do primeiro ator apenas
para a telemetria histórica do `train.py` (histograma da conv1); o update
nunca compartilha parâmetros entre atores, o que é essencial para a
atualização sequencial do HAPPO.
"""
import torch
from torch import nn


class ActorNetwork(nn.Module):
    """Ator local de um agente: CNN + cabeça categórica."""

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
            n_feat = self.features(torch.zeros(1, c, h, w)).shape[1]
        self.n_features = int(n_feat)
        self.actor = nn.Sequential(
            nn.Linear(self.n_features, 128), nn.ReLU(),
            nn.Linear(128, n_actions))

    def forward(self, x):
        return self.actor(self.features(x))


class CentralizedCritic(nn.Module):
    """Crítico centralizado: features conjuntas + one-hot → V por agente.

    A CNN que extrai features é compartilhada entre as linhas do lote (cada
    linha é a observação de um agente); assim o crítico não multiplica
    parâmetros por agente. A MLP final vê o estado conjunto e o one-hot.
    """

    def __init__(self, input_shape, n_agents):
        super().__init__()
        c, h, w = input_shape
        self.obs_shape = (c, h, w)
        self.n_agents = int(n_agents)
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
            n_feat = self.features(torch.zeros(1, c, h, w)).shape[1]
        self.n_features = int(n_feat)
        self.critic = nn.Sequential(
            nn.Linear(self.n_agents * n_feat + self.n_agents, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.register_buffer(
            "agent_onehot", torch.eye(self.n_agents), persistent=False)

    def joint_features(self, states):
        """Features de `(batch, n_agents, C, H, W)` → `(batch, n, F)`."""
        batch, n_agents = states.shape[0], states.shape[1]
        if n_agents != self.n_agents:
            raise ValueError(
                f"crítico espera {self.n_agents} agentes, "
                f"recebeu {n_agents}")
        flat = states.reshape(batch * n_agents, *self.obs_shape)
        feats = self.features(flat)
        return feats.reshape(batch, n_agents, self.n_features)

    def values_from_features(self, feats):
        """V por agente a partir de features conjuntas `(batch, n, F)`."""
        batch = feats.shape[0]
        joint = feats.reshape(batch, self.n_agents * self.n_features)
        shared = joint.unsqueeze(1).expand(batch, self.n_agents, joint.shape[1])
        onehot = self.agent_onehot.unsqueeze(0).expand(
            batch, self.n_agents, self.n_agents)
        return self.critic(torch.cat([shared, onehot], dim=2)).squeeze(-1)

    def joint_values(self, states):
        """V por agente direto das observações conjuntas."""
        return self.values_from_features(self.joint_features(states))


class HAPPONetwork(nn.Module):
    """Container que o contrato de checkpoint serializa como `online`."""

    def __init__(self, input_shape, n_actions, n_agents):
        super().__init__()
        self.n_agents = int(n_agents)
        self.n_actions = int(n_actions)
        self.actors = nn.ModuleList(
            [ActorNetwork(input_shape, n_actions)
             for _ in range(self.n_agents)])
        self.critic = CentralizedCritic(input_shape, self.n_agents)

    @property
    def features(self):
        """Extrator do primeiro ator (só para telemetria do treino)."""
        return self.actors[0].features

    def actor_logits(self, states, agent_index):
        """Logits do ator `agent_index` para `(batch, C, H, W)`."""
        return self.actors[int(agent_index)](states)

    def joint_values(self, states):
        """V centralizado `(batch, n_agents)` para `(batch, n, C, H, W)`."""
        return self.critic.joint_values(states)
