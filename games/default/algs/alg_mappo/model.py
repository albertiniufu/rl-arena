"""Rede actor-critic com crítico centralizado do MAPPO.

A diferença central para o IPPO está no crítico (CTDE — treino centralizado,
execução descentralizada):

* o **ator** é compartilhado e vê só a observação local (egocêntrica) do
  agente que age — na execução não há informação global;
* o **crítico** é centralizado: recebe a representação de todos os agentes no
  mesmo instante e devolve um V por agente. Como a observação egocêntrica de
  cada agente já separa self/inimigos/aliados, a concatenação das features
  de todos é um estado global; o one-hot do agente permite que o crítico
  aprenda valores diferentes por papel (importante em times).

A CNN é compartilhada entre ator e crítico: o mesmo extrator aprende tanto a
política quanto a função de valor, e o gradiente do crítico também atualiza a
base convolucional. `joint_values` só é usado no update; `forward` devolve os
logits do ator.
"""
import torch
from torch import nn


class CentralizedActorCriticNetwork(nn.Module):
    """CNN compartilhada + ator local + crítico centralizado.

    `forward(x)` recebe observações locais `(batch, C, H, W)` e devolve os
    logits do ator `(batch, n_actions)`. O crítico é chamado por
    `joint_values(feats)` com as features de **todos** os agentes.
    """
    def __init__(self, input_shape, n_actions, n_agents):
        super().__init__()
        c, h, w = input_shape
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
        self.actor = nn.Sequential(
            nn.Linear(n_feat, 128), nn.ReLU(), nn.Linear(128, n_actions))
        # Entrada: features de todos os agentes concatenadas + one-hot do
        # agente consultado. A saída é escalar (V para aquele agente).
        self.critic = nn.Sequential(
            nn.Linear(self.n_agents * n_feat + self.n_agents, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        # persistent=False: não entra no state_dict do checkpoint (é derivável
        # do número de agentes gravado na config).
        self.register_buffer(
            "agent_onehot", torch.eye(self.n_agents), persistent=False)

    def forward(self, x):
        """Logits do ator para observações locais `(batch, C, H, W)`."""
        return self.actor(self.features(x))

    def joint_values(self, feats):
        """V centralizado por agente a partir das features conjuntas.

        `feats`: `(batch, n_agents, n_features)` na ordem dos agentes.
        Devolve `(batch, n_agents)`.
        """
        batch = feats.shape[0]
        joint = feats.reshape(batch, self.n_agents * self.n_features)
        # Todo agente vê o mesmo estado global; o one-hot o diferencia.
        shared = joint.unsqueeze(1).expand(
            batch, self.n_agents, joint.shape[1])
        onehot = self.agent_onehot.unsqueeze(0).expand(
            batch, self.n_agents, self.n_agents)
        critic_in = torch.cat([shared, onehot], dim=2)
        return self.critic(critic_in).squeeze(-1)
