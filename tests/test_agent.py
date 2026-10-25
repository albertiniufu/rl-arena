import numpy as np
import pytest

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from train import epsilon_for_episode, obs_to_image


def test_epsilon_for_episode():
    # Rampa linear de 1.0 no episódio 1 até o piso no episódio `ramp_end`.
    assert epsilon_for_episode(1, 60, 0.1) == pytest.approx(1.0 - (1 - 0.1) / 60 * 1, abs=1e-9)
    assert epsilon_for_episode(60, 60, 0.1) == pytest.approx(0.1)
    assert epsilon_for_episode(61, 60, 0.1) == pytest.approx(0.1)


def test_warmup():
    agent = DQNAgent((5, 32, 20), 12, warmup=1000)
    for _ in range(999):
        agent.buffer.add(np.zeros((5, 32, 20)), 0, 0.0, np.zeros((5, 32, 20)), False, False)
    assert agent.learn() is None
    agent.buffer.add(np.zeros((5, 32, 20)), 0, 0.0, np.zeros((5, 32, 20)), False, False)
    assert agent.learn() is not None  # 1000 transições


def test_learn_does_not_decay_epsilon():
    """O epsilon é agendado por episódio em train.py; o agente só o lê."""
    agent = DQNAgent((5, 32, 20), 12, epsilon=0.5)
    zeros = np.zeros((5, 32, 20), dtype=np.float32)
    for _ in range(2000):
        agent.buffer.add(zeros, 0, 0.0, zeros, False, False)
    assert agent.learn() is not None  # buffer cheio: o passo de aprendizado roda
    assert agent.epsilon == 0.5


def test_obs_to_image_uses_last_frames():
    # Empilhada (frame_stack=4): (20, 20, 32) = 4 frames de (5, 20, 32).
    obs = np.zeros((20, 20, 32), dtype=np.float32)
    obs[:5, 0, 0] = 1.0   # frame antigo com ruído
    obs[-5:, 1, 1] = 1.0  # frame atual com jogador
    img = obs_to_image(obs)
    assert np.array_equal(img, obs_to_image(obs[-5:]))


def test_sample_n_step():
    """Retornos n-step: acumula γ^k·r_{t+k} e para na fronteira de episódio."""
    buf = ReplayBuffer(10)
    # 4 transições contíguas, sem terminais: r = 1, 2, 3, 4; estados s0..s4.
    states = [np.full((2, 2), float(i)) for i in range(5)]
    for i in range(4):
        buf.add(states[i], i, float(i + 1), states[i + 1], False, False)
    s, a, r_n, ns, term = buf.sample(4, n=3, gamma=0.99)
    # A amostra do estado s0 percorre a janela completa de 3 passos.
    j0 = next(j for j, st in enumerate(s) if np.array_equal(st, states[0]))
    assert a[j0] == 0
    assert r_n[j0] == pytest.approx(1 + 0.99 * 2 + 0.99**2 * 3)
    assert np.array_equal(ns[j0], states[3])
    assert term[j0] == 0.0
    # Perto do fim do buffer, a janela para no último elemento.
    j2 = next(j for j, st in enumerate(s) if np.array_equal(st, states[2]))
    assert r_n[j2] == pytest.approx(3 + 0.99 * 4)
    assert np.array_equal(ns[j2], states[4])

    # Terminated na transição 1 clampa a janela e zera o bootstrap:
    # ns/term vêm da transição terminal (reward incluído).
    buf2 = ReplayBuffer(10)
    for i in range(4):
        buf2.add(states[i], i, float(i + 1), states[i + 1], i == 1, False)
    s2, _, r_n2, ns2, term2 = buf2.sample(4, n=3, gamma=0.99)
    j0 = next(j for j, st in enumerate(s2) if np.array_equal(st, states[0]))
    assert r_n2[j0] == pytest.approx(1 + 0.99 * 2)
    assert np.array_equal(ns2[j0], states[2])
    assert term2[j0] == 1.0
    # A transição terminal sorteada não acumula: r_n == r1, ns == s2.
    j1 = next(j for j, st in enumerate(s2) if np.array_equal(st, states[1]))
    assert r_n2[j1] == pytest.approx(2.0)
    assert term2[j1] == 1.0

    # Truncated também clampa, mas mantém o bootstrap (term == 0).
    buf3 = ReplayBuffer(10)
    for i in range(4):
        buf3.add(states[i], i, float(i + 1), states[i + 1], False, i == 1)
    s3, _, r_n3, ns3, term3 = buf3.sample(4, n=3, gamma=0.99)
    j0 = next(j for j, st in enumerate(s3) if np.array_equal(st, states[0]))
    assert r_n3[j0] == pytest.approx(1 + 0.99 * 2)
    assert np.array_equal(ns3[j0], states[2])
    assert term3[j0] == 0.0

    # n=1 reproduz o comportamento antigo: r_n == r, ns == next_state, term == terminated.
    s1_, a1_, r_n1, ns1_, term1_ = buf2.sample(4, n=1)
    for j, st in enumerate(s1_):
        i = int(st[0, 0])
        assert a1_[j] == i
        assert r_n1[j] == pytest.approx(float(i + 1))
        assert np.array_equal(ns1_[j], states[i + 1])
        assert term1_[j] == (1.0 if i == 1 else 0.0)


def test_td_targets_double():
    """Alvo de TD: com double, a ação vem da online e o valor da target.

    Redes minúsculas e fixas: a online prefere a ação 2 em `ns`, enquanto o
    máximo da target está na ação 3 — assim as duas variantes se distinguem.
    A segunda amostra é terminal (term == 1) e não pode bootstrapar.
    """
    import torch
    from torch import nn
    from dqn.agent import _td_targets

    class FixedQ(nn.Module):
        """Devolve Q-valores fixos (uma linha) para qualquer entrada."""
        def __init__(self, values):
            super().__init__()
            self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

        def forward(self, x):
            return self.values.expand(x.shape[0], -1)

    online = FixedQ([0.0, 1.0, 5.0, 2.0])  # argmax = ação 2
    target = FixedQ([0.0, 0.5, 3.0, 4.0])  # max = ação 3 (4.0)
    ns = torch.zeros(2, 3)
    r_n = torch.tensor([1.0, 2.0])  # já descontado pelo buffer: não descontar de novo
    term = torch.tensor([0.0, 1.0])
    gamma = 0.9

    double = _td_targets(online, target, r_n, ns, term, gamma, double=True)
    # Ação 2 escolhida pela online: 1.0 + 0.9 * target[2] = 3.7.
    assert torch.allclose(double, torch.tensor([1.0 + 0.9 * 3.0, 2.0]))

    vanilla = _td_targets(online, target, r_n, ns, term, gamma, double=False)
    # Sem double: 1.0 + 0.9 * max(target) = 4.6.
    assert torch.allclose(vanilla, torch.tensor([1.0 + 0.9 * 4.0, 2.0]))


def test_checkpoint_policy_uses_stacked_obs():
    """Com frame stack, o adversário checkpoint age sobre a obs empilhada:
    reset + step completam sem erro de shape na CNN."""
    from arena.config import ArenaConfig
    from arena.environment import ArenaEnv
    from dqn.agent import CheckpointPolicy

    agent = DQNAgent((20, 20, 32), 12)
    env = ArenaEnv(ArenaConfig(frame_stack=4), opponent_policy=CheckpointPolicy(agent))
    obs, _ = env.reset(seed=1)
    assert obs.shape == (20, 20, 32)
    next_obs, *_ = env.step(1)  # não pode estourar por shape no conv
    assert next_obs.shape == (20, 20, 32)
