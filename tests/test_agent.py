import numpy as np
import pytest

from arena.config import ArenaConfig
from arena.vector import make_vector_env
from dqn.agent import DQNAgent
from dqn.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from train import obs_to_image, parse_args, validate_train_args


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
    s, a, r_n, ns, term, window = buf.sample(4, n=3, gamma=0.99)
    assert window.dtype == np.int64
    # A amostra do estado s0 percorre a janela completa de 3 passos.
    j0 = next(j for j, st in enumerate(s) if np.array_equal(st, states[0]))
    assert a[j0] == 0
    assert r_n[j0] == pytest.approx(1 + 0.99 * 2 + 0.99**2 * 3)
    assert np.array_equal(ns[j0], states[3])
    assert term[j0] == 0.0
    assert window[j0] == 3
    # Perto do fim do buffer, a janela para no último elemento.
    j2 = next(j for j, st in enumerate(s) if np.array_equal(st, states[2]))
    assert r_n[j2] == pytest.approx(3 + 0.99 * 4)
    assert np.array_equal(ns[j2], states[4])
    assert window[j2] == 2  # existem só as transições 2 e 3 à frente

    # Terminated na transição 1 clampa a janela e zera o bootstrap:
    # ns/term vêm da transição terminal (reward incluído).
    buf2 = ReplayBuffer(10)
    for i in range(4):
        buf2.add(states[i], i, float(i + 1), states[i + 1], i == 1, False)
    s2, _, r_n2, ns2, term2, window2 = buf2.sample(4, n=3, gamma=0.99)
    j0 = next(j for j, st in enumerate(s2) if np.array_equal(st, states[0]))
    assert r_n2[j0] == pytest.approx(1 + 0.99 * 2)
    assert np.array_equal(ns2[j0], states[2])
    assert term2[j0] == 1.0
    assert window2[j0] == 2  # a janela inclui a transição terminal
    # A transição terminal sorteada não acumula: r_n == r1, ns == s2.
    j1 = next(j for j, st in enumerate(s2) if np.array_equal(st, states[1]))
    assert r_n2[j1] == pytest.approx(2.0)
    assert term2[j1] == 1.0
    assert window2[j1] == 1

    # Truncated também clampa, mas mantém o bootstrap (term == 0).
    buf3 = ReplayBuffer(10)
    for i in range(4):
        buf3.add(states[i], i, float(i + 1), states[i + 1], False, i == 1)
    s3, _, r_n3, ns3, term3, window3 = buf3.sample(4, n=3, gamma=0.99)
    j0 = next(j for j, st in enumerate(s3) if np.array_equal(st, states[0]))
    assert r_n3[j0] == pytest.approx(1 + 0.99 * 2)
    assert np.array_equal(ns3[j0], states[2])
    assert term3[j0] == 0.0
    assert window3[j0] == 2

    # n=1 reproduz o comportamento antigo: r_n == r, ns == next_state,
    # term == terminated e a janela tem sempre 1 transição.
    s1_, a1_, r_n1, ns1_, term1_, window1_ = buf2.sample(4, n=1)
    for j, st in enumerate(s1_):
        i = int(st[0, 0])
        assert a1_[j] == i
        assert r_n1[j] == pytest.approx(float(i + 1))
        assert np.array_equal(ns1_[j], states[i + 1])
        assert term1_[j] == (1.0 if i == 1 else 0.0)
        assert window1_[j] == 1


def test_td_targets_double():
    """Alvo de TD: com double, a ação vem da online e o valor da target.

    Redes minúsculas e fixas: a online prefere a ação 2 em `ns`, enquanto o
    máximo da target está na ação 3 — assim as duas variantes se distinguem.
    O bootstrap desconta por `gamma ** window` (janela real da amostra) e a
    segunda amostra é terminal (term == 1), então não pode bootstrapar.
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
    ns = torch.zeros(3, 3)
    r_n = torch.tensor([1.0, 2.0, 3.0])  # já descontado pelo buffer: não descontar de novo
    term = torch.tensor([0.0, 1.0, 0.0])
    window = torch.tensor([3, 2, 1], dtype=torch.int64)  # comprimento real da janela
    gamma = 0.9

    double = _td_targets(online, target, r_n, ns, term, gamma, window=window, double=True)
    # Janela completa (k=3): 1.0 + 0.9**3 * target[2].
    # Amostra terminal (term=1): bootstrap zerado, sobra r_n.
    # Janela curta (k=1): 3.0 + 0.9**1 * target[2].
    assert torch.allclose(double, torch.tensor([
        1.0 + 0.9**3 * 3.0, 2.0, 3.0 + 0.9 * 3.0]))

    vanilla = _td_targets(online, target, r_n, ns, term, gamma, window=window, double=False)
    # Sem double: o valor vem do máximo da target (4.0).
    assert torch.allclose(vanilla, torch.tensor([
        1.0 + 0.9**3 * 4.0, 2.0, 3.0 + 0.9 * 4.0]))


def test_dueling_equals_v_plus_advantage():
    """Com dueling, o forward é V + (A - média(A)) sobre os dois ramos."""
    import torch
    from dqn.model import DQN

    net = DQN((5, 20, 32), 12, dueling=True)
    x = torch.randn(1, 5, 20, 32)
    feats = net.features(x)
    v = net.v_head(feats)
    a = net.a_head(feats)
    expected = v + a - a.mean(dim=1, keepdim=True)
    assert torch.allclose(net(x), expected)
    assert net.head is None


def test_dueling_learn_smoke():
    """Um passo de aprendizado exercita target dueling, gather e backward."""
    agent = DQNAgent((5, 20, 32), 12, dueling=True, batch_size=2)
    obs = np.zeros((5, 20, 32), dtype=np.float32)
    for action in (0, 1):
        agent.buffer.add(obs, action, 1.0, obs, False, False)
    loss = agent.learn()
    assert loss is not None


def test_per_sample_favors_high_priority():
    """Rank-based: prioridade alta vira rank 1 e domina a amostragem."""
    buf = PrioritizedReplayBuffer(100)
    # Sorteio usa o RNG global (semeado pelo train.py). Seed fixa deixa a
    # contagem determinística; seed 1 dá ~139 vs ~8, representando o esperado.
    np.random.seed(1)
    obs = np.zeros((5, 20, 32), dtype=np.float32)
    for i in range(20):
        buf.add(obs, 0, float(i + 1), obs, False, False)
    # Sem erro informado, todas as transições nascem com a mesma prioridade.
    assert len(set(buf.priorities)) == 1
    buf.update_priorities([3], [1000.0])  # índice 3 vira rank 1
    out = buf.sample(500, n=1)
    assert len(out) == 7  # s, a, r_n, ns, term, window, indices
    s, a, r_n, ns, term, window, indices = out
    assert indices.dtype == np.int64
    assert len(indices) == 500
    # O índice 17 ficou com rank 18 (peso 1/18): aparece bem menos.
    count_high = int((indices == 3).sum())
    count_low = int((indices == 17).sum())
    assert count_high > 10 * count_low
    # Consistência dos campos: com n=1, r_n é a recompensa da transição amostrada.
    for j in range(8):
        assert r_n[j] == pytest.approx(float(indices[j] + 1))
        assert np.array_equal(s[j], obs)
        assert np.array_equal(ns[j], obs)
        assert window[j] == 1


def test_per_sample_n_step_windows():
    """O walk n-step do PER é idêntico ao do ReplayBuffer, sobre o snapshot."""
    buf = PrioritizedReplayBuffer(10)
    states = [np.full((2, 2), float(i)) for i in range(5)]
    for i in range(4):
        buf.add(states[i], i, float(i + 1), states[i + 1], False, False)
    s, a, r_n, ns, term, window, indices = buf.sample(4, n=3, gamma=0.99)
    assert window.dtype == np.int64
    assert indices.dtype == np.int64
    for j in range(4):
        idx = int(indices[j])
        # Sem terminais, a janela para só no fim do buffer.
        expected_window = min(3, 4 - idx)
        assert window[j] == expected_window
        expected_r = sum(0.99 ** k * float(idx + k + 1) for k in range(expected_window))
        assert r_n[j] == pytest.approx(expected_r)
        assert a[j] == idx
        assert np.array_equal(s[j], states[idx])
        assert np.array_equal(ns[j], states[idx + expected_window])
        assert term[j] == 0.0


def test_per_update_priorities_and_new_default():
    """update_priorities grava |erro| + 1e-6; add sem prioridade herda a máxima."""
    buf = PrioritizedReplayBuffer(100)
    obs = np.zeros((5, 20, 32), dtype=np.float32)
    buf.add(obs, 0, 1.0, obs, False, False)
    buf.add(obs, 1, 2.0, obs, False, False)
    assert list(buf.priorities) == [1.0, 1.0]
    buf.update_priorities([0], [0.5])
    assert buf.priorities[0] == pytest.approx(0.5 + 1e-6)
    buf.add(obs, 2, 3.0, obs, False, False)  # sem prioridade
    assert buf.priorities[2] == pytest.approx(1.0)  # maior prioridade atual
    assert len(buf) == 3


def test_agent_per_learn_updates_priorities():
    """Com per=True, learn() amostra com índices e reescreve as prioridades."""
    # Seeds fixas: o sorteio do buffer e a init da rede ficam determinísticos
    # e as duas transições são amostradas (índices [0, 1]).
    import torch
    np.random.seed(0)
    torch.manual_seed(0)
    agent = DQNAgent((5, 20, 32), 12, per=True, batch_size=2)
    assert isinstance(agent.buffer, PrioritizedReplayBuffer)
    assert isinstance(DQNAgent((5, 20, 32), 12).buffer, ReplayBuffer)
    obs = np.zeros((5, 20, 32), dtype=np.float32)
    for action in (0, 1):
        agent.buffer.add(obs, action, 1.0, obs, False, False)
    loss = agent.learn()
    assert loss is not None
    assert agent._last_indices is not None
    assert len(agent._last_indices) == 2
    # As prioridades deixaram de ser 1.0: foram trocadas por |erro| + 1e-6.
    assert all(p != pytest.approx(1.0) for p in agent.buffer.priorities)


def test_vector_env_and_act_batch():
    """VectorEnv síncrono (sem subprocessos) e ações em lote com shapes reais."""
    venv = make_vector_env(2, ArenaConfig(), "random", asynchronous=False)
    try:
        obs, _ = venv.reset(seed=7)
        assert obs.shape == (2, 5, 20, 32)
        agent = DQNAgent((5, 20, 32), 12)
        acts = agent.act_batch(obs, greedy=True)
        assert acts.shape == (2,)
        assert np.issubdtype(acts.dtype, np.integer)
        obs2, rew, term, trunc, info = venv.step(acts)
        assert obs2.shape == (2, 5, 20, 32)
        assert rew.shape == (2,)
    finally:
        venv.close()


def test_add_batch_both_buffers():
    """add_batch adiciona em ordem nos dois buffers; PER usa a prioridade padrão."""
    obs = np.zeros((5, 20, 32), dtype=np.float32)
    transitions = [(obs, i, float(i), obs, False, False) for i in range(3)]
    for cls in (ReplayBuffer, PrioritizedReplayBuffer):
        buf = cls(10)
        buf.add_batch(transitions)
        assert len(buf) == 3
    per = PrioritizedReplayBuffer(10)
    per.add_batch(transitions)
    # Sem prioridade explícita, cada transição herda a maior prioridade atual (1.0).
    assert list(per.priorities) == [1.0, 1.0, 1.0]
    sample = per.sample(3, n=1)
    assert len(sample) == 7  # s, a, r_n, ns, term, window, indices
    indices = sample[-1]
    assert indices.dtype == np.int64
    assert ((indices >= 0) & (indices < 3)).all()


def test_act_batch_greedy_matches_act():
    """greedy ignora o epsilon (1.0) e escolhe a mesma ação que act()."""
    agent = DQNAgent((5, 20, 32), 12, epsilon=1.0)
    state = np.zeros((5, 20, 32), dtype=np.float32)
    assert agent.act(state, greedy=True) == agent.act_batch(
        np.stack([state]), greedy=True)[0]
    # Sem greedy e com epsilon=1.0, cada linha é sorteada (sem consultar a rede).
    acts = agent.act_batch(np.stack([state, state]), greedy=False)
    assert acts.shape == (2,)
    assert np.issubdtype(acts.dtype, np.integer)


def test_parse_args_envs_flag():
    """--envs é opcional (default 1) e aceita N > 1."""
    assert parse_args([]).envs == 1
    assert parse_args(["--envs", "4"]).envs == 4


def test_envs_validation_rejects_incompatible_flags():
    """--envs > 1 exige adversário fixo e desliga render/telemetria."""
    cases = [
        ["--envs", "4"],  # opponent auto é o default
        ["--envs", "4", "--opponent", "random", "--render"],
        ["--envs", "4", "--opponent", "random", "--stats"],
        ["--envs", "4", "--opponent", "random", "--tensorboard"],
        ["--envs", "0", "--opponent", "random"],
    ]
    for argv in cases:
        with pytest.raises(SystemExit):
            validate_train_args(parse_args(argv))
    # Caminho válido da Fase 5: adversário fixo e telemetria desligada.
    validate_train_args(parse_args(["--envs", "4", "--opponent", "random"]))


def test_envs_gt_1_rejects_n_step_gt_1():
    """Com workers concorrentes, a janela n-step do buffer atravessaria workers."""
    with pytest.raises(SystemExit):
        validate_train_args(parse_args(
            ["--envs", "2", "--opponent", "random", "--n-step", "3"]))
    # n-step 1 no vetor e n-step > 1 no loop histórico continuam aceitos.
    validate_train_args(parse_args(
        ["--envs", "2", "--opponent", "random", "--n-step", "1"]))
    validate_train_args(parse_args(["--envs", "1", "--n-step", "3"]))


def test_make_vector_env_validates_args():
    """make_vector_env rejeita n_envs inválido e adversário que não seja nome."""
    with pytest.raises(ValueError):
        make_vector_env(0, ArenaConfig(), "random", asynchronous=False)
    with pytest.raises(ValueError):
        make_vector_env(1, ArenaConfig(), object(), asynchronous=False)


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
