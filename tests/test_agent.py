import argparse
import random

import numpy as np
import pytest
import torch
from torch import nn

from alg_doubledqn_per_dueling.agent import (
    DoubleDQNPerDuelingAgent,
    double_td_target,
)
from alg_doubledqn_per_dueling.model import DuelingQNetwork
from alg_doubledqn_per_dueling.replay_buffer import PrioritizedReplayBuffer
from alg_dqn.agent import DQNAgent, td_target
from alg_dqn.model import QNetwork
from alg_dqn.replay_buffer import ReplayBuffer
from alg_registry import CheckpointPolicy, available_algorithms, load_algorithm
from arena.config import ArenaConfig
from arena.environment import ArenaEnv
from arena.vector import make_vector_env
from train import (
    epsilon_for_episode,
    obs_to_image,
    parse_args,
    schedule_epsilon,
    tensorboard_run_name,
    time_budget_reached,
    validate_train_args,
)


def test_algorithm_discovery_and_loading():
    """Os nomes da CLI são os sufixos dos pacotes `alg_*`."""
    names = available_algorithms()
    assert "dqn" in names and "doubledqn_per_dueling" in names
    assert names == sorted(names)
    module = load_algorithm("dqn")
    assert module.Agent is DQNAgent
    assert load_algorithm("doubledqn_per_dueling").Agent is DoubleDQNPerDuelingAgent
    with pytest.raises(ValueError):
        load_algorithm("inexistente")


def test_plain_network_returns_direct_q_per_action():
    net = QNetwork((5, 20, 32), 6)
    out = net(torch.zeros(2, 5, 20, 32))
    assert out.shape == (2, 6)
    assert hasattr(net, "head")
    assert not hasattr(net, "v_head") and not hasattr(net, "a_head")


def test_dueling_network_is_v_plus_centered_advantage():
    net = DuelingQNetwork((5, 20, 32), 6)
    x = torch.randn(1, 5, 20, 32)
    feats = net.features(x)
    v = net.v_head(feats)
    a = net.a_head(feats)
    expected = v + a - a.mean(dim=1, keepdim=True)
    assert torch.allclose(net(x), expected)
    assert not hasattr(net, "head")


def test_td_target_bootstraps_only_on_true_terminals():
    reward = torch.tensor([1.0, 2.0])
    next_q = torch.tensor([4.0, 5.0])
    terminated = torch.tensor([0.0, 1.0])
    out = td_target(reward, next_q, terminated, gamma=0.9)
    assert torch.allclose(out, torch.tensor([1.0 + 0.9 * 4.0, 2.0]))


def test_double_td_target_online_chooses_target_evaluates():
    class FixedQ(nn.Module):
        def __init__(self, values):
            super().__init__()
            self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

        def forward(self, x):
            return self.values.expand(x.shape[0], -1)

    online = FixedQ([0.0, 1.0, 5.0, 2.0])  # escolhe a ação 2
    target = FixedQ([0.0, 0.5, 3.0, 4.0])  # max seria a ação 3
    next_states = torch.zeros(2, 3)
    reward = torch.tensor([1.0, 2.0])
    terminated = torch.tensor([0.0, 1.0])
    out = double_td_target(online, target, next_states, reward, terminated, gamma=0.9)
    # A ação 2 vale 3.0 na target (não o máximo 4.0 da ação 3).
    assert torch.allclose(out, torch.tensor([1.0 + 0.9 * 3.0, 2.0]))


def test_uniform_buffer_gives_each_transition_same_chance():
    """Sorteio uniforme: cada transição tem a mesma chance de entrar no batch."""
    buf = ReplayBuffer(100)
    obs = np.zeros((2,), dtype=np.float32)
    for i in range(5):
        buf.add(obs, i, float(i), obs, False)
    random.seed(0)
    counts = np.zeros(5)
    for _ in range(1000):
        # sample() é sem reposição: 4 de 5 por chamada, ~800 inclusões cada.
        s, a, r, ns, term = buf.sample(4)
        assert len(s) == len(a) == len(r) == len(ns) == len(term) == 4
        counts[np.asarray(r, dtype=int)] += 1
    assert counts.min() > 650 and counts.max() < 950


def test_per_buffer_favors_high_priority():
    """Rank-based: prioridade alta vira rank 1 e domina a amostragem (α atenua)."""
    buf = PrioritizedReplayBuffer(100)
    np.random.seed(1)
    obs = np.zeros((5, 20, 32), dtype=np.float32)
    for i in range(20):
        buf.add(obs, 0, float(i + 1), obs, False)
    assert len(set(buf.priorities)) == 1
    buf.update_priorities([3], [1000.0])
    s, a, r, ns, term, indices, weights = buf.sample(2000)
    assert indices.dtype == np.int64
    assert len(indices) == 2000
    count_high = int((indices == 3).sum())
    count_low = int((indices == 17).sum())
    assert count_high > 4 * count_low
    for j in range(8):
        assert r[j] == pytest.approx(float(indices[j] + 1))
        assert np.array_equal(s[j], obs)
        assert np.array_equal(ns[j], obs)
        assert term[j] == 0.0
        assert 0.0 < weights[j] <= 1.0


def test_per_update_priorities_and_new_default():
    """update_priorities grava |erro| + 1e-6; add sem prioridade herda a máxima."""
    buf = PrioritizedReplayBuffer(100)
    obs = np.zeros((5, 20, 32), dtype=np.float32)
    buf.add(obs, 0, 1.0, obs, False)
    buf.add(obs, 1, 2.0, obs, False)
    assert list(buf.priorities) == [1.0, 1.0]
    buf.update_priorities([0], [0.5])
    assert buf.priorities[0] == pytest.approx(0.5 + 1e-6)
    buf.add(obs, 2, 3.0, obs, False)  # sem prioridade
    assert buf.priorities[2] == pytest.approx(1.0)
    assert len(buf) == 3


def test_per_sample_returns_importance_weights():
    """sample() devolve pesos de importância normalizados pelo máximo."""
    buf = PrioritizedReplayBuffer(100)
    obs = np.zeros((5, 20, 32), dtype=np.float32)
    for i in range(50):
        buf.add(obs, 0, float(i + 1), obs, False)
    np.random.seed(2)
    s, a, r, ns, term, indices, weights = buf.sample(32, alpha=0.6, beta=0.4)
    assert weights.shape == (32,)
    assert np.all(weights > 0) and np.all(np.isfinite(weights))
    assert weights.max() == pytest.approx(1.0)


def test_per_beta_zero_has_unit_weights():
    """Com β=0 não há correção de viés: todos os pesos valem 1."""
    buf = PrioritizedReplayBuffer(100)
    obs = np.zeros((5, 20, 32), dtype=np.float32)
    for i in range(30):
        buf.add(obs, 0, 1.0, obs, False)
    np.random.seed(3)
    *_, indices, weights = buf.sample(16, alpha=0.6, beta=0.0)
    assert np.allclose(weights, 1.0)


def test_per_alpha_controls_concentration():
    """α menor espalha a amostragem: o topo do rank domina menos."""
    def top_count(alpha):
        buf = PrioritizedReplayBuffer(500)
        obs = np.zeros((5, 20, 32), dtype=np.float32)
        for i in range(500):
            buf.add(obs, 0, 1.0, obs, False)
        buf.update_priorities([0], [1000.0])  # item 0 vira rank 1
        np.random.seed(4)
        _, _, _, _, _, indices, _ = buf.sample(500, alpha=alpha)
        return int((indices == 0).sum())

    assert top_count(0.5) < top_count(1.0) / 2


def test_ddqn_beta_anneals_to_one():
    """β sobe linearmente de beta_start até 1 ao longo de beta_steps."""
    agent = DoubleDQNPerDuelingAgent(
        (5, 20, 32), 6, beta_start=0.4, beta_steps=100)
    assert agent._beta() == pytest.approx(0.4)
    agent.steps = 50
    assert agent._beta() == pytest.approx(0.7)
    agent.steps = 100
    assert agent._beta() == pytest.approx(1.0)
    agent.steps = 250
    assert agent._beta() == pytest.approx(1.0)


def test_agents_use_the_expected_replay_buffers():
    plain = DQNAgent((5, 20, 32), 6)
    advanced = DoubleDQNPerDuelingAgent((5, 20, 32), 6)
    assert isinstance(plain.buffer, ReplayBuffer)
    assert not isinstance(plain.buffer, PrioritizedReplayBuffer)
    assert isinstance(advanced.buffer, PrioritizedReplayBuffer)


@pytest.mark.parametrize("agent_cls", [DQNAgent, DoubleDQNPerDuelingAgent])
def test_hard_target_sync_every_c(agent_cls):
    """O alvo fixo copia a online a cada C learn() bem-sucedidos."""
    agent = agent_cls((5, 20, 32), 6, batch_size=2)
    obs = np.zeros((5, 20, 32), dtype=np.float32)
    for action in (0, 1):
        agent.buffer.add(obs, action, 1.0, obs, False)
    with torch.no_grad():
        for p in agent.online.parameters():
            p.fill_(1.0)
        for p in agent.target.parameters():
            p.fill_(0.0)
    agent.steps = agent.C - 1
    assert agent.learn() is not None
    assert agent.steps == agent.C
    for param_t, param_o in zip(agent.target.parameters(), agent.online.parameters()):
        assert torch.equal(param_t.data, param_o.data)


def _tiny_mdp_obs(state):
    """Observação (5, 3, 3) com um pixel aceso no canal do estado."""
    obs = np.zeros((5, 3, 3), dtype=np.float32)
    obs[state, 0, 0] = 1.0
    return obs


@pytest.mark.parametrize("agent_cls", [DQNAgent, DoubleDQNPerDuelingAgent])
def test_offpolicy_agents_learn_two_state_mdp(agent_cls):
    """O laço completo (buffer, alvo, perda, otimizador) aprende um MDP trivial.

    No estado 0 a ação 0 dá +1; no estado 1 a ação 1 dá +1. É um teste de
    fumaça do aprendizado, não de desempenho na arena: pega regressões em que
    o agente para de aprender até uma tarefa de dois estados.
    """
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    agent = agent_cls((5, 3, 3), 2, lr=1e-3, gamma=.9, epsilon=1.0,
                      epsilon_min=.05, epsilon_decay=.995, batch_size=32,
                      buffer_size=2000)
    agent.C = 10
    for i in range(800):
        state = i % 2
        obs = _tiny_mdp_obs(state)
        action = agent.act(obs)
        reward = 1.0 if action == state else 0.0
        agent.buffer.add(obs, action, reward,
                         np.zeros_like(obs), True)
        agent.learn()
    assert agent.act(_tiny_mdp_obs(0), greedy=True) == 0
    assert agent.act(_tiny_mdp_obs(1), greedy=True) == 1


@pytest.mark.parametrize("agent_cls", [DQNAgent, DoubleDQNPerDuelingAgent])
def test_warmup_delays_learning(agent_cls):
    """`learn()` não atualiza enquanto o buffer não alcança `warmup`.

    Sem warmup, o primeiro update acontece com `batch_size` transições (64),
    todas do início do episódio e altamente correlacionadas.
    """
    agent = agent_cls((5, 20, 32), 6, warmup=100, batch_size=32)
    obs = np.zeros((5, 20, 32), dtype=np.float32)
    for _ in range(99):
        agent.buffer.add(obs, 0, 0.0, obs, False)
    assert len(agent.buffer) >= agent.batch_size
    assert agent.learn() is None
    assert agent.steps == 0
    agent.buffer.add(obs, 0, 1.0, obs, False)
    assert agent.learn() is not None
    assert agent.steps == 1


def test_epsilon_for_episode_ramps_linearly_to_floor():
    """Rampa de 1 → piso ao longo de `ramp_end` episódios (número absoluto)."""
    assert epsilon_for_episode(1, 60, 0.1) == pytest.approx(1.0 - 0.9 / 60)
    assert epsilon_for_episode(30, 60, 0.1) == pytest.approx(0.55)
    assert epsilon_for_episode(60, 60, 0.1) == pytest.approx(0.1)
    assert epsilon_for_episode(61, 60, 0.1) == pytest.approx(0.1)


class _DummyAgent:
    def __init__(self, on_policy=False, epsilon=1.0):
        self.on_policy = on_policy
        self.epsilon = epsilon
        self.epsilon_decay = 0.9995


def test_schedule_epsilon_applies_ramp_to_offpolicy_agents():
    agent = _DummyAgent()
    schedule_epsilon(agent, ep=15, ramp_episodes=60, floor=0.1)
    assert agent.epsilon == pytest.approx(1.0 - 0.9 * 15 / 60)
    # A rampa substitui o decay por passo: dentro do episódio ε não muda.
    assert agent.epsilon_decay == 1.0


def test_schedule_epsilon_leaves_onpolicy_agents_untouched():
    agent = _DummyAgent(on_policy=True, epsilon=0.0)
    schedule_epsilon(agent, ep=15, ramp_episodes=60, floor=0.1)
    assert agent.epsilon == 0.0


def test_parse_args_defaults_epsilon_ramp():
    """A rampa é ancorada em episódios absolutos, não em `--episodes`."""
    args = parse_args([])
    assert args.epsilon_ramp_episodes == 150
    assert args.epsilon_min == 0.1
    assert args.warmup == 1000


@pytest.mark.parametrize("agent_cls", [DQNAgent, DoubleDQNPerDuelingAgent])
def test_target_is_not_synced_before_c(agent_cls):
    agent = agent_cls((5, 20, 32), 6, batch_size=2)
    obs = np.zeros((5, 20, 32), dtype=np.float32)
    for action in (0, 1):
        agent.buffer.add(obs, action, 1.0, obs, False)
    with torch.no_grad():
        for p in agent.online.parameters():
            p.fill_(1.0)
        for p in agent.target.parameters():
            p.fill_(0.0)
    assert agent.learn() is not None
    assert agent.steps == 1
    for param_t in agent.target.parameters():
        assert torch.all(param_t.data == 0.0)


def test_advanced_learn_updates_priorities():
    np.random.seed(0)
    torch.manual_seed(0)
    agent = DoubleDQNPerDuelingAgent((5, 20, 32), 6, batch_size=2)
    obs = np.zeros((5, 20, 32), dtype=np.float32)
    for action in (0, 1):
        agent.buffer.add(obs, action, 1.0, obs, False)
    loss = agent.learn()
    assert loss is not None
    assert agent._last_indices is not None and len(agent._last_indices) == 2
    assert all(p != pytest.approx(1.0) for p in agent.buffer.priorities)


@pytest.mark.parametrize("agent_cls", [DQNAgent, DoubleDQNPerDuelingAgent])
def test_learn_smoke_and_grad_norm(agent_cls):
    agent = agent_cls((5, 20, 32), 6, batch_size=2)
    obs = np.zeros((5, 20, 32), dtype=np.float32)
    for action in (0, 1):
        agent.buffer.add(obs, action, 1.0, obs, False)
    assert agent.learn() is not None
    assert agent.last_grad_norm >= 0.0


@pytest.mark.parametrize("agent_cls", [DQNAgent, DoubleDQNPerDuelingAgent])
def test_act_batch_greedy_matches_act(agent_cls):
    agent = agent_cls((5, 20, 32), 6, epsilon=1.0)
    state = np.zeros((5, 20, 32), dtype=np.float32)
    assert agent.act(state, greedy=True) == agent.act_batch(
        np.stack([state]), greedy=True)[0]
    acts = agent.act_batch(np.stack([state, state]), greedy=False)
    assert acts.shape == (2,)
    assert np.issubdtype(acts.dtype, np.integer)


def test_vector_env_and_act_batch():
    """VectorEnv síncrono (sem subprocessos) e ações em lote com shapes reais."""
    venv = make_vector_env(2, ArenaConfig(), "random", asynchronous=False)
    try:
        obs, _ = venv.reset(seed=7)
        assert obs.shape == (2, 5, 20, 32)
        agent = DQNAgent((5, 20, 32), 6)
        acts = agent.act_batch(obs, greedy=True)
        assert acts.shape == (2,)
        assert np.issubdtype(acts.dtype, np.integer)
        obs2, rew, term, trunc, info = venv.step(acts)
        assert obs2.shape == (2, 5, 20, 32)
        assert rew.shape == (2,)
    finally:
        venv.close()


def test_obs_to_image_uses_last_frames():
    obs = np.zeros((20, 20, 32), dtype=np.float32)
    obs[:5, 0, 0] = 1.0
    obs[-5:, 1, 1] = 1.0
    img = obs_to_image(obs)
    assert np.array_equal(img, obs_to_image(obs[-5:]))


def test_train_cli_alg_flag():
    """--alg aceita os sufixos dos pacotes alg_* e default é dqn."""
    assert parse_args([]).alg == "dqn"
    assert parse_args(["--alg", "doubledqn_per_dueling"]).alg == "doubledqn_per_dueling"
    with pytest.raises(SystemExit):
        parse_args(["--alg", "inexistente"])


def test_train_help_lists_algorithm_and_grid_flags(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--help"])
    out = capsys.readouterr().out
    for flag in ("--alg", "--lr", "--batch-size"):
        assert flag in out
    # As flags que agora definem o algoritmo saíram da CLI.
    for flag in ("--per", "--dueling", "--no-double", "--n-step",
                 "--tau", "--target-update"):
        assert flag not in out


def test_train_cli_lr_and_batch():
    defaults = parse_args([])
    assert (defaults.lr, defaults.batch_size) == (1e-3, 64)
    args = parse_args(["--lr", "2.5e-4", "--batch-size", "128"])
    assert args.lr == 2.5e-4
    assert args.batch_size == 128


def test_tensorboard_run_name_uses_algorithm_and_configs():
    """O run do TensorBoard identifica algoritmo e configurações do treino."""
    args = parse_args([
        "--alg", "ppo", "--opponent", "rule", "--frame-stack", "4",
        "--frame-skip", "3", "--lr", "2.5e-4", "--batch-size", "64",
        "--seed", "7",
    ])
    assert tensorboard_run_name(args) == (
        "ppo_configs_rule_fs4_fk3_lr0.00025_b64_seed7")


def test_tensorboard_run_name_accepts_override():
    args = parse_args(["--run-name", "meu_experimento"])
    assert tensorboard_run_name(args) == "meu_experimento"


def test_parse_args_accepts_time_budget():
    args = parse_args(["--max-hours", "3"])
    assert args.max_hours == 3.0


def test_time_budget_reached_uses_elapsed_wall_clock():
    """O orçamento é wall-clock: 3 h a partir do início, não episódios."""
    start = 100.0
    budget_seconds = 3 * 3600
    assert not time_budget_reached(start, 3.0, now=start + budget_seconds - 1)
    assert time_budget_reached(start, 3.0, now=start + budget_seconds)
    assert not time_budget_reached(start, None, now=start + 10 * budget_seconds)


def test_validate_train_args_rejects_non_positive_time_budget():
    args = argparse.Namespace(envs=1, alg="dqn", max_hours=0.0)
    with pytest.raises(SystemExit):
        validate_train_args(args)


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
    validate_train_args(parse_args(["--envs", "4", "--opponent", "random"]))


def test_make_vector_env_validates_args():
    with pytest.raises(ValueError):
        make_vector_env(0, ArenaConfig(), "random", asynchronous=False)
    with pytest.raises(ValueError):
        make_vector_env(1, ArenaConfig(), object(), asynchronous=False)


def test_checkpoint_policy_uses_stacked_obs():
    """Com frame stack, o adversário checkpoint age sobre a obs empilhada."""
    agent = DQNAgent((20, 20, 32), 6)
    env = ArenaEnv(ArenaConfig(frame_stack=4), opponent_policy=CheckpointPolicy(agent))
    obs, _ = env.reset(seed=1)
    assert obs.shape == (20, 20, 32)
    next_obs, *_ = env.step(1)
    assert next_obs.shape == (20, 20, 32)
    env.close()
