import os
import random

import numpy as np
import pytest
import torch

# Vídeo/áudio dummy do SDL: os testes que tocam pygame rodam headless,
# sem abrir janela (e sem depender de um display na máquina).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from agents.baseline import CoinRulePolicy, PolicyContext, PushRulePolicy, RandomPolicy, make_policy
from alg_dqn.agent import DQNAgent
from alg_registry import (
    CheckpointPolicy,
    load_algorithm,
    load_checkpoint,
    save_checkpoint,
)
from arena.actions import (
    JUMP,
    LEFT,
    N_ACTIONS,
    NON_PUSH_ACTIONS,
    PUSH_INTENSITY,
    PUSH_LEFT,
    PUSH_RIGHT,
    RIGHT,
    STOP,
)
from arena.environment import ArenaEnv
from arena.config import ArenaConfig
from arena.vector import make_vector_env

def test_reset_step_observation():
    env = ArenaEnv(ArenaConfig())
    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    obs, reward, term, trunc, info = env.step(0)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    env.close()

def test_action_space_has_six_actions():
    """Uma única intensidade de pulo e de empurrão por direção."""
    env = ArenaEnv()
    assert env.action_space.n == N_ACTIONS == 6
    assert (STOP, LEFT, RIGHT, JUMP, PUSH_LEFT, PUSH_RIGHT) == (0, 1, 2, 3, 4, 5)
    env.close()

def test_push_actions_use_max_intensity():
    """PUSH_LEFT/PUSH_RIGHT chamam o engine com a intensidade máxima."""
    env = ArenaEnv(ArenaConfig(opponent_count=0, frame_skip=1))
    env.reset(seed=1)
    calls = []
    env.engine.push = lambda actor, direction, intensity: calls.append(
        (actor, direction, intensity)) or True
    env.step(PUSH_RIGHT)
    env.step(PUSH_LEFT)
    assert calls == [(0, 1, PUSH_INTENSITY), (0, -1, PUSH_INTENSITY)]
    env.close()

def test_single_jump_action_leaves_ground():
    env = ArenaEnv(ArenaConfig(opponent_count=0))
    env.reset(seed=1)
    p = env.engine.characters[0]
    p.x, p.y, p.vx, p.vy, p.on_ground = 100.0, 360.0, 0.0, 0.0, True
    env.step(JUMP)
    assert p.vy < 0 and not p.on_ground
    env.close()

def test_vector_env_same_seed_reproducible_across_autoreset():
    """Mesmo seed => corridas idênticas mesmo atravessando autoresets.

    O reset inicial do VectorEnv semeia o worker `i` com `seed + i`, mas o
    autoreset (NEXT_STEP) chama `reset()` sem seed. O engine precisa continuar
    o RNG existente nesse caso; se ele fosse re-semeado com entropia, o
    segundo episódio de cada worker divergiria entre duas corridas iguais.
    """
    cfg = ArenaConfig(episode_seconds=0.2)  # ~7 steps por episódio (fps=30)

    def run():
        venv = make_vector_env(2, cfg, "random", asynchronous=False)
        try:
            obs, _ = venv.reset(seed=7)
            # A cópia é necessária: o passo seguinte pode sobrescrever o array.
            obs_trace, rew_trace = [obs.copy()], []
            truncations = np.zeros(2, dtype=int)  # uma por worker
            # Dois episódios por worker: cada worker passa por 2 autoresets.
            for _ in range(200):
                if truncations.sum() >= 4:
                    break
                obs, rew, term, trunc, _ = venv.step(np.zeros(2, dtype=np.int64))
                obs_trace.append(obs.copy())
                rew_trace.append(rew.copy())
                truncations += trunc.astype(int)
            assert truncations.sum() >= 4, "os workers não truncaram o esperado"
            return np.stack(obs_trace), np.stack(rew_trace)
        finally:
            venv.close()

    obs_a, rew_a = run()
    obs_b, rew_b = run()
    assert np.array_equal(obs_a, obs_b)
    assert np.array_equal(rew_a, rew_b)


def test_frame_stack_observation():
    env = ArenaEnv(ArenaConfig(frame_stack=4))
    obs, _ = env.reset(seed=1)
    assert obs.shape == (20, 20, 32)  # (5 * frame_stack, H, W): mapa 20 linhas x 32 colunas
    assert np.array_equal(obs[:5], obs[5:10]) and np.array_equal(obs[15:], obs[:5])  # pilha cheia de cópias
    obs2, *_ = env.step(1)
    assert np.array_equal(obs2[:5], obs[-5:])  # frame antigo virou o mais antigo
    env.close()

def test_frame_skip_repeats_action():
    env = ArenaEnv(ArenaConfig(frame_skip=3))
    env.reset(seed=1)
    t0 = env.elapsed
    # Moeda posta na posição do personagem antes do step (padrão dos testes:
    # manipular o engine diretamente): a coleta deve acontecer nos substeps.
    c = env.engine.characters[0]
    env.engine.coins[0].x, env.engine.coins[0].y = c.x, c.y
    obs, reward, *_ = env.step(1)
    assert env.elapsed - t0 == pytest.approx(3 / env.cfg.fps)
    assert reward == pytest.approx(env.cfg.coin_reward)
    env.close()

def test_stop_action_penalized_once_per_agent_step():
    """STOP custa -0.1 por step do agente, sem multiplicar pelo frame_skip."""
    env = ArenaEnv(ArenaConfig(opponent_count=0, frame_skip=3))
    env.reset(seed=1)
    env.engine.coins.clear()
    _, reward, *_ = env.step(STOP)
    assert reward == pytest.approx(-0.1)
    env.close()


def test_non_stop_action_has_no_stop_penalty():
    """Andar (ou qualquer ação ativa) não paga a penalidade de ficar parado."""
    env = ArenaEnv(ArenaConfig(opponent_count=0, frame_skip=3))
    env.reset(seed=1)
    env.engine.coins.clear()
    _, reward, *_ = env.step(LEFT)
    assert reward == pytest.approx(0.0)
    env.close()


def test_push_without_opponent_penalized_once_per_agent_step():
    """Empurrar sem oponente em contato custa -0.1, uma vez por step."""
    env = ArenaEnv(ArenaConfig(opponent_count=1, frame_skip=3))
    env.reset(seed=1)
    env.engine.coins.clear()
    # Oponente existe, mas está longe: o empurrão não tem alvo.
    env.engine.characters[1].x = 600.0
    _, reward, *_ = env.step(PUSH_LEFT)
    assert reward == pytest.approx(-0.1)
    env.close()


def test_push_near_opponent_has_no_miss_penalty():
    """Empurrar com oponente em contato não paga a penalidade de erro."""
    env = ArenaEnv(ArenaConfig(opponent_count=1, frame_skip=1))
    env.reset(seed=1)
    env.engine.coins.clear()
    player, opponent = env.engine.characters
    player.x, player.y, player.vx, player.vy = 400.0, 360.0, 0.0, 0.0
    player.on_ground = True
    opponent.x, opponent.y, opponent.vx, opponent.vy = 424.0, 360.0, 0.0, 0.0
    opponent.on_ground = True
    _, reward, *_ = env.step(PUSH_RIGHT)
    assert opponent.vx > 0  # o empurrão encontrou o alvo
    assert reward == pytest.approx(0.0)
    env.close()


def test_invalid_action():
    env = ArenaEnv()
    env.reset(seed=1)
    with pytest.raises(ValueError): env.step(N_ACTIONS)
    env.close()

def test_cnn_and_checkpoint(tmp_path):
    env = ArenaEnv()
    obs, _ = env.reset(seed=1)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n, batch_size=2)
    assert agent.online(torch.as_tensor(obs).unsqueeze(0)).shape == (1, N_ACTIONS)
    p = tmp_path / "test.pt"
    save_checkpoint(p, agent)
    agent2 = DQNAgent(env.observation_space.shape, env.action_space.n)
    load_checkpoint(p, agent2, resume=True)
    env.close()

def test_one_episode_without_render():
    env = ArenaEnv(ArenaConfig(episode_seconds=.1))
    obs, _ = env.reset(seed=2)
    while True:
        obs, _, term, trunc, info = env.step(STOP)
        if term or trunc: break
    assert trunc
    env.close()


def test_character_and_events_have_no_elimination_concept():
    """Eliminações deixaram de existir: nem o personagem nem os eventos as expõem."""
    from arena.engine import Character
    assert "eliminations" not in Character.__dataclass_fields__
    env = ArenaEnv(ArenaConfig(opponent_count=1, frame_skip=1))
    env.reset(seed=1)
    event = env.engine.step([STOP, STOP], 1 / env.cfg.fps)
    assert "eliminations_delta" not in event
    env.close()


def test_info_has_no_eliminations():
    env = ArenaEnv(ArenaConfig(opponent_count=0, frame_skip=1))
    env.reset(seed=1)
    _, _, _, _, info = env.step(STOP)
    assert "eliminations" not in info
    env.close()


def test_opponent_death_gives_no_elimination_bonus():
    """Morte do oponente não rende recompensa extra ao jogador."""
    env = ArenaEnv(ArenaConfig(opponent_count=1, frame_skip=1))
    env.reset(seed=1)
    env.engine.coins.clear()
    opponent = env.engine.characters[1]
    opponent.y = env.cfg.height + 50
    _, reward, *_ = env.step(STOP)
    assert not opponent.alive
    assert reward == pytest.approx(env.cfg.stop_penalty)
    env.close()


def test_config_has_no_elimination_reward():
    assert not hasattr(ArenaConfig(), "elimination_reward")


def test_death_penalty_and_random_platform_respawn():
    cfg = ArenaConfig(respawn_seconds=0.1)
    env = ArenaEnv(cfg)
    env.reset(seed=4)
    player = env.engine.characters[0]
    player.coins = 6
    player.y = cfg.height + 50
    # One step advances physics and triggers the fatal-region logic.
    env.step(STOP)
    assert player.coins == 0
    assert player.deaths == 1
    assert not player.alive
    # Advance beyond the respawn timer; the character should return to a platform.
    for _ in range(10):
        env.step(STOP)
    assert player.alive
    assert any(abs(player.y - py) < 1e-5 for _, py, _, _ in env.engine.platforms)
    env.close()


def test_jump_reaches_first_platform_tier():
    # O maior degrau entre plataformas adjacentes é 75px (chão 360 → plataforma 285);
    # o pulo precisa alcançá-lo para o agente conseguir subir na arena.
    env = ArenaEnv(ArenaConfig(opponent_count=0))
    env.reset(seed=1)
    p = env.engine.characters[0]
    p.x, p.y, p.vx, p.vy, p.on_ground = 100.0, 360.0, 0.0, 0.0, True
    env.step(JUMP)
    for _ in range(90):
        env.step(JUMP)
        if p.on_ground and p.y < 360:
            break
    assert p.y == 285.0
    env.close()


def test_human_policy_returns_valid_action_headless():
    from agents.human import HumanPolicy
    policy = HumanPolicy()
    ctx = PolicyContext(obs=np.zeros((5, 2, 2), dtype=np.float32), rng=np.random.default_rng(0))
    # Sem janela/teclas pressionadas, deve devolver uma ação válida (parado).
    assert 0 <= policy.act(ctx) < N_ACTIONS


def test_dino_renderer_does_not_touch_global_random():
    # Skins não podem deslocar o fluxo global de `random`: a exploração e a
    # amostragem do buffer dependem dele, então treinar com --render deve
    # produzir a mesma trajetória que sem janela.
    from arena.renderer_dino import ArenaRenderer as DinoRenderer
    env = ArenaEnv(ArenaConfig(opponent_count=0))
    env.reset(seed=1)
    renderer = DinoRenderer(env.cfg)
    before = random.getstate()
    renderer.render(env.engine, 0.1, {})
    assert random.getstate() == before
    renderer.close()
    env.close()


def test_skins_discovery_and_loading():
    from arena.renderer import ArenaRenderer as BaseRenderer
    from arena.skins import list_skins, load_renderer_class
    skins = list_skins()
    assert "neon" in skins and "plain" in skins
    # Skin None = renderer clássico (arena/renderer.py).
    assert load_renderer_class(None) is BaseRenderer
    neon = load_renderer_class("neon")
    assert neon.__name__ == "ArenaRenderer"
    assert hasattr(neon, "render") and hasattr(neon, "close")
    with pytest.raises(ValueError):
        load_renderer_class("inexistente")


def test_env_skin_selection():
    from unittest.mock import patch
    env = ArenaEnv(skin="neon")
    with patch("arena.environment.load_renderer_class") as loader:
        env.render()
        loader.assert_called_once_with("neon")
    env.close()
    env = ArenaEnv()
    with patch("arena.environment.load_renderer_class") as loader:
        env.render()
        loader.assert_called_once_with(None)
    env.close()


def test_make_policy_registry():
    assert isinstance(make_policy("rule"), PushRulePolicy)
    assert isinstance(make_policy("random"), RandomPolicy)
    assert isinstance(make_policy("coins"), CoinRulePolicy)
    ready = RandomPolicy(n_actions=3)
    assert make_policy(ready) is ready
    with pytest.raises(ValueError):
        make_policy("inexistente")


def test_random_opponent_keeps_historic_range():
    # O adversário "random" histórico não empurrava; o espaço novo tem os
    # empurrões nas ações 4 e 5, então o sorteio fica nas 4 primeiras.
    env = ArenaEnv(opponent_policy="random")
    env.reset(seed=1)
    ctx = PolicyContext(obs=np.zeros(env.observation_space.shape, dtype=np.float32), rng=env.np_random)
    for _ in range(100):
        assert 0 <= env.opponent_policy.act(ctx) < len(NON_PUSH_ACTIONS)
    env.close()


def test_all_policies_return_valid_actions():
    """Nenhuma política devolve ação fora do espaço de 6 ações."""
    env = ArenaEnv(ArenaConfig(opponent_count=1))
    env.reset(seed=3)
    env.step(STOP)
    ctx = PolicyContext(
        obs=env.observation_space.sample() if hasattr(env.observation_space, "sample") else np.zeros(env.observation_space.shape, dtype=np.float32),
        rng=env.np_random,
        self_character=env.engine.characters[1],
        player_character=env.engine.characters[0],
        stacked_obs=None,
    )
    for name in ("rule", "random", "coins"):
        policy = make_policy(name)
        for _ in range(50):
            assert 0 <= policy.act(ctx) < N_ACTIONS
    env.close()


def test_coins_opponent_runs_full_episode():
    env = ArenaEnv(ArenaConfig(episode_seconds=0.1), opponent_policy="coins")
    env.reset(seed=2)
    while True:
        _, _, term, trunc, _ = env.step(STOP)
        if term or trunc:
            break
    assert trunc
    env.close()


def test_resolve_resume_path(tmp_path):
    from train import resolve_resume_path
    assert resolve_resume_path("latest.pt", tmp_path) == tmp_path / "latest.pt"
    other = tmp_path / "outra" / "run.pt"
    assert resolve_resume_path(str(other), "checkpoints") == other


def test_obs_to_image_colors():
    from train import obs_to_image
    obs = np.zeros((5, 3, 4), dtype=np.float32)
    obs[0, 0, 0] = 1.0  # plataforma
    obs[1, 1, 1] = 1.0  # jogador
    obs[2, 2, 2] = 1.0  # oponente
    obs[3, 0, 3] = 1.0  # moeda
    obs[4, 2, 3] = 1.0  # morto/respawn
    img = obs_to_image(obs)
    assert img.shape == (3, 4, 3)
    assert img.dtype == np.uint8
    assert img[0, 0].tolist() == [120, 130, 140]
    assert img[1, 1].tolist() == [70, 190, 250]
    assert img[2, 2].tolist() == [240, 90, 95]
    assert img[0, 3].tolist() == [245, 195, 55]
    assert img[2, 3].tolist() == [180, 120, 200]


def test_tensorboard_writer_smoke(tmp_path):
    pytest.importorskip("torch.utils.tensorboard")
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(tmp_path))
    writer.add_scalar("train/loss", 1.0, 1)
    writer.add_scalar("train/loss", 0.5, 2)
    writer.close()
    assert any(tmp_path.rglob("events.out.tfevents*"))


def test_play_recording_produces_video(tmp_path):
    pytest.importorskip("imageio")
    pytest.importorskip("imageio_ffmpeg")
    import imageio
    from play import record_frame
    # SDL dummy (definido no topo do módulo) permite renderizar headless.
    env = ArenaEnv(ArenaConfig(episode_seconds=0.1), render_mode="human")
    env.reset(seed=1)
    out = tmp_path / "video.mp4"
    writer = imageio.get_writer(str(out), fps=env.cfg.fps)
    for _ in range(3):
        env.step(STOP)
        record_frame(writer)
    writer.close()
    env.close()
    assert out.stat().st_size > 1000
    reader = imageio.get_reader(str(out))
    meta = reader.get_meta_data()
    assert meta["fps"] == pytest.approx(env.cfg.fps)
    assert reader.count_frames() == 3
    # H.264: o formato aceito pelo WhatsApp (e por qualquer player).
    assert meta["codec"].startswith("h264")


def test_checkpoint_stores_episodes(tmp_path):
    env = ArenaEnv()
    env.reset(seed=1)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n)
    p = tmp_path / "ep.pt"
    save_checkpoint(p, agent, episodes=42)
    payload = torch.load(p, map_location="cpu", weights_only=False)
    assert payload["episodes"] == 42
    # Sem informar, o campo existe mas fica None (compatível com checkpoints antigos).
    p2 = tmp_path / "sem.pt"
    save_checkpoint(p2, agent)
    payload2 = torch.load(p2, map_location="cpu", weights_only=False)
    assert payload2["episodes"] is None
    env.close()


def test_checkpoint_policy_acts_greedily():
    env = ArenaEnv()
    obs, _ = env.reset(seed=1)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n)
    cp = CheckpointPolicy(agent)
    ctx = PolicyContext(obs=obs, rng=env.np_random)
    assert cp.act(ctx) == agent.act(obs, greedy=True)
    env.close()


def test_agent_q_stats_and_grad_norm():
    env = ArenaEnv()
    obs, _ = env.reset(seed=1)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n, batch_size=2)
    mean_q, max_q = agent.q_stats(obs)
    assert isinstance(mean_q, float) and isinstance(max_q, float)
    assert max_q >= mean_q
    # learn() preenche last_grad_norm (norma do gradiente antes do clip).
    agent.last_grad_norm = -1.0
    for _ in range(2):
        action = agent.act(obs)
        next_obs, reward, term, trunc, _ = env.step(action)
        agent.buffer.add(obs, action, reward, next_obs, term)
        obs = next_obs
    loss = agent.learn()
    assert loss is not None
    assert agent.last_grad_norm >= 0.0
    env.close()


def _make_auto_opponent(tmp_path):
    from train import AutoOpponent
    env = ArenaEnv(ArenaConfig(), opponent_policy="random")
    env.reset(seed=1)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n)
    auto = AutoOpponent(env, load_algorithm("dqn"), checkpoint_dir=str(tmp_path),
                        random_episodes=2, baseline_episodes=3, alt_prob=0.0)
    return env, agent, auto


def test_auto_opponent_phases(tmp_path):
    env, agent, auto = _make_auto_opponent(tmp_path)
    # Fase 1 (n <= 2): random.
    assert isinstance(auto.choose(1), RandomPolicy)
    assert isinstance(auto.choose(2), RandomPolicy)
    # Fase 2 (3..5): alterna coins/rule.
    assert {type(auto.choose(3)), type(auto.choose(4))} == {CoinRulePolicy, PushRulePolicy}
    # update_best congela os pesos atuais do treino como oponente.
    auto.update_best(agent)
    for p_c, p_a in zip(auto.checkpoint_agent.online.parameters(), agent.online.parameters()):
        assert torch.equal(p_c, p_a)
    # Fase 3 (6+): melhor checkpoint até o momento.
    assert isinstance(auto.choose(6), CheckpointPolicy)
    env.close()


def test_auto_opponent_alternates_in_checkpoint_phase(tmp_path):
    env, agent, auto = _make_auto_opponent(tmp_path)
    auto.alt_prob = 1.0
    save_checkpoint(tmp_path / "outro.pt", agent)
    auto.update_best(agent)
    random.seed(0)
    seen = {type(auto.choose(6)) for _ in range(40)}
    assert seen == {CheckpointPolicy, RandomPolicy, PushRulePolicy, CoinRulePolicy}
    env.close()


def test_auto_opponent_init_from_disk(tmp_path):
    from train import AutoOpponent
    env = ArenaEnv()
    env.reset(seed=1)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n)
    save_checkpoint(tmp_path / "best.pt", agent)
    auto = AutoOpponent(env, load_algorithm("dqn"), checkpoint_dir=str(tmp_path))
    auto.init_from_disk()
    for p_c, p_a in zip(auto.checkpoint_agent.online.parameters(), agent.online.parameters()):
        assert torch.equal(p_c, p_a)
    env.close()


def _run_episodes(cfg, seed, n_episodes, save_at=None, path=None, resume_from=None):
    """Mini-loop espelhando train.py, para testar retomada com checkpoint pesos-only."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = ArenaEnv(cfg, opponent_policy="rule")
    agent = DQNAgent(env.observation_space.shape, env.action_space.n, batch_size=8)
    start, best = 1, -float("inf")
    if resume_from is not None:
        payload = load_checkpoint(resume_from, agent, resume=True)
        episodes_done = payload.get("episodes")
        if episodes_done is None:
            episodes_done = payload.get("extra", {}).get("episodes_done", 0)
        start = int(episodes_done) + 1
        best = float(payload.get("extra", {}).get("best_reward", -float("inf")))
    trace = []
    for ep in range(start, n_episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        total, steps = 0.0, 0
        while True:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.buffer.add(obs, action, reward, next_obs, terminated)
            agent.learn()
            obs = next_obs
            total += reward
            steps += 1
            if terminated or truncated:
                break
        trace.append((ep, total, steps, agent.epsilon, agent.steps))
        if total > best:
            best = total
        if ep == save_at:
            save_checkpoint(path, agent, cfg,
                            {"episode": ep, "reward": total, "episodes_done": ep, "best_reward": best},
                            episodes=ep)
            break
    return trace, agent, env


def test_resume_loads_weights_and_continues_episode_numbering(tmp_path):
    cfg = ArenaConfig(episode_seconds=0.5)
    seed = 123
    ckpt = tmp_path / "mid.pt"
    half, agent_half, env_half = _run_episodes(cfg, seed, 2, save_at=2, path=ckpt)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert payload["episodes"] == 2
    # O checkpoint pesos-only não persiste buffer, RNGs nem estado do ambiente.
    assert "buffer" not in payload and "env_state" not in payload and "rng_state" not in payload
    # Um agente novo carregado com resume=True recebe pesos, otimizador e contadores.
    agent_check = DQNAgent(env_half.observation_space.shape, env_half.action_space.n)
    load_checkpoint(ckpt, agent_check, resume=True)
    for p_c, p_h in zip(agent_check.online.parameters(), agent_half.online.parameters()):
        assert torch.equal(p_c, p_h)
    assert agent_check.steps == agent_half.steps
    assert agent_check.epsilon == agent_half.epsilon
    # A retomada continua a numeração de episódios (3 e 4) e o treino anda.
    resumed, agent_resumed, env_resumed = _run_episodes(cfg, seed, 4, resume_from=ckpt)
    assert [row[0] for row in resumed] == [3, 4]
    assert all(row[2] > 0 for row in resumed)
    assert agent_resumed.steps > agent_check.steps
    env_half.close()
    env_resumed.close()
