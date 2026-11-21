"""Coleta (CPU) sobreposta ao update (GPU) para PPO/A2C."""
from __future__ import annotations

import copy
import queue
import threading
import time

import numpy as np
import torch

from arenas.seek_coin.arena.vector import make_vector_env
from arenas.seek_coin.training.common import (
    _append_rollout_segment,
    _final_observation,
    _rollout_target,
    _worker_final_info,
    time_budget_reached,
)
from arenas.seek_coin.training.telemetry import (
    _finish_training,
    _record_episode,
    _training_csv,
)


@torch.no_grad()
def _act_batch_on_cpu(net, obs, generator):
    """Ações e log-probs de um lote, pela cópia CPU da política.

    O coletor roda numa thread separada e não toca na rede da GPU: assim a
    coleta (CPU) e o update (GPU) acontecem de fato ao mesmo tempo. O
    `generator` é exclusivo do coletor — o RNG default do torch é global
    entre threads, e usar um gerador próprio evita que a amostragem da
    coleta corrompa/consuma o RNG do update.
    """
    x = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
    logits, _ = net(x)
    probs = torch.softmax(logits, dim=1)
    actions = torch.multinomial(probs, 1, generator=generator).squeeze(1)
    log_probs = torch.log_softmax(logits, dim=1).gather(
        1, actions.unsqueeze(1)).squeeze(1)
    return actions.numpy(), log_probs.numpy()


def _train_on_policy_overlap(args, cfg, agent, ckpt_dir, start_ep, last_ep,
                             best, budgets, start_time):
    """Coleta (CPU) sobreposta ao update (GPU) para PPO/A2C.

    Uma thread coleta rollouts com uma cópia CPU da política e os entrega por
    uma fila de tamanho 1; a thread principal aplica o rollout ao buffer e faz
    o update na GPU. Enquanto o update roda, o coletor já está no próximo
    rollout, usando os pesos de antes do update — por isso a política fica
    defasada em uma atualização. O PPO compensa isso guardando o log-prob da
    ação no momento da coleta (`RolloutBuffer.log_probs_array`), então o ratio
    compara com a política de comportamento correta; no A2C o único update por
    rollout já é aproximadamente on-policy com a defasagem.

    A fila limitada a 1 garante que o coletor fique no máximo um rollout à
    frente: o ganho é a sobreposição, sem deixar a política envelhecer sem
    limite.
    """
    n_envs = args.envs
    chunk_steps = max(1, _rollout_target(agent) // n_envs)
    total_episodes = start_ep - 1
    last_total = 0.0
    fired_budgets = {b for b in budgets if b <= start_ep - 1}

    venv = make_vector_env(n_envs, cfg, args.opponent, asynchronous=True)
    # A rede de coleta vive na CPU; a rede de treino continua no device
    # escolhido (GPU quando disponível). `eval()` desliga dropout/BN (não há),
    # mas documenta que esta cópia nunca é treinada.
    rollout_net = copy.deepcopy(agent.online).cpu().eval()
    actor_net = copy.deepcopy(rollout_net)
    weights_lock = threading.Lock()
    payloads = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    # Gerador da amostragem da coleta, separado do RNG global do update.
    action_gen = torch.Generator().manual_seed(args.seed + 1)

    def collect():
        try:
            obs, _ = venv.reset(seed=args.seed)
            worker_rewards = np.zeros(n_envs, dtype=np.float64)
            worker_steps = np.zeros(n_envs, dtype=np.int64)
            worker_ep_start = np.full(n_envs, time.monotonic(),
                                      dtype=np.float64)
            pending_reset = np.zeros(n_envs, dtype=bool)
            total = start_ep - 1
            while not stop_event.is_set():
                # Pesos do início do rollout: uma sincronização do update
                # pode terminar no meio, mas só vale para o próximo rollout.
                with weights_lock:
                    rollout_net.load_state_dict(actor_net.state_dict())
                segments = [[] for _ in range(n_envs)]
                episodes = []
                reason = None
                for _ in range(chunk_steps):
                    actions, log_probs = _act_batch_on_cpu(
                        rollout_net, obs, action_gen)
                    next_obs, rewards, terminations, truncations, infos = \
                        venv.step(actions)
                    for i in range(n_envs):
                        if pending_reset[i]:
                            pending_reset[i] = False
                            worker_ep_start[i] = time.monotonic()
                            continue
                        done = bool(terminations[i]) or bool(truncations[i])
                        final_obs = _final_observation(infos, i) if done else None
                        next_state = final_obs if final_obs is not None else next_obs[i]
                        segments[i].append((
                            obs[i], int(actions[i]), float(rewards[i]),
                            next_state, bool(terminations[i]),
                            float(log_probs[i]), done))
                        worker_rewards[i] += float(rewards[i])
                        worker_steps[i] += 1
                        if not done:
                            continue
                        if final_obs is None:
                            pending_reset[i] = True
                        total += 1
                        info = _worker_final_info(infos, i)
                        episodes.append({
                            "episode": total,
                            "reward": float(worker_rewards[i]),
                            "steps": int(worker_steps[i]),
                            "coins": info.get("coins", 0),
                            "deaths": info.get("deaths", 0),
                            "seconds": time.monotonic() - worker_ep_start[i],
                        })
                        worker_rewards[i] = 0.0
                        worker_steps[i] = 0
                        worker_ep_start[i] = time.monotonic()
                    obs = next_obs
                    if total >= last_ep:
                        reason = "episodes"
                        break
                    if time_budget_reached(start_time, args.max_hours):
                        reason = "budget"
                        break
                payloads.put({"segments": segments, "episodes": episodes,
                              "total": total, "reason": reason})
                if reason is not None:
                    break
        finally:
            venv.close()

    thread = threading.Thread(target=collect, name="coleta", daemon=True)
    thread.start()
    last_loss = None
    with _training_csv(args.resume) as (f, writer):
        try:
            while True:
                payload = payloads.get()
                for i in range(n_envs):
                    _append_rollout_segment(agent, payload["segments"][i])
                loss = agent.learn()
                if loss is not None:
                    last_loss = float(loss)
                total_episodes = payload["total"]
                for rec in payload["episodes"]:
                    last_total = rec["reward"]
                    best = _record_episode(
                        f, writer, args, agent, cfg, ckpt_dir, budgets,
                        fired_budgets, best, rec["episode"], rec["reward"],
                        rec["steps"], rec["coins"], rec["deaths"],
                        last_loss if last_loss is not None else "",
                        rec["seconds"])
                # Publica os pesos pós-update para o próximo rollout.
                with weights_lock:
                    actor_net.load_state_dict(
                        {key: value.detach().cpu()
                         for key, value in agent.online.state_dict().items()})
                if payload["reason"] is not None:
                    if payload["reason"] == "budget":
                        print(f"orçamento de {args.max_hours:g} h atingido; "
                              f"encerrando após o episódio {total_episodes}")
                    break
        finally:
            stop_event.set()
            thread.join(timeout=5)
    _finish_training(agent, cfg, ckpt_dir, total_episodes, last_total, best)
