"""Helpers puros compartilhados pelos loops de treino do seek_coin."""
from __future__ import annotations

import time


def time_budget_reached(start, max_hours, now=None):
    """O orçamento wall-clock do treino terminou?

    `max_hours=None` desliga o orçamento. O relógio é monotônico e pode ser
    injetado (`now`) para os testes; o treino checa ao fim de cada episódio.
    """
    if max_hours is None:
        return False
    now = time.monotonic() if now is None else now
    return (now - start) >= max_hours * 3600.0


def _final_observation(infos, index):
    """Observação terminal do worker `index`, se o VectorEnv a expôs.

    O Gymnasium >= 1.0 expõe `final_obs` no modo SAME_STEP. No modo padrão
    (NEXT_STEP, usado por `make_vector_env`), a chave não existe e o
    `next_obs` do passo terminal já é o estado final.
    """
    array = infos.get("final_obs")
    if array is not None and array[index] is not None:
        return array[index]
    return None


def _worker_final_info(infos, index):
    """Info do episódio encerrado do worker `index`.

    Com autoreset NEXT_STEP (padrão do Gymnasium >= 1.0), o `info` do passo
    terminal já é o info final do episódio. As chaves internas de máscara
    (`_key`) não fazem parte do info do episódio.
    """
    return {key: value[index] for key, value in infos.items()
            if not str(key).startswith("_") and key != "final_info"
            and hasattr(value, "__getitem__")}


def _rollout_target(agent):
    """Transições por update: `rollout_steps` no PPO, `n_steps` no A2C."""
    target = getattr(agent, "rollout_steps", None) or getattr(agent, "n_steps", 1)
    return max(1, int(target))


def _append_rollout_segment(agent, segment):
    """Adiciona um trecho contíguo de um worker ao rollout on-policy.

    O buffer é uma lista plana; para o GAE/n-step não atravessar de um worker
    para outro, o último passo de cada trecho é marcado como fim de episódio.
    O alvo ainda faz bootstrap pelo `next_state` guardado, como num
    truncamento — só a propagação da vantagem é interrompida.
    """
    last = len(segment) - 1
    for k, (state, action, reward, next_state, terminated, log_prob, done) in \
            enumerate(segment):
        agent.buffer.add(state, action, reward, next_state, terminated,
                         log_prob)
        if done or k == last:
            agent.buffer.mark_episode_end()
