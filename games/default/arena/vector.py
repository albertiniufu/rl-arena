"""Fábrica de `VectorEnv` para o treino vetorizado (`train.py --envs N`).

O adversarial é fixo por nome de política: o currículo do `--opponent auto`
escolhe a política por episódio e não mapeia em workers concorrentes.
"""
import gymnasium as gym

from .environment import ArenaEnv


def make_vector_env(n_envs, cfg, opponent, asynchronous=True):
    """VectorEnv com `n_envs` cópias independentes de `ArenaEnv`.

    `opponent` é o NOME de uma política de `agents.baseline` (str); cada worker
    constrói a própria `ArenaEnv(cfg, opponent_policy=opponent)`, então o
    estado de jogo não é compartilhado. Com `asynchronous=True` usa-se
    `AsyncVectorEnv` (um processo por worker, caminho do treino); com
    `asynchronous=False`, `SyncVectorEnv` (sem subprocessos, determinístico,
    usado nos testes). O `reset(seed=N)` do VectorEnv semeia o worker `i`
    com `N + i`.
    """
    n_envs = int(n_envs)
    if n_envs < 1:
        raise ValueError(f"n_envs deve ser >= 1 (recebido {n_envs})")
    if not isinstance(opponent, str):
        raise ValueError(
            "opponent deve ser o nome de uma política (str), recebido "
            f"{type(opponent).__name__}")
    # Cada factory cria a própria env. O lambda captura só cfg/opponent (fixos);
    # a variável do loop não entra no fechamento, então todas são equivalentes.
    env_fns = [
        (lambda: ArenaEnv(cfg, opponent_policy=opponent))
        for _ in range(n_envs)
    ]
    vector_cls = gym.vector.AsyncVectorEnv if asynchronous else gym.vector.SyncVectorEnv
    return vector_cls(env_fns)
