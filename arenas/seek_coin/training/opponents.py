"""Currículo de adversários e carregamento tolerante de checkpoints."""
from __future__ import annotations

import random
import warnings
from pathlib import Path

from arenas.algs.alg_registry import (
    CheckpointPolicy,
    checkpoint_meta,
    load_checkpoint,
)
from arenas.seek_coin.agents.baseline import make_policy


def _try_load_checkpoint(path, agent, frame_stack=None):
    """Carrega `path` em `agent`; devolve False (com aviso) se não der.

    Um `.pt` de outra arquitetura (ou corrompido) no diretório de checkpoints
    não pode quebrar o treino do adversário `auto`: o arquivo é pulado.
    Com `frame_stack` informado, checkpoints gravados com outro valor são
    pulados antes do load. O motivo não é só a mensagem de erro: como a
    primeira conv recebe `5 * frame_stack` canais, o `load_state_dict`
    estoura no meio e deixa os tensores anteriores já sobrescritos — pular
    antes evita esse carregamento parcial de pesos incompatíveis.
    """
    try:
        if frame_stack is not None:
            saved = int(checkpoint_meta(path)["config"].get("frame_stack", 1))
            if saved != frame_stack:
                warnings.warn(
                    f"Checkpoint ignorado ({path}): frame_stack={saved} difere do "
                    f"treino atual ({frame_stack})", UserWarning)
                return False
        load_checkpoint(path, agent, resume=False)
    except Exception as exc:  # arquitetura antiga, arquivo corrompido, etc.
        warnings.warn(f"Checkpoint ignorado ({path}): {exc}", UserWarning)
        return False
    return True


class AutoOpponent:
    """Currículo de adversários do `--opponent auto` (padrão).

    Por episódio global (1-based; independe de `--episodes` e de retomadas):
      - episódios 1..`random_episodes`: RandomPolicy
      - próximos `baseline_episodes`: alterna `coins` e `rule`
      - resto: joga contra o melhor checkpoint até o momento (guloso),
        com probabilidade `alt_prob` de alternar para `random`, `coins`,
        `rule` ou outro checkpoint disponível em `checkpoints/`.
    As fases usam contagens absolutas (não frações do total) justamente para
    a retomada reproduzível: o mesmo episódio global recebe a mesma política
    em qualquer segmentação da execução. A aleatoriedade vem do módulo
    `random` (seedado pelo --seed).
    """
    def __init__(self, env, alg, checkpoint_dir="checkpoints",
                 random_episodes=50, baseline_episodes=100, alt_prob=0.1):
        self.env = env
        self.alg = alg
        self.checkpoint_dir = Path(checkpoint_dir)
        self.random_episodes = random_episodes
        self.baseline_episodes = baseline_episodes
        self.alt_prob = alt_prob
        shape, n_actions = env.observation_space.shape, env.action_space.n
        # Os agentes oponentes usam o MESMO algoritmo do treino: o state_dict
        # do melhor checkpoint é copiado para cá em `update_best`, e as chaves
        # precisam casar.
        self.checkpoint_agent = alg.Agent(shape, n_actions)
        self.alt_agent = alg.Agent(shape, n_actions)

    def init_from_disk(self):
        """Na retomada, carrega o melhor checkpoint existente como oponente.

        Checkpoint incompatível é ignorado com aviso; o agente fica com os
        pesos iniciais até o primeiro `update_best`.
        """
        best = self.checkpoint_dir / "best.pt"
        if best.exists():
            _try_load_checkpoint(best, self.checkpoint_agent,
                                 frame_stack=self.env.cfg.frame_stack)
        return self

    def update_best(self, agent):
        """Congela os pesos atuais do treino como 'melhor até o momento'."""
        self.checkpoint_agent.online.load_state_dict(agent.online.state_dict())

    def _checkpoints(self):
        if not self.checkpoint_dir.is_dir():
            return []
        return sorted(self.checkpoint_dir.glob("*.pt"))

    def choose(self, episode_number):
        """Política do adversário para o episódio global (1-based)."""
        n = episode_number
        if n <= self.random_episodes:
            return make_policy("random")
        if n <= self.random_episodes + self.baseline_episodes:
            return make_policy("coins" if n % 2 else "rule")
        # Fase de checkpoint: melhor até o momento, com alternância ocasional.
        if random.random() < self.alt_prob:
            options = ["random", "coins", "rule"]
            checkpoints = self._checkpoints()
            if checkpoints:
                options.append("checkpoint")
            pick = random.choice(options)
            if pick == "checkpoint":
                # Tenta os arquivos em ordem sorteada; incompatíveis são
                # pulados com aviso. Se nenhum carregar, cai para random.
                for path in random.sample(checkpoints, len(checkpoints)):
                    if _try_load_checkpoint(path, self.alt_agent,
                                            frame_stack=self.env.cfg.frame_stack):
                        return CheckpointPolicy(self.alt_agent)
                return make_policy("random")
            return make_policy(pick)
        return CheckpointPolicy(self.checkpoint_agent)
