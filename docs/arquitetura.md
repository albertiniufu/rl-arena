# Arquitetura

O repositório tem **três camadas** e uma regra simples: cada camada só
importa a de baixo. Código de uma arena nunca é importado por outra arena.

```text
┌──────────────────────────────────────────────────────────────┐
│ 1. Entrada        train.py / evaluate.py / play.py (raiz)    │
│                   arenas/cli.py + arenas/registry.py         │
├──────────────────────────────────────────────────────────────┤
│ 2. Aplicação      arenas/common/   (loop, avaliação, ckpt)   │
│                   arenas/algs/     (algoritmos)              │
├──────────────────────────────────────────────────────────────┤
│ 3. Domínio        arenas/<arena>/  (ambiente + regras)       │
└──────────────────────────────────────────────────────────────┘
```

## 1. Entrada: o CLI unificado

`train.py`, `evaluate.py` e `play.py` na raiz são cascas finas:

```python
from arenas.cli import dispatch
dispatch("train")
```

`arenas/cli.py` lê **apenas** `--arena` (padrão `seek_coin`), consulta
`arenas/registry.py` e importa o `main()` da arena. O resto dos argumentos é
repassado intacto. Isso evita importar torch/pygame quando o usuário só quer
ver o `--help`.

```text
python train.py --arena risk --alg dqn
         │           │        └── vai para arenas/risk/train.py
         │           └── ArenaSpec("risk") em arenas/registry.py
         └── arenas/cli.py
```

Cada `ArenaSpec` declara nome, descrição, pacote, entry points, algoritmo
padrão e tipo de ação. Para adicionar uma arena, basta registrar uma entrada
em `ARENAS`.

> As arenas também têm entry points próprios (`arenas/risk/train.py`,
> `arenas/economy/train.py`, ...). Eles funcionam sozinhos e são o que o CLI
> chama; existem para permitir `python arenas/risk/train.py ...` sem a raiz.

## 2. Aplicação: infraestrutura e algoritmos

### `arenas/common/`

O que é reutilizável por qualquer arena:

| Arquivo | Papel |
| --- | --- |
| `agent_factory.py` | `build_agent(alg, obs_shape, action_space)` valida o tipo de ação e instancia o agente. |
| `trainer.py` | `train_agent(...)`: loop de episódios, CSV, checkpoints, TensorBoard, orçamento por tempo e vídeo. |
| `evaluator.py` | `evaluate_agent(...)`: episódios gulosos, sem update, com métricas agregadas. |
| `checkpoint.py` | `agent_from_checkpoint(...)`: reconstrói o agente a partir do `.pt`. |
| `report.py` | Relatório estatístico por experimento: descritivas, IC, testes, tamanho de efeito e correção de múltiplas comparações. |
| `yaml_config.py` | Presets YAML com precedência CLI > YAML > defaults e `--set chave=valor`. |

O `report.py` é a ponte entre os resultados brutos e a conclusão: ele lê
CSV/JSONL (uma linha por seed/execução), compara algoritmos e gera
`report.md`. O formato canônico está em
[`docs/experimentos/README.md`](experimentos/README.md).

O `trainer.py` é o **trainer genérico** usado por `risk` e `economy`. O
`seek_coin` mantém o seu trainer histórico dentro do próprio pacote
(`arenas/seek_coin/train.py`) porque precisa de currículo de oponentes,
multiagente, vetorização e sobreposição de coleta/update. Isso é uma exceção
consciente, não duplicação acidental.

### `arenas/algs/`

Cada algoritmo é um pacote autocontido:

```text
arenas/algs/alg_dqn/
├── model.py           # rede neural
├── replay_buffer.py   # memória (off-policy) ou rollout (on-policy)
└── agent.py           # Agent: act, learn, checkpoint...
```

`alg_registry.py` faz a **descoberta**: varre os diretórios `alg_*/` que
tenham `agent.py`, importa o módulo e centraliza o checkpoint
(`save_checkpoint`/`load_checkpoint`) e o adaptador `CheckpointPolicy`.
Nenhuma lista de algoritmos fica hardcoded: criar `alg_meu/agent.py` já
habilita `--alg meu`.

O contrato completo está em [`algoritmos.md`](algoritmos.md).

## 3. Domínio: as arenas

Cada arena segue o contrato Gymnasium:

```python
obs, info = env.reset(seed=...)
obs, reward, terminated, truncated, info = env.step(action)
```

Convenções:

- `terminated` = fim real (vitória, falência); `truncated` = limite de tempo.
- `info` é um dicionário com métricas do episódio (ex.: `winner`, `bank`).
- O ambiente recebe/expõe RNG e `reset(seed=...)` é reprodutível.
- Ações discretas usam `gymnasium.spaces.Discrete`; contínuas, `Box`.

Exemplos de organização interna:

- **seek_coin**: `arena/engine.py` (física pura) → `arena/environment.py`
  (Gym) → `arena/multi_environment.py` (ações conjuntas). Renderizadores são
  descobertos por arquivo (`arena/renderer_<skin>.py`). O pacote
  `training/` guarda telemetria, currículo de oponentes e avaliação;
  `train.py` fica com a CLI e os loops.
- **risk**: `game.py` (regras) → `env.py` (Gym) → `agents.py` (bot).
- **economy**: `model.py` (macroeconomia) + `bank.py` (crédito) →
  `bank_env.py` (Gym).

## O caminho de um treino

```text
train.py --arena risk --alg dqn --episodes 300
  │
  ├─ arenas/cli.py: dispatch("train")
  ├─ arenas/registry.py: ARENAS["risk"]
  └─ arenas/risk/train.py: main()
       ├─ RiskConfig + RiskEnv
       ├─ build_agent("dqn", obs_shape, action_space)
       └─ train_agent(env, agent, config=..., ...)
            ├─ loop: reset → act → step → buffer.add → learn
            ├─ logs/training.csv
            └─ checkpoints/risk/{best,latest}.pt
```

O `train_agent` chama, por episódio:

1. `env.reset(seed=seed + episódio)`;
2. `agent.act(obs)` até `terminated or truncated`;
3. `agent.buffer.add(...)` e `agent.learn()` a cada `learn_every` passos;
4. ganchos opcionais `agent.end_episode(info)` e `agent.finish_training()`;
5. gravação de CSV, TensorBoard e checkpoints.

Algoritmos on-policy ignoram `learn_every`: o `learn()` só atualiza quando o
rollout enche. Algoritmos off-policy usam replay e epsilon-greedy.

## O que vai no checkpoint

O checkpoint é **pesos-only** (não guarda buffer nem RNG):

```text
{
  "env_version", "algorithm", "arch_version",
  "n_actions", "obs_shape",
  "online", "target", "optimizer",
  "steps", "epsilon", "gamma", "episodes",
  "config",   # parâmetros da arena (frame_stack, nível, cenário...)
  "extra"     # melhor recompensa, episódio, etc.
}
```

`load_checkpoint` valida algoritmo, arquitetura, número de ações e formato da
observação **antes** de copiar qualquer peso — um checkpoint incompatível
falha com mensagem clara em vez de deixar o agente pela metade.

Por isso `evaluate.py`/`play.py` **não pedem `--alg`**: leem o algoritmo do
checkpoint e reconstroem a arena com a `config` gravada. Retomar com
`--resume` restaura pesos, otimizador e contadores; a numeração de episódios
continua de onde parou.

## Onde cada coisa é testada

| Escopo | Diretório |
| --- | --- |
| Contrato Gym, física e multiagente do seek_coin | `arenas/seek_coin/tests/` |
| Algoritmos (DQN, PPO, EPPO, MAPPO, ...) | `arenas/seek_coin/tests/` |
| Regras do Risk | `arenas/risk/tests/` |
| Economia e banco | `arenas/economy/tests/` |
| CLI, fábrica, trainer e checkpoints | `arenas/common/tests/` |

Os algoritmos são compartilhados, mas os testes de integração deles vivem na
arena que os exercita (hoje, `seek_coin`).
