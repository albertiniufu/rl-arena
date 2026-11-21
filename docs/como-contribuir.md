# Como contribuir

Guia curto para mexer no repositório sem quebrar as outras arenas.

## Preparar o ambiente

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Dica: se aparecer muito aviso do NNPACK, rode com
`TORCH_CPP_LOG_LEVEL=ERROR`.

## Antes de abrir um PR

```bash
pytest -q                                  # suíte completa
pytest -q arenas/seek_coin/tests/test_arena.py # o que você mexeu
python train.py --arena <arena> --episodes 2   # smoke test do caminho real
```

Checklist:

- [ ] Testes passam.
- [ ] O smoke test de treino roda (2 episódios bastam).
- [ ] Documentação afetada foi atualizada (README da arena, `docs/`).
- [ ] Nada de `checkpoints/`, `logs/`, vídeos ou `__pycache__` no commit.
- [ ] Comentários e mensagens em português.

## Convenções do projeto

- **Português** em documentação, comentários e mensagens de erro.
- **Uma arena não importa outra arena.** O compartilhado fica em
  `arenas/common/` ou `arenas/algs/`.
- **Sem listas hardcoded de algoritmos**: a descoberta é por diretório
  `alg_*/agent.py`.
- **Contrato Gymnasium**: `reset → (obs, info)` e
  `step → (obs, reward, terminated, truncated, info)`.
- **Seed reprodutível**: `reset(seed=...)` deve controlar o RNG do ambiente.
- **Checkpoint pesos-only**: sem buffer, sem RNG, sem estado do ambiente.
- **Teste o que importa**: contrato Gym, física/regras, checkpoint e um
  caminho de treino curto.

## Tipos de contribuição

### Corrigir um bug

1. Reproduza com um teste pequeno em `arenas/<arena>/tests/`.
2. Corrija.
3. Rode o teste e a suíte inteira.

### Configurações de execução em YAML

Presets ficam em `configs/<arena>/` e espaços de busca em
`configs/search_spaces/`. Use `--config` e `--set`:

```bash
python train.py --config configs/seek_coin/ppo-level3.yaml
python train.py --config configs/seek_coin/ppo-level3.yaml \
  --set training.max_hours=0.5 --set seed=42
```

A precedência é **CLI > YAML > defaults**. Chaves desconhecidas falham com
mensagem clara; veja [`configs/README.md`](../configs/README.md).

### Adicionar um algoritmo

Siga [`algoritmos.md`](algoritmos.md#adicionar-um-algoritmo). Resumo:

```text
arenas/algs/alg_meu/
├── __init__.py      # exporta Agent
├── model.py
├── replay_buffer.py # ou rollout_buffer.py
└── agent.py         # class Agent
```

O `--alg meu` funciona automaticamente. Atualize a tabela de
[`algoritmos.md`](algoritmos.md) e, se fizer sentido, escreva uma aula HTML
em `arenas/seek_coin/docs/`.

### Adicionar uma arena

1. Crie o pacote:

   ```text
   arenas/minha_arena/
   ├── __init__.py
   ├── config.py          # dataclass de configuração
   ├── env.py             # ambiente Gymnasium
   ├── train.py           # parse_args + main
   ├── evaluate.py
   ├── play.py
   ├── README.md
   └── tests/test_minha_arena.py
   ```

2. Use a infraestrutura comum em `train.py`:

   ```python
   from arenas.common.agent_factory import build_agent
   from arenas.common.trainer import train_agent

   env = MinhaEnv(config)
   agent = build_agent(args.alg, env.observation_space.shape,
                       env.action_space, lr=args.lr)
   train_agent(env, agent, config=config, episodes=args.episodes,
               seed=args.seed, checkpoint_dir="checkpoints/minha_arena",
               log_path=Path("logs/minha_arena/training.csv"))
   ```

3. Registre a arena em `arenas/registry.py`:

   ```python
   "minha_arena": ArenaSpec(
       name="minha_arena",
       description="Uma frase sobre o domínio.",
       package="arenas.minha_arena",
       train_entry="arenas.minha_arena.train",
       evaluate_entry="arenas.minha_arena.evaluate",
       play_entry="arenas.minha_arena.play",
       default_algorithm="dqn",
       action_space="discrete",   # ou "box"
   ),
   ```

4. Adicione o README da arena à tabela do README da raiz e ao
   [`docs/README.md`](README.md).

5. `python train.py --arena minha_arena --episodes 2` e `pytest -q`.

### Adicionar uma skin/renderizador (seek_coin)

Crie `arenas/seek_coin/arena/renderer_<nome>.py` com uma classe
`ArenaRenderer(cfg)` que tenha `render(...)` e `close()`. A skin é
descoberta sozinha e vira `--skin <nome>`.

### Adicionar um experimento

Scripts de execução longa e comparações ficam em
`arenas/seek_coin/scripts/`. Resultados vão para `experiments/`, que é
ignorado pelo git. Se o protocolo for importante, documente-o em
`arenas/seek_coin/scripts/README.md` ou em `docs/`.

### Projetar e reportar um experimento

Antes de rodar, responda no [`TEMPLATE.md`](experimentos/TEMPLATE.md):

1. **Pergunta** — uma frase testável.
2. **Hipótese e falsificação** — o que você espera e o que refutaria.
3. **Métrica primária** — definida antes, com direção (`maximize`/`minimize`).
4. **Baseline forte** — com tuning comparável ao candidato.
5. **Orçamento e seeds** — mesmo orçamento; 3+ seeds de treino.
6. **Validação × teste** — tuning e checkpoint na validação; conclusão no teste.
7. **Critério de decisão** — suporta / inconclusivo / abandona.

Use a skill `experimenter-helper` para gerar o par
`docs/experimentos/EXP-....md` + `.yaml`:

```bash
python .agents/skills/experimenter-helper/scripts/validate_yaml.py   docs/experimentos/EXP-AAAAMMDD-slug.yaml
```

Depois de rodar:

```bash
python -m arenas.common.report   --input experiments/EXP-AAAAMMDD-slug/results-by-train-seed.csv   --metric reward_mean --paired --baseline <baseline>   --out experiments/EXP-AAAAMMDD-slug/report.html
```

Regras de reporte:

- uma linha por **seed de treino** na análise primária;
- episódios de avaliação são análise secundária, não runs independentes;
- reporte média, mediana, desvio, IC, tamanho de efeito e p corrigido;
- não escolha a métrica, o checkpoint ou o filtro depois de ver o resultado;
- inclua runs falhas, seeds descartadas e resultados negativos no CSV.

Leia [`experimentos/empirical-design.md`](experimentos/empirical-design.md)
para os princípios e os critérios de decisão.

## Onde documentar o quê

| Mudança | Onde atualizar |
| --- | --- |
| Comando novo de CLI | README da arena e `--help` |
| Regra/de física nova | README da arena + testes |
| Camada compartilhada | `docs/arquitetura.md` |
| Algoritmo novo | `docs/algoritmos.md` + aula HTML (opcional) |
| Decisão de projeto | `BACKLOG.md` ou comentário no código |
| Instruções para agentes | `CLAUDE.md` (raiz ou arena) |

## Armadilhas comuns

- **Dois `train.py`**: o da raiz despacha por `--arena`; o da arena é o
  `main()` real. Ao depurar, rode o da arena para ver o traceback direto.
- **`--episodes` ao retomar** significa episódios **adicionais**, não o total.
- **On-policy ignora `--learn-every`**: o `learn()` espera o rollout encher.
- **Checkpoint de outro algoritmo** falha de propósito; retreine em vez de
  tentar forçar.
- **`--envs > 1` exige adversário fixo** no seek_coin e não combina com
  `--overlap` em todos os casos; leia o README da arena.
- **Não versione artefatos**: `checkpoints/`, `logs/`, `experiments/` e
  `videos/` já estão no `.gitignore`.
