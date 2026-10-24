# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Sobre o projeto

Plataforma 2D didática para treinar um agente DQN (PyTorch + Gymnasium + NumPy + Pygame). O mapa é fixo; o agente (personagem 0) aprende a coletar moedas e empurrar um adversário controlado por política simples. Documentação e comentários do código são em português — responda e escreva código/comentários em português também.

Repositório git local (branch `main`). Artefatos gerados (`venv/`, `checkpoints/`, `logs/`) são ignorados pelo `.gitignore`.

## Comandos

Use o venv existente em `venv/` (Python 3.10; o README menciona `.venv`, mas o diretório real é `venv`):

```bash
source venv/bin/activate          # ou venv/bin/python diretamente
pip install -r requirements.txt   # só se faltar dependência
pytest -q                         # 5 testes, ~4s
pytest tests/test_arena.py::test_reset_step_observation -q   # um único teste
```

Treinar (sem janela por padrão; `--episodes` = episódios **adicionais** ao retomar):

```bash
python train.py --episodes 300                          # do zero (adversário padrão: auto)
python train.py --episodes 300 --render                 # com visualização
python train.py --episodes 300 --opponent random        # adversário aleatório
python train.py --episodes 300 --opponent coins         # adversário que persegue moedas
python train.py --episodes 300 --resume checkpoints/latest.pt
python train.py --episodes 300 --checkpoint-dir run1 --resume latest.pt   # retoma de outro dir
python train.py --episodes 300 --stats        # perda/grad/Q por episódio em logs/stats.csv
python train.py --episodes 300 --tensorboard  # curvas/histogramas/obs em logs/runs (abra com tensorboard --logdir logs/runs)
```

Ao executar scripts que importam torch, prefixe com `TORCH_CPP_LOG_LEVEL=ERROR` para suprimir o spam de avisos do NNPACK.

Avaliar e reproduzir (política gulosa, sem atualizar pesos):

```bash
python evaluate.py --checkpoint checkpoints/best.pt --episodes 20
python play.py --checkpoint checkpoints/best.pt --render
python play.py --player coins --opponent coins --render   # baseline vs baseline, sem DQN
python play.py --player human --opponent coins            # jogar no teclado (abre janela)
python play.py --player human --opponent coins --skin neon  # --skin escolhe o renderer (auto-descoberto)
```

Saídas: `checkpoints/best.pt` e `checkpoints/latest.pt`; `logs/training.csv` e `logs/evaluation.csv`.

## Arquitetura

Três camadas, cada uma importando apenas a camada de baixo:

- **`arena/config.py`** — `ArenaConfig` (dataclass): todos os parâmetros do mundo (física, recompensas, duração do episódio, contagem de oponentes/moedas). Sem dependências externas.
- **`arena/engine.py`** — `ArenaEngine`: simulação pura (física de plataforma em pixels, respawn, moedas, empurrões), sem nenhuma dependência de Gym/Torch. `Character` e `Coin` são dataclasses mutáveis; `step(actions, dt)` retorna um dict de eventos (`collected`, `deaths_delta`, `eliminations_delta`). O RNG é `np.random.default_rng(seed)` reseedado em `reset`.
- **`arena/environment.py`** — `ArenaEnv(gym.Env)` envolve o engine. A observação é uma grade semântica channels-first `(5, H, W)` float32: canal 0 = plataformas, 1 = jogador, 2 = oponentes, 3 = moedas, 4 = mortos/respawn (célula de 20px, mapa 640×400 → 32×20). O espaço de ações é `Discrete(12)`: 0 parado, 1–2 andar, 3–5 pular (vertical/esquerda/direita), 6–11 empurrar adversário (3 intensidades × 2 direções). A política do oponente vem de `agents/baseline.py`, resolvida por `make_policy()` (nomes `rule`/`random`/`coins` ou instância pronta); a cada passo o env monta um `PolicyContext` (obs pré-passo + personagens + `np_random`) por oponente. `get_state()`/`set_state()` dão o snapshot serializável usado pelos checkpoints.
- **`arena/renderer.py` + `arena/renderer_*.py`** — skins de visualização: cada arquivo define `ArenaRenderer(cfg)` com `render(engine, elapsed, info=None)`/`close()`. `arena/skins.py` (`list_skins()`/`load_renderer_class()`) descobre as skins pelos arquivos e carrega por sufixo — criar `arena/renderer_dinosaur.py` já habilita `--skin dinosaur` nos entry points (as choices do argparse vêm de `list_skins()`), sem tocar em mais código. O renderer é criado **lazily** em `env.render()` (só com `render_mode="human"`); fechar a janela encerra via `SystemExit`.
- **`agents/baseline.py`** — políticas do adversário, todas com assinatura `act(ctx: PolicyContext)`: `PushRulePolicy` (`rule`: aproxima do jogador e às vezes empurra), `RandomPolicy` (`random`: sorteia 0..5 — comportamento histórico preservado, sem empurrões) e `CoinRulePolicy` (`coins`: persegue a moeda visível). Registro de nomes em `POLICIES`/`make_policy()`; o `rng` vem do contexto (o `np_random` do env), nunca de um gerador próprio.
- **`agents/human.py`** — `HumanPolicy` para o jogador: lê o teclado via pygame (setas/WASD andar, espaço/↑/W pular, Q/E empurrar). Devolve 0 se não há sistema de vídeo inicializado. Usada por `play.py --player human` (que ativa o render automaticamente); não entra no registro do env para não puxar pygame no treino.
- **`dqn/model.py`** — `DQN(nn.Module)`: CNN (3 conv + ReLU → flatten) + MLP head (128) → 12 Q-valores. O número de features do flatten é inferido com um forward de zeros no `__init__`.
- **`dqn/agent.py`** — `DQNAgent`: rede online + target (sincronizada a cada `target_update=500` passos), epsilon-greedy com decay, Adam + `SmoothL1Loss`, grad clip 10 (`last_grad_norm` guarda a norma pré-clip). `learn()` retorna `None` enquanto o buffer tem menos que `batch_size` — `train.py` conta com isso. `q_stats(state)` devolve (média, máx) dos Q-valores online sob `no_grad`. `CheckpointPolicy` adapta um `DQNAgent` ao contrato de política `act(ctx)` (guloso) — usada pelo `--opponent auto`.
- **`dqn/replay_buffer.py`** — `ReplayBuffer` com `deque(maxlen)`. Guarda `terminated` e `truncated`, mas o bootstrap em `learn()` usa apenas `terminated`. `state_dict()` serializa **empilhado** (um array por campo) — determinístico e compacto; `load_state_dict()` aceita também o formato antigo (lista de tuplas).
- **`dqn/checkpoint.py`** — `save_checkpoint`/`load_checkpoint` com verificação de compatibilidade (`arch_version` e `n_actions`; `ValueError` se não bater). Além de pesos/otimizador/`steps`/`epsilon`/`episodes` (episódios consumidos), persiste buffer, `rng_state` (Python, NumPy, Torch, engine e `np_random`) e `env_state` (snapshot do ambiente) quando `env` é passado. `resume=True` restaura tudo isso; caso contrário só carrega pesos e põe a rede em `eval()`. Usa `torch.load(..., weights_only=False)`.
- **Entry points na raiz** — `train.py`, `evaluate.py`, `play.py`: `train.py`/`evaluate.py` montam `ArenaConfig → ArenaEnv → DQNAgent` e rodam o loop. `train.py` aprende **a cada passo** (1 transição → 1 `learn()`), não por episódio; seeda `random`/`numpy`/`torch` com `--seed`, continua a numeração de episódios ao retomar (via `payload["episodes"]`, com fallback para `extra["episodes_done"]` em checkpoints antigos) e passa `env=env` a `save_checkpoint`/`load_checkpoint`. O adversário padrão é `auto` (`AutoOpponent` em train.py: 50 eps `random` → 100 alternando `coins`/`rule` → melhor checkpoint, com 10% de alternância para `random`/`coins`/`rule`/outro `.pt` de `checkpoints/`; fases por **contagem absoluta** para retomada reproduzível, e o objeto é criado **antes** do load para o consumo de torch RNG casar com a run contínua). `--checkpoint-dir` define onde salvar/ler os checkpoints (e resolve `--resume` com nome simples). `--stats` grava `logs/stats.csv` por episódio (perda mín/máx, grad pré-clip, Q online, buffer) sem consumir RNG — treino continua bit a bit reproduzível com ou sem a flag. `--tensorboard` grava em `logs/runs/seed<N>` escalares por episódio e por passo, histogramas de pesos (conv1 e head) e a observação como imagem (`obs_to_image`); retomar com o mesmo seed continua as curvas; também não consome RNG (verificado byte a byte). `play.py` roda o checkpoint DQN como jogador por padrão, ou uma política baseline com `--player` (`coins`/`random`/`human`; com `human`, o render é ativado automaticamente) sem precisar de checkpoint.

## Semântica que não é óbvia

- **Episódios só terminam por truncamento** (45s simulados, `truncated=True`); morte nunca termina episódio. Morrer custa 10 moedas (saldo mínimo 0), 5s inativo e respawn em plataforma aleatória. Eliminação é creditada ao jogador de forma simplificada (oponente morre com jogador vivo/adjacente). Recompensas: +1 moeda, +2 eliminação, −1 morte.
- **Observação não é Markoviana**: velocidades e timers (respawn, moedas) não estão codificados na grade — simplificação didática deliberada.
- **Retomada é bit a bit reproduzível**: o checkpoint persiste buffer, RNGs e estado do ambiente; com os mesmos argumentos, retomar gera os mesmos episódios, pesos e até os mesmos bytes de `latest.pt`/`best.pt` que um treino contínuo. Checkpoints antigos (sem esses campos) ainda carregam, sem reproduzibilidade. Coberto por `test_resume_is_bit_identical`.
- **Seeds**: `train.py` faz `env.reset(seed=args.seed + ep)` por episódio; `evaluate.py` usa `--seed 1000`. O RNG do oponente é `env.np_random`, o do engine é separado.
- Testes manipulam o engine diretamente (ex.: `env.engine.characters[0].y = ...`) para forçar mortes — essa é a forma esperada de testar a física sem esperar 45s.
