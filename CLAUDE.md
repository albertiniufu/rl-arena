# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Sobre o projeto

Plataforma 2D didática para treinar agentes DQN (PyTorch + Gymnasium + NumPy + Pygame). O mapa é fixo; o agente (personagem 0) aprende a coletar moedas e empurrar um adversário controlado por política simples. Há quatro algoritmos prontos, selecionados por `--alg`: `alg_dqn` (DQN enxuto), `alg_doubledqn_per_dueling` (Double DQN + PER + dueling), `alg_ppo` (PPO) e `alg_a2c` (A2C). Documentação e comentários do código são em português — responda e escreva código/comentários em português também.

Repositório git local (branch `main`). Artefatos gerados (`venv/`, `checkpoints/`, `logs/`) são ignorados pelo `.gitignore`.

## Comandos

Use o venv existente em `venv/` (Python 3.10; o README menciona `.venv`, mas o diretório real é `venv`):

```bash
source venv/bin/activate          # ou venv/bin/python diretamente
pip install -r requirements.txt   # só se faltar dependência
pytest -q                         # 116 testes, ~10–30s
pytest tests/test_arena.py::test_reset_step_observation -q   # um único teste
```

Treinar (sem janela por padrão; `--episodes` = episódios **adicionais** ao retomar; `--alg` = sufixo do pacote `alg_*`):

```bash
python train.py --episodes 300                          # alg_dqn (padrão), adversário auto
python train.py --episodes 300 --alg doubledqn_per_dueling
python train.py --episodes 300 --alg ppo                # on-policy (exige --envs 1)
python train.py --episodes 300 --alg a2c                # on-policy (exige --envs 1)
python train.py --episodes 300 --render                 # com visualização
python train.py --episodes 300 --opponent random        # adversário aleatório
python train.py --episodes 300 --opponent coins         # adversário que persegue moedas
python train.py --episodes 300 --envs 4 --opponent coins  # vetorizado (exige adversário fixo)
python train.py --episodes 300 --resume checkpoints/latest.pt
python train.py --episodes 300 --checkpoint-dir run1 --resume latest.pt   # retoma de outro dir
python train.py --episodes 300 --stats        # perda/grad/Q por episódio em logs/stats.csv
python train.py --episodes 300 --tensorboard  # curvas/histogramas/obs em logs/runs (abra com tensorboard --logdir logs/runs)
```

Ao executar scripts que importam torch, prefixe com `TORCH_CPP_LOG_LEVEL=ERROR` para suprimir o spam de avisos do NNPACK.

Avaliar e reproduzir (política gulosa, sem atualizar pesos; o checkpoint identifica o algoritmo, então `evaluate.py`/`play.py` não têm `--alg`):

```bash
python evaluate.py --checkpoint checkpoints/best.pt --episodes 20
python play.py --checkpoint checkpoints/best.pt --render
python play.py --player coins --opponent coins --render   # baseline vs baseline, sem DQN
python play.py --player human --opponent coins            # jogar no teclado (abre janela)
python play.py --player human --opponent coins --skin neon  # --skin escolhe o renderer (auto-descoberto)
python play.py --player coins --opponent coins --record demo.mp4   # grava vídeo (30fps, 640x400)
```

Saídas: `checkpoints/best.pt` e `checkpoints/latest.pt`; `logs/training.csv` e `logs/evaluation.csv`.

## Arquitetura

Três camadas, cada uma importando apenas a camada de baixo:

- **`arena/config.py`** — `ArenaConfig` (dataclass): todos os parâmetros do mundo (física, recompensas, duração do episódio, contagem de oponentes/moedas). Sem dependências externas.
- **`arena/actions.py`** — códigos de ação (fonte única): `STOP=0`, `LEFT=1`, `RIGHT=2`, `JUMP=3`, `PUSH_LEFT=4`, `PUSH_RIGHT=5`, `N_ACTIONS=6`, `PUSH_INTENSITY=3`, `NON_PUSH_ACTIONS=(0,1,2,3)`. Engine, ambiente, políticas e agentes importam daqui.
- **`arena/engine.py`** — `ArenaEngine`: simulação pura (física de plataforma em pixels, respawn, moedas, empurrões), sem nenhuma dependência de Gym/Torch. `Character` e `Coin` são dataclasses mutáveis; `step(actions, dt)` retorna um dict de eventos (`collected`, `deaths_delta`). `PUSH_LEFT`/`PUSH_RIGHT` chamam `push(actor, direction, PUSH_INTENSITY)`; o personagem não anda no mesmo passo em que empurra. O RNG é `np.random.default_rng(seed)` reseedado em `reset`.
- **`arena/environment.py`** — `ArenaEnv(gym.Env)` envolve o engine. A observação é uma grade semântica channels-first `(5, H, W)` float32: canal 0 = plataformas, 1 = jogador, 2 = oponentes, 3 = moedas, 4 = mortos/respawn (célula de 20px, mapa 640×400 → 32×20). Com `frame_stack > 1` a observação vira `(5*stack, H, W)`; `frame_skip` repete a ação por N substeps. O espaço de ações é `Discrete(6)` (ver `arena/actions.py`): parado, andar ←/→, pulo vertical máximo, empurrar ←/→ com força máxima. A política do oponente vem de `agents/baseline.py`, resolvida por `make_policy()` (nomes `rule`/`random`/`coins` ou instância pronta); a cada passo o env monta um `PolicyContext` (obs pré-passo, personagens, `np_random` e `stacked_obs`) por oponente. `get_state()`/`set_state()` dão o snapshot serializável do ambiente (legado da retomada bit a bit; hoje sem chamadores).
- **`arena/renderer.py` + `arena/renderer_*.py`** — skins de visualização: cada arquivo define `ArenaRenderer(cfg)` com `render(engine, elapsed, info=None)`/`close()`. `arena/skins.py` (`list_skins()`/`load_renderer_class()`) descobre as skins pelos arquivos e carrega por sufixo — criar `arena/renderer_dinosaur.py` já habilita `--skin dinosaur` nos entry points (as choices do argparse vêm de `list_skins()`), sem tocar em mais código. O renderer é criado **lazily** em `env.render()` (só com `render_mode="human"`); fechar a janela encerra via `SystemExit`.
- **`agents/baseline.py`** — políticas do adversário, todas com assinatura `act(ctx: PolicyContext)`: `PushRulePolicy` (`rule`: aproxima do jogador e às vezes empurra com `PUSH_LEFT`/`PUSH_RIGHT`), `RandomPolicy` (`random`: sorteia entre `NON_PUSH_ACTIONS` — 0..3, sem empurrões) e `CoinRulePolicy` (`coins`: persegue a moeda visível e pula com `JUMP`). Registro de nomes em `POLICIES`/`make_policy()`; o `rng` vem do contexto (o `np_random` do env), nunca de um gerador próprio.
- **`agents/human.py`** — `HumanPolicy` para o jogador: lê o teclado via pygame (setas/WASD andar, espaço/↑/W pular, Q/E empurrar). Devolve `STOP` se não há sistema de vídeo inicializado. Usada por `play.py --player human` (que ativa o render automaticamente); não entra no registro do env para não puxar pygame no treino.
- **`alg_registry.py`** — infraestrutura comum aos algoritmos: `available_algorithms()` descobre diretórios `alg_*/agent.py` na raiz; `load_algorithm(name)` importa o pacote. Checkpoint pesos-only com `algorithm`, `arch_version`, `n_actions`, `obs_shape`, pesos online/alvo, otimizador, `steps`, `epsilon`, `gamma`, `episodes`, `config` e `extra` — sem buffer, RNGs ou estado do ambiente. `load_checkpoint` valida algoritmo, arquitetura, número de ações e formato da observação **antes** de copiar pesos; `resume=True` restaura otimizador/contadores e põe a rede em `train()`, senão só carrega pesos e põe em `eval()`. `algorithm_from_checkpoint(path)` resolve o pacote pelo campo `algorithm` (checkpoint antigo sem o campo é rejeitado). `apply_env_config(cfg, meta)` aplica `frame_stack`/`frame_skip` gravados. `CheckpointPolicy(agent)` adapta um agente ao contrato `act(ctx)` (guloso, usando `ctx.stacked_obs` quando há stack) — usada pelo `--opponent auto` e pelos testes.
- **`alg_dqn/model.py`** — `QNetwork(nn.Module)`: CNN (3 conv + ReLU → flatten) + MLP head (128) → `n_actions` Q-valores **diretos**, um por ação. O número de features do flatten é inferido com um forward de zeros no `__init__`.
- **`alg_dqn/replay_buffer.py`** — `ReplayBuffer` uniforme com `deque(maxlen)`: toda transição tem a mesma chance. Guarda a 5-tupla `(state, action, reward, next_state, terminated)` (sem `truncated`); `sample()` converte o deque em lista uma vez para indexação O(1).
- **`alg_dqn/agent.py`** — `DQNAgent` (`algorithm="dqn"`, `arch_version="alg-dqn-v1"`): online + target com **alvo fixo `C=10`** sincronizado a cada 10 `learn()` bem-sucedidos; alvo `r + γ·max Q_target(s', ·)`; epsilon-greedy com decay, Adam + `SmoothL1Loss`, grad clip 10 (`last_grad_norm` = norma pré-clip). `learn()` retorna `None` enquanto o buffer tem menos que `batch_size` — `train.py` conta com isso. `q_stats(state)` devolve (média, máx) dos Q-valores online sob `no_grad`; `act_batch(states)` age em lote (treino vetorizado); `head_weights()` devolve a última camada para o histograma do TensorBoard. `td_target(reward, next_q, terminated, gamma)` mantém o bootstrap quando `terminated=0` (truncamento).
- **`alg_doubledqn_per_dueling/model.py`** — `DuelingQNetwork`: mesma CNN; separa `V(s)` (`v_head` → 1) de `A(s, a)` (`a_head` → `n_actions`) e devolve `Q = V + A − média(A)`.
- **`alg_doubledqn_per_dueling/replay_buffer.py`** — `PrioritizedReplayBuffer` **rank-based**: prioridade = `|erro de TD| + 1e-6`, amostragem por rank (`1/rank`), `add(..., priority=None)` herda a prioridade máxima; `sample()` devolve também os índices para `update_priorities`.
- **`alg_doubledqn_per_dueling/agent.py`** — `DoubleDQNPerDuelingAgent` (`algorithm="doubledqn_per_dueling"`, `arch_version="alg-ddqn-per-dueling-v1"`): mesmas opções de treino e o **mesmo `C=10`** do `alg_dqn`, mas `double_td_target` usa a online para escolher a ação e a target para avaliar; o erro de TD do batch vira prioridade no PER (`_last_indices` guarda os índices amostrados). `head_weights()` usa o ramo de valor.
- **`alg_ppo/model.py`** — `ActorCriticNetwork`: a mesma CNN, mas com `actor` (logits, um por ação) e `critic` (V(s) escalar); `forward` devolve `(logits, value)`.
- **`alg_ppo/rollout_buffer.py`** — `RolloutBuffer` on-policy em lista, sem amostragem: guarda `(s, a, r, ns, terminated, episode_end)`; `mark_episode_end()` marca a última transição para o GAE não atravessar episódios.
- **`alg_ppo/agent.py`** — `PPOAgent` (`algorithm="ppo"`, `arch_version="alg-ppo-v1"`, `on_policy=True`): ator categórico estocástico no treino e guloso em `act(..., greedy=True)`; `learn()` retorna `None` até o rollout ter `rollout_steps` (1024 por padrão) e então faz 4 épocas de minibatches (`batch_size`), com GAE(λ=0.95), clip 0.2, perda de valor 0.5, entropia 0.01 e grad clip 10. `target` é alias de `online` (não há rede-alvo); `epsilon=0`. `end_episode()` só marca a fronteira; `finish_training()` consome o rollout parcial.
- **`alg_a2c/rollout_buffer.py`** — mesmo buffer on-policy, autocontido.
- **`alg_a2c/agent.py`** — `A2CAgent` (`algorithm="a2c"`, `arch_version="alg-a2c-v1"`, `on_policy=True`): mesma rede actor-critic; `n_steps = batch_size` (64 por padrão), retornos n-step com bootstrap no truncamento e corte na fronteira do episódio; um único passo de otimização por rollout. `learn()` espera `n_steps`; `end_episode()` marca a fronteira e, se o rollout ficou incompleto, faz o update do rabo; `finish_training()` também consome resto. `target` alias de `online`, `epsilon=0`.
- **Entry points na raiz** — `train.py`, `evaluate.py`, `play.py`: `train.py` importa o pacote via `load_algorithm(args.alg)` e monta `ArenaConfig → ArenaEnv → alg.Agent`, aprendendo **a cada passo** (1 transição → 1 `learn()`), não por episódio; em PPO/A2C o `learn()` só atualiza quando o rollout enche (PPO) ou a cada `n_steps` (A2C), e os ganchos opcionais `end_episode(info)`/`finish_training()` marcam a fronteira e consomem o rollout parcial; seeda `random`/`numpy`/`torch` com `--seed`, continua a numeração de episódios ao retomar (via `payload["episodes"]`, com fallback para `extra["episodes_done"]`). O adversário padrão é `auto` (`AutoOpponent(env, alg, ...)` em train.py: 50 eps `random` → 100 alternando `coins`/`rule` → melhor checkpoint, com 10% de alternância para `random`/`coins`/`rule`/outro `.pt` de `checkpoints/`; fases por **contagem absoluta**, então retomar continua o currículo de onde parou; os agentes oponentes usam o mesmo `alg` do treino; o objeto é criado **antes** do load para `init_from_disk` já o deixar pronto). `--checkpoint-dir` define onde salvar/ler os checkpoints (e resolve `--resume` com nome simples). `--stats` grava `logs/stats.csv` por episódio (perda mín/máx, grad pré-clip, Q online/valor do crítico, buffer) sem consumir RNG — não altera o treino. `--max-hours N` encerra o treino de forma limpa após N horas de parede (checa ao fim de cada episódio, salva o `latest.pt` com o episódio real e consome o rollout on-policy parcial); `time_budget_reached` é o helper testável. `--tensorboard` grava em `logs/runs/<alg>_configs_<opponent>_fs<stack>_fk<skip>_lr<lr>_b<batch>_seed<seed>` escalares por episódio e por passo, histogramas de pesos (conv1 e `head_weights()`) e a observação como imagem (`obs_to_image`); retomar com os mesmos argumentos continua as curvas; `--run-name` sobrescreve o nome; também não consome RNG (verificado byte a byte). `evaluate.py`/`play.py` leem o algoritmo do checkpoint (`algorithm_from_checkpoint`), aplicam `frame_stack`/`frame_skip` da config gravada e criam o agente do pacote certo. `play.py` roda o checkpoint como jogador por padrão, ou uma política baseline com `--player` (`coins`/`random`/`human`; com `human`, o render é ativado automaticamente) sem precisar de checkpoint; no modo humano o `PolicyContext` passa `self_character`/`player_character` para o controle no ar. `--record ARQUIVO.mp4` grava a sessão (imageio + `pygame.surfarray`, captura por passo + frame inicial; ativa o render, e com `SDL_VIDEODRIVER=dummy` grava headless; o writer fecha em `finally`, então fechar a janela no meio não corrompe o vídeo).

- **`scripts/train_long.sh`** — treino por tempo de parede dos quatro algoritmos em sequência: `HOURS` (padrão 3) por algoritmo via `--max-hours`, mesmo protocolo de seed/oponente/frame stack/batch, avaliação de `best.pt` e `latest.pt` (20 eps, seed 1000) ao fim de cada treino. Grava em `checkpoints/long3h/<tag>` e `logs/long3h/`, com runs do TensorBoard sufixados `_3h`; recusa sobrescrever diretórios existentes.

- **`scripts/compare_onpolicy.sh` + `scripts/summarize_all.py`** — runner reprodutível de PPO/A2C no mesmo protocolo da comparação DQN/Double DQN (seed 7, 100 eps, `rule`, `frame_stack=4`, `frame_skip=3`, `batch=64`, TensorBoard, snapshots `ep050`/`ep100`/`best`, avaliação gulosa de 20 eps com seed 1000). Grava tudo em `logs/compare` e deixa os runs em `logs/runs/<alg>_configs_<opponent>_fs<stack>_fk<skip>_lr<lr>_b<batch>_seed<seed>`; recusa sobrescrever diretórios existentes. O sumarizador lê os quatro algoritmos, usa campos ausentes como `n/d`, calcula deltas pareados contra o DQN e escreve `summary-all.txt`, `training_summary_all.csv`, `evaluation_summary_all.csv` e `comparacao-all.md`. Com `--hours N`, descreve o protocolo por tempo de parede e inclui o checkpoint `latest`/`final` (ausente nos CSVs antigos vira `n/d`).

## Semântica que não é óbvia

- **Ações reduzidas**: 6 ações, com uma única intensidade de pulo e uma de empurrão por direção (as máximas). `JUMP` é vertical puro; o deslocamento horizontal no ar continua possível com `LEFT`/`RIGHT` no passo seguinte. `PUSH_LEFT`/`PUSH_RIGHT` usam `PUSH_INTENSITY=3` e não movem o personagem no mesmo passo. O adversário `random` sorteia apenas `NON_PUSH_ACTIONS` (0..3).
- **Episódios só terminam por truncamento** (45s simulados, `truncated=True`); morte nunca termina episódio. Morrer custa 10 moedas (saldo mínimo 0), 5s inativo e respawn em plataforma aleatória. Não existe mais o conceito de eliminação: a morte de um oponente não rende recompensa nem crédito ao jogador. Recompensas: +1 moeda, −2 morte, −0.1 por step do agente escolhendo `STOP` e −0.1 por step empurrando sem oponente vivo em contato no início do step (`push_miss_penalty`). As duas penalidades são aplicadas uma vez por step, não por substep de `frame_skip`; as demais ações não pagam.
- **Observação não é Markoviana**: velocidades e timers (respawn, moedas) não estão codificados na grade — simplificação didática deliberada.
- **Quatro algoritmos, mesma base**: `dqn` e `doubledqn_per_dueling` são off-policy e usam `C=10` (alvo fixo a cada 10 `learn()`), a mesma observação, recompensa e seed. `alg_dqn` usa Q direto, replay uniforme e `max Q_target`; `alg_doubledqn_per_dueling` usa dueling, PER rank-based e Double DQN (online escolhe, target avalia). O bootstrap usa apenas `terminated`; truncamento por tempo continua bootstrapping. `ppo` e `a2c` são on-policy, com ator categórico (política estocástica no treino e gulosa na avaliação) e crítico compartilhando a CNN; não há replay nem rede-alvo, então `target` é alias de `online` e `epsilon=0`. PPO usa rollout de 1024, GAE(λ=0.95), 4 épocas e clip 0.2; A2C usa retornos n-step (`n_steps=batch_size`) e um update por rollout, consumindo o rabo no fim do episódio. Ambos exigem `--envs 1`.
- **Retomada não é bit a bit reproduzível** (decisão do autor): o checkpoint guarda apenas pesos, otimizador, contadores e metadados; retomar continua a numeração de episódios e o treino, mas buffer e RNGs recomeçam do zero. Retomar exige o mesmo `--alg` e o mesmo formato de observação; checkpoints antigos (`cnn-dqn-*`, 12 ações, sem campo `algorithm`) não carregam (`ValueError`). Coberto por `test_resume_loads_weights_and_continues_episode_numbering` e pelos testes de `test_checkpoint.py`.
- **Seeds**: `train.py` faz `env.reset(seed=args.seed + ep)` por episódio; `evaluate.py` usa `--seed 1000`. O RNG do oponente é `env.np_random`, o do engine é separado.
- Testes manipulam o engine diretamente (ex.: `env.engine.characters[0].y = ...`) para forçar mortes — essa é a forma esperada de testar a física sem esperar 45s.
- **Adicionar um algoritmo**: crie `alg_<nome>/agent.py` com uma classe `Agent` que cumpra o contrato de `alg_dqn/agent.py` (atributos `algorithm`, `arch_version`, `obs_shape`, `n_actions`, `online`, `target`, `optimizer`, `buffer`, `steps`, `epsilon`; métodos `act`, `act_batch`, `learn`, `q_stats`, `head_weights`). Para um algoritmo on-policy, declare `on_policy = True`, use `target = online`, implemente `end_episode(info)`/`finish_training()` opcionais e aceite que `--envs > 1` será rejeitado. `--alg <nome>` passa a funcionar sem tocar em `train.py`; `evaluate.py`/`play.py` seguem o campo `algorithm` do checkpoint.
