# Melhorias do DQN — Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fazer o DQN do RL Arena aprender mais rápido (≥2× no orçamento de 150 episódios) e treinar mais rápido (≥3×), por fases medidas.

**Architecture:** Fases incrementais sobre a stack atual (Gymnasium + PyTorch, CPU). Cada fase termina com gate de A/B (15% de ganho sobre a fase anterior, média de 3 seeds, orçamento 150 eps). Checkpoints viram pesos-only; o ambiente ganha frame stack/skip; o agente ganha warmup, Double DQN, n-step, PER, dueling; a Fase 5 vetoriza o treino em 8 processos.

**Tech Stack:** Python 3.10+, PyTorch, Gymnasium, NumPy, pytest (venv existente em `venv/`).

**Spec:** `docs/superpowers/specs/2026-10-24-melhorias-dqn-design.md` — o plano argumenta a partir do spec; leia os dois juntos.

## Global Constraints

- Python 3.10+; CPU apenas (8 núcleos); dependências atuais de `requirements.txt`.
- Comandos de treino/avaliação prefixados com `TORCH_CPP_LOG_LEVEL=ERROR`.
- Testes com `venv/bin/python -m pytest -q` (o `pytest` puro falha no import).
- Comentários e mensagens de commit em português; commits sem trailer de coautoria.
- CLI de `train.py`, `evaluate.py`, `play.py` mantém todas as flags atuais funcionando.
- Checkpoint: `ARCH_VERSION` sobe para `"cnn-dqn-v2"`; checkpoints antigos deixam de carregar (documentado); retomada bit a bit é abandonada (checkpoint = pesos, otimizador, contadores).
- Corridas de A/B: adversário fixo `random`; seeds 7/42/123; avaliação gulosa com `evaluate.py --seed 1000 --episodes 20`; teto ~12 h de CPU por fase.

## Review Focus

- **Checkpoint antigo no diretório do adversário `auto`** — `AutoOpponent` varre `checkpoints/*.pt`; após o bump de arquitetura, um `.pt` antigo (ex.: `best.pt` da run baseline) não pode quebrar o treino; deve ser pulado com aviso. (Task 3)
- **Snapshots de orçamento virando adversários** — `checkpoints/budget/*.pt` não pode entrar no glob do `auto` (glob não-recursivo já exclui; teste pina isso). (Task 3)
- **Retomada após o abandono da reprodutibilidade** — `--resume` com checkpoint novo deve continuar do episódio correto sem exigir `env_state`/`buffer` ausentes. (Task 1)
- **Episódio mais curto que n** — n-step com n=3 não pode atravessar fronteira de episódio (parar em `terminated` ou `truncated`). (Task 9)
- **Obs empilhada em `obs_to_image` e no `CheckpointPolicy`** — com frame stack, a imagem do TensorBoard deve mostrar o frame atual (últimos 5 canais) e o adversário `checkpoint` deve agir sobre a obs empilhada sem erro de shape. (Task 8, Task 6)
- **Frame skip e a contagem de passos/episódios** — com `frame_skip=3`, o episódio de 45 s tem ~450 decisões; os testes de duração e o `steps` do CSV devem refletir isso. (Task 7)

## File Structure

- `dqn/checkpoint.py` — checkpoints pesos-only (remove buffer/RNG/env_state).
- `train.py` — rampa de epsilon por episódio, warmup, snapshots de orçamento, `--envs`, loop vetorizado.
- `evaluate.py` — ganha `--opponent`; mantém avaliação gulosa.
- `dqn/agent.py` — warmup, `double`, `n_step`, PER integrado, `act_batch`, helper `_td_targets`.
- `dqn/replay_buffer.py` — n-step no `sample()`, `PrioritizedReplayBuffer`.
- `dqn/model.py` — cabeça dueling.
- `arena/config.py` — `frame_stack`, `frame_skip`.
- `arena/environment.py` — pilha de frames e repetição de ação no `step()`.
- `arena/vector.py` (novo) — fábrica de `VectorEnv` para o treino.
- `tests/test_checkpoint.py` (novo) — testes das Tasks 1–3.
- `tests/test_arena.py` — testes de env existentes adaptados/estendidos (Tasks 6–7).
- `tests/test_agent.py` (novo) — teste das Tasks 4–5, 10–12.

---

### Task 1: Checkpoint pesos-only

**Files:**
- Modify: `dqn/checkpoint.py` (todo o arquivo)
- Modify: `train.py:243-252,258-260` (chamadas a `save_checkpoint` sem `env=`)
- Test: `tests/test_checkpoint.py` (criar)

**Interfaces:**
- Consumes: nada novo.
- Produces: `save_checkpoint(path, agent, config=None, extra=None, episodes=None)` (sem `env`); `load_checkpoint(path, agent, resume=False, map_location=None)` (sem `env`); `ARCH_VERSION = "cnn-dqn-v2"`.

- [ ] **Step 1: Escrever o teste que falha**

```python
def test_checkpoint_lightweight(tmp_path):
    agent = DQNAgent((5, 32, 20), 12)
    path = tmp_path / "c.pt"
    save_checkpoint(path, agent, extra={"a": 1}, episodes=5)
    payload = torch.load(path, weights_only=False)
    assert "buffer" not in payload and "env_state" not in payload and "rng_state" not in payload
    assert payload["episodes"] == 5
    agent2 = DQNAgent((5, 32, 20), 12)
    load_checkpoint(path, agent2, resume=True)  # sem env
    assert agent2.steps == agent.steps
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_checkpoint.py::test_checkpoint_lightweight -v`
Expected: FAIL (assinatura com `env`/payload ainda tem `buffer`).

- [ ] **Step 3: Implementar em `dqn/checkpoint.py`**

Remover `capture_rng_state`/`restore_rng_state` e os campos `buffer`, `rng_state`, `env_state` do payload; subir `ARCH_VERSION` para `"cnn-dqn-v2"`; `load_checkpoint(resume=True)` restaura otimizador, `steps`, `epsilon`, `gamma` e põe `online.train()`; atualizar as chamadas de `train.py` (remover `env=env`).

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_checkpoint.py::test_checkpoint_lightweight -v`
Expected: PASS

- [ ] **Step 5: Rodar a suíte inteira**

Run: `TORCH_CPP_LOG_LEVEL=ERROR venv/bin/python -m pytest -q`
Expected: falham APENAS os testes que dependem da retomada bit a bit (ex.: `test_resume_is_bit_identical`) — remova/reescreva-os neste passo (virar teste de "resume carrega pesos e episódios"). Atualizar também o README: na seção "Checkpoints e retomada", remover a promessa de retomada bit a bit e descrever o novo formato (pesos/otimizador/contador).

- [ ] **Step 6: Commit**

```bash
git add dqn/checkpoint.py train.py tests/test_checkpoint.py tests/test_arena.py README.md
git commit -m "feat: checkpoints pesos-only (arquitetura v2, retomada não é mais bit a bit)"
```

---

### Task 2: `evaluate.py --opponent` e snapshots de orçamento

**Files:**
- Modify: `evaluate.py`
- Modify: `train.py` (helper de orçamentos + chamada no fim do episódio)
- Test: `tests/test_checkpoint.py`

**Interfaces:**
- Consumes: Task 1 (`save_checkpoint` pesos-only).
- Produces: `evaluate.py --opponent {rule,random,coins}` (default `rule`); `train.py::budget_snapshots(total_episodes) -> set[int]`; snapshots em `checkpoints/budget/ep<NNN>.pt` só com pesos.

- [ ] **Step 1: Escrever o teste que falha**

```python
def test_evaluate_opponent_flag():
    from evaluate import main
    # parse_args com ["--opponent", "coins"] devolve opponent="coins"
    # (refatore o argparse para uma função parse_args(argv) testável)

def test_budget_snapshots():
    from train import budget_snapshots
    assert budget_snapshots(300) == {50, 100, 150, 300}
    assert budget_snapshots(100) == {50, 100}
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_checkpoint.py::test_evaluate_opponent_flag tests/test_checkpoint.py::test_budget_snapshots -v`
Expected: FAIL

- [ ] **Step 3: Implementar**

`evaluate.py`: flag `--opponent` (choices `rule/random/coins`, default `rule`) passada a `ArenaEnv(..., opponent_policy=args.opponent)`; extrair `parse_args(argv)` testável. `train.py`: `budget_snapshots(total) = {b for b in (50, 100, 150, 300) if b <= total}`; ao fim de cada episódio com `ep in budgets` → `save_checkpoint(ckpt_dir / "budget" / f"ep{ep:03d}.pt", agent, cfg, {"episode": ep}, episodes=ep)`.

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_checkpoint.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluate.py train.py tests/test_checkpoint.py
git commit -m "feat: evaluate.py com --opponent e snapshots de orçamento (checkpoints/budget)"
```

---

### Task 3: `AutoOpponent` pula checkpoints incompatíveis

**Files:**
- Modify: `train.py` (`_checkpoints`, novo helper `_try_load_checkpoint`, `choose`)
- Test: `tests/test_checkpoint.py`

**Interfaces:**
- Consumes: Task 1 (`load_checkpoint` novo).
- Produces: `train.py::_try_load_checkpoint(path, agent) -> bool` (True se carregou; False com `warnings.warn` em qualquer exceção de `load_checkpoint`).

- [ ] **Step 1: Escrever o teste que falha**

```python
def test_try_load_skips_incompatible(tmp_path):
    agent = DQNAgent((5, 32, 20), 12)
    junk = tmp_path / "old.pt"
    junk.write_bytes(b"lixo")  # não é um checkpoint válido
    with pytest.warns(UserWarning):
        assert _try_load_checkpoint(junk, agent) is False
    # e um checkpoint válido (Task 1) carrega: True
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_checkpoint.py::test_try_load_skips_incompatible -v`
Expected: FAIL

- [ ] **Step 3: Implementar**

`_try_load_checkpoint` embrulha `load_checkpoint` em `try/except Exception` com `warnings.warn`; `choose()` usa o helper para `checkpoint_agent`/`alt_agent` e pula o arquivo quando retorna False; `_checkpoints()` continua glob não-recursivo de `*.pt` (subdiretório `budget/` fica fora — teste pina com um arquivo em `tmp/budget/ep050.pt` não aparecendo na lista).

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_checkpoint.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_checkpoint.py
git commit -m "fix: adversário auto pula .pt incompatíveis com aviso"
```

**Gate Fase 0:** rodar a suíte inteira (`TORCH_CPP_LOG_LEVEL=ERROR venv/bin/python -m pytest -q`) — tudo verde; e um treino curto de sanidade (`venv/bin/python train.py --episodes 2 --checkpoint-dir /tmp/fase0 --opponent random`) termina sem erro.

---

### Task 4: Rampa de epsilon por episódio (sem decay no agente)

**Files:**
- Modify: `dqn/agent.py` (remover decay de `learn()`)
- Modify: `train.py` (helper + atribuição por episódio)
- Test: `tests/test_agent.py` (criar)

**Interfaces:**
- Consumes: Task 1.
- Produces: `train.py::epsilon_for_episode(ep, ramp_end, floor) -> float`; `DQNAgent.epsilon` é só um atributo (o agente não o altera).

- [ ] **Step 1: Escrever o teste que falha**

```python
def test_epsilon_for_episode():
    assert epsilon_for_episode(1, 60, 0.1) == pytest.approx(1.0 - (1 - 0.1) / 60 * 1, abs=1e-9)
    assert epsilon_for_episode(60, 60, 0.1) == pytest.approx(0.1)
    assert epsilon_for_episode(61, 60, 0.1) == pytest.approx(0.1)

def test_learn_does_not_decay_epsilon():
    agent = DQNAgent((5, 32, 20), 12, epsilon=0.5)
    # encher buffer com 2000 transições sintéticas (obs zeros, ação 0, r 0)
    agent.learn()
    assert agent.epsilon == 0.5
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_agent.py -v`
Expected: FAIL

- [ ] **Step 3: Implementar**

`train.py`: `ramp_end = max(1, round(args.episodes * 0.2))`; no início de cada episódio `agent.epsilon = epsilon_for_episode(ep, ramp_end, 0.1)` (ep é o número absoluto, então retomadas continuam a rampa). Remover `self.epsilon = max(...)` de `learn()` e o parâmetro `epsilon_decay` (manter `epsilon`, `epsilon_min=0.1`).

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_agent.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dqn/agent.py train.py tests/test_agent.py
git commit -m "feat: exploração com rampa linear por episódio (20% do horizonte)"
```

---

### Task 5: Warmup de aprendizado

**Files:**
- Modify: `dqn/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: Task 4.
- Produces: `DQNAgent(..., warmup=1000)`; `learn()` devolve `None` enquanto `len(buffer) < warmup`.

- [ ] **Step 1: Escrever o teste que falha**

```python
def test_warmup():
    agent = DQNAgent((5, 32, 20), 12, warmup=1000)
    for _ in range(999):
        agent.buffer.add(np.zeros((5, 32, 20)), 0, 0.0, np.zeros((5, 32, 20)), False, False)
    assert agent.learn() is None
    agent.buffer.add(np.zeros((5, 32, 20)), 0, 0.0, np.zeros((5, 32, 20)), False, False)
    assert agent.learn() is not None  # 1000 transições
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_agent.py::test_warmup -v`
Expected: FAIL

- [ ] **Step 3: Implementar**

Em `learn()`, antes de amostrar: `if len(self.buffer) < max(self.warmup, self.batch_size): return None`.

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_agent.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dqn/agent.py tests/test_agent.py
git commit -m "feat: warmup de 1000 transições antes de aprender"
```

**Gate Fase 1 (A/B):** rodar `venv/bin/python train.py --episodes 150 --opponent random --seed {7,42,123}` (3 corridas, com snapshots de orçamento) e avaliar cada `checkpoints/budget/ep150.pt` com `evaluate.py --checkpoint ... --episodes 20 --seed 1000`; comparar a média das 3 seeds com o baseline (mesmas seeds, código da Fase 0) — avançar se ≥ 15% melhor no orçamento 150; registrar tabela no commit message body.

---

### Task 6: Frame stack na observação

**Files:**
- Modify: `arena/config.py` (`frame_stack: int = 1`)
- Modify: `arena/environment.py`
- Test: `tests/test_arena.py`

**Interfaces:**
- Consumes: nada novo.
- Produces: `ArenaConfig.frame_stack`; `ArenaEnv` com obs `(5 * frame_stack, H, W)` (ordem do frame mais antigo para o mais novo); `_observation()` devolve o frame único atual; a pilha é preenchida no `reset` com cópias do primeiro frame.

- [ ] **Step 1: Escrever o teste que falha**

```python
def test_frame_stack_observation():
    env = ArenaEnv(ArenaConfig(frame_stack=4))
    obs, _ = env.reset(seed=1)
    assert obs.shape == (20, 32, 20)
    assert np.array_equal(obs[:5], obs[5:10]) and np.array_equal(obs[15:], obs[:5])  # pilha cheia de cópias
    obs2, *_ = env.step(1)
    assert np.array_equal(obs2[:5], obs[-5:])  # frame antigo virou o mais antigo
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_arena.py::test_frame_stack_observation -v`
Expected: FAIL

- [ ] **Step 3: Implementar**

`ArenaEnv.__init__`: `self._frames = deque(maxlen=cfg.frame_stack)`; `_observation()` inalterado (frame único); novo `_stacked()` concatena `self._frames` no eixo 0; `reset` e `step` devolvem `_stacked()`; `reset` preenche com o primeiro frame repetido; `step` faz `append` do frame pós-passo.

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_arena.py -v`
Expected: PASS (testes antigos com `frame_stack=1` default continuam)

- [ ] **Step 5: Commit**

```bash
git add arena/config.py arena/environment.py tests/test_arena.py
git commit -m "feat: frame stack na observação (ArenaConfig.frame_stack)"
```

---

### Task 7: Frame skip no ambiente

**Files:**
- Modify: `arena/config.py` (`frame_skip: int = 1`)
- Modify: `arena/environment.py` (`step`)
- Test: `tests/test_arena.py`

**Interfaces:**
- Consumes: Task 6.
- Produces: `ArenaConfig.frame_skip`; `step(action)` repete a ação do jogador por `frame_skip` substeps (o adversário decide a cada substep com a obs corrente), soma recompensas e avança `elapsed` de `frame_skip / fps`.

- [ ] **Step 1: Escrever o teste que falha**

```python
def test_frame_skip_repeats_action():
    env = ArenaEnv(ArenaConfig(frame_skip=3))
    env.reset(seed=1)
    t0 = env.elapsed
    # moeda posta na posição do personagem antes do step (padrão dos testes:
    # manipular o engine diretamente): a coleta deve acontecer nos substeps
    c = env.engine.characters[0]
    env.engine.coins[0].x, env.engine.coins[0].y = c.x, c.y
    obs, reward, *_ = env.step(1)
    assert env.elapsed - t0 == pytest.approx(3 / env.cfg.fps)
    assert reward == pytest.approx(env.cfg.coin_reward)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_arena.py::test_frame_skip_repeats_action -v`
Expected: FAIL

- [ ] **Step 3: Implementar**

`step()`: loop de `frame_skip` iterações chamando `self.engine.step(actions, dt)` (recalculando a ação do oponente a cada substep via `_opponent_action`) e acumulando `reward`/`info` (campos `collected`/`deaths`/`eliminations` do `info` são os valores finais do personagem, não somas); a obs devolvida é a do último substep, anexada à pilha.

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_arena.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add arena/config.py arena/environment.py tests/test_arena.py
git commit -m "feat: frame skip no ambiente (ArenaConfig.frame_skip)"
```

---

### Task 8: `obs_to_image` com obs empilhada

**Files:**
- Modify: `train.py` (`obs_to_image`)
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: Task 6.
- Produces: `obs_to_image(obs)` usa apenas os últimos 5 canais (`obs[-5:]`).

- [ ] **Step 1: Escrever o teste que falha**

```python
def test_obs_to_image_uses_last_frames():
    obs = np.zeros((20, 32, 20), dtype=np.float32)
    obs[:5, 0, 0] = 1.0   # frame antigo com ruído
    obs[-5:, 1, 1] = 1.0  # frame atual com moeda
    img = obs_to_image(obs)
    assert np.array_equal(img, obs_to_image(obs[-5:]))
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_agent.py::test_obs_to_image_uses_last_frames -v`
Expected: FAIL

- [ ] **Step 3: Implementar**

Em `obs_to_image`, operar sobre `obs[-5:]`.

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_agent.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_agent.py
git commit -m "fix: TensorBoard mostra o frame atual da pilha"
```

**Gate Fase 2 (A/B):** mesma receita da Fase 1, com `ArenaConfig(frame_stack=4, frame_skip=3)` aplicado em `train.py` via config (valores definidos em `main()`: `cfg.frame_stack, cfg.frame_skip = 4, 3`); avaliar e comparar com a Fase 1 — avançar se ≥ 15% melhor.

---

### Task 9: n-step no replay buffer

**Files:**
- Modify: `dqn/replay_buffer.py` (`sample`)
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: Task 1.
- Produces: `ReplayBuffer.sample(batch_size, n=1, gamma=0.99)` devolve `(s, a, r_n, ns, term)` com `r_n = Σ_{k=0..n-1} γ^k·r_{t+k}` (acumulação para em `terminated` ou `truncated` na transição t+k, usando como `ns`/`term` o par dessa posição).

- [ ] **Step 1: Escrever o teste que falha**

```python
def test_sample_n_step():
    buf = ReplayBuffer(10)
    # 4 transições contíguas, sem terminais: r = 1, 2, 3, 4; estados s0..s3
    ...
    s, a, r_n, ns, term = buf.sample(1, n=3, gamma=0.99)
    assert r_n[0] == pytest.approx(1 + 0.99 * 2 + 0.99**2 * 3)
    assert np.array_equal(ns[0], s3)
    # com terminated na transição 1: r_n == r0 e ns == s1
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_agent.py::test_sample_n_step -v`
Expected: FAIL

- [ ] **Step 3: Implementar**

`sample`: sortear `batch_size` índices; para cada índice, percorrer até `n` transições à frente parando em `terminated or truncated`; devolver arrays empilhados como hoje. Amostragem continua via `random.sample` (módulo `random` global).

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_agent.py::test_sample_n_step -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dqn/replay_buffer.py tests/test_agent.py
git commit -m "feat: retornos n-step no replay buffer"
```

---

### Task 10: Double DQN com helper `_td_targets`

**Files:**
- Modify: `dqn/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: Task 9.
- Produces: `DQNAgent(..., double=True, n_step=1)`; `_td_targets(online, target, s, a, r_n, ns, term, gamma) -> Tensor` (com `double`: ação de `online(ns).argmax`, valor de `target(ns).gather`; bootstrap zerado quando `term==1`); `learn()` usa o helper.

- [ ] **Step 1: Escrever o teste que falha**

```python
def test_td_targets_double():
    # redes minúsculas com pesos fixos: online favorece a ação 2 em ns,
    # target tem valores conhecidos por ação
    # double=True: target == r_n + gamma * target(ns)[2]
    # double=False: target == r_n + gamma * max(target(ns))
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_agent.py::test_td_targets_double -v`
Expected: FAIL

- [ ] **Step 3: Implementar**

Extrair o cálculo do alvo de `learn()` para `_td_targets` (parâmetro `double`); `learn()` amostra com `self.buffer.sample(self.batch_size, n=self.n_step, gamma=self.gamma)` e usa o helper; remover o uso direto de `truncated` (a amostra já clampa).

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_agent.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dqn/agent.py tests/test_agent.py
git commit -m "feat: Double DQN (alvo com ação escolhida na rede online)"
```

**Gate Fase 3 (A/B):** `DQNAgent(..., double=True, n_step=3)` em `train.py`; mesma receita de A/B — avançar se ≥ 15% melhor que a Fase 2.

---

### Task 11: Cabeça dueling

**Files:**
- Modify: `dqn/model.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: Task 10.
- Produces: `DQN(input_shape, n_actions, dueling=False)`; com `dueling=True`, `head` vira `V + (A − média(A))` (dois ramos lineares de 128 após o flatten, expostos como `self.v_head`/`self.a_head`).

- [ ] **Step 1: Escrever o teste que falha**

```python
def test_dueling_equals_v_plus_advantage():
    net = DQN((5, 32, 20), 12, dueling=True)
    x = torch.randn(1, 5, 32, 20)
    feats = net.features(x)
    v = net.v_head(feats)
    a = net.a_head(feats)
    expected = v + a - a.mean(dim=1, keepdim=True)
    assert torch.allclose(net(x), expected)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_agent.py::test_dueling_equals_v_plus_advantage -v`
Expected: FAIL

- [ ] **Step 3: Implementar**

Em `DQN.__init__`, quando `dueling=True`, substituir o `head` sequencial por `v_head`/`a_head` (cada um `Linear(128) -> ...`) e `forward = v + a − média(a)`; `dueling=False` mantém o comportamento atual exato.

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_agent.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dqn/model.py tests/test_agent.py
git commit -m "feat: cabeça dueling no DQN"
```

---

### Task 12: Replay priorizado (rank-based)

**Files:**
- Modify: `dqn/replay_buffer.py` (novo `PrioritizedReplayBuffer`)
- Modify: `dqn/agent.py` (integração)
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: Tasks 9–10.
- Produces: `PrioritizedReplayBuffer(capacity)` com `add(state, action, reward, next_state, terminated, truncated, priority=None)`; `sample(batch_size, n=1, gamma=0.99) -> (s, a, r_n, ns, term, indices)` com pesos `∝ 1/rank(prioridade)`; `update_priorities(indices, errors)` (erro → prioridade = |erro| + 1e-6); `DQNAgent(..., per=False)`: com `per=True`, `learn()` usa o buffer priorizado, calcula erros `|target − q|` e chama `update_priorities` com os índices amostrados.

- [ ] **Step 1: Escrever o teste que falha**

```python
def test_per_favors_high_error():
    buf = PrioritizedReplayBuffer(100)
    # 50 transições com prioridade baixa, 1 com prioridade alta (erro grande)
    random.seed(0)
    _, _, _, _, _, indices = buf.sample(50)
    # com rank-based, a transição de erro alto aparece entre as amostradas
    assert high_index in indices
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_agent.py::test_per_favors_high_error -v`
Expected: FAIL

- [ ] **Step 3: Implementar**

`PrioritizedReplayBuffer`: `self.priorities = deque(maxlen)`; no `sample`, pesos `1/(rank+1)` via `random.choices(..., weights=...)`; `add` com `priority=None` recebe a prioridade máxima atual (novas transições amostradas primeiro); `update_priorities` atualiza por índice. Rank-based não usa agendamentos α/β (sem annealing — desvio consciente da menção no spec). Em `learn()` com `per=True`, guardar `self._last_indices` do `sample` e atualizar prioridades com `|target − q|.detach().cpu().numpy()`.

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_agent.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dqn/replay_buffer.py dqn/agent.py tests/test_agent.py
git commit -m "feat: replay priorizado rank-based (PER)"
```

**Gate Fase 4 (A/B):** `DQNAgent(..., per=True)` + `DQN(..., dueling=True)` + varredura (lr 2.5e-4 vs 1e-3, batch 64/128, target_update 500/1000 vs soft τ=0.005) — triagem 1 seed/100 eps por ponto do grid, gate 3 seeds/150 eps para o melhor ponto; avançar se ≥ 15% melhor que a Fase 3. Registrar a tabela do grid no corpo do commit.

---

### Task 13: Treino vetorizado (Fase 5)

**Files:**
- Create: `arena/vector.py`
- Modify: `train.py` (flag `--envs`, loop vetorizado, contagem de episódios)
- Modify: `dqn/agent.py` (`act_batch`)
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: Tasks 6–7 (env), Task 12 (agente).
- Produces: `arena/vector.py::make_vector_env(n_envs, cfg, opponent) -> gym.VectorEnv` (AsyncVectorEnv de `ArenaEnv`); `DQNAgent.act_batch(states) -> np.ndarray` (ações por linha, greedy ou epsilon conforme `greedy`); `train.py --envs N` (default 1); com `N>1`: ações em lote, `buffer.add_batch`, `learn()` a cada `K=4` passos do vetor, episódios contados pela soma de `terminated|truncated` por worker, snapshots de orçamento pelo total acumulado, seeds dos workers = `seed + i`.

- [ ] **Step 1: Escrever o teste que falha**

```python
def test_vector_env_and_act_batch():
    cfg = ArenaConfig()
    venv = make_vector_env(2, cfg, "random")   # SyncVectorEnv no modo de teste
    obs, _ = venv.reset(seed=7)
    assert obs.shape == (2, 5, 32, 20)
    agent = DQNAgent((5, 32, 20), 12)
    acts = agent.act_batch(obs, greedy=True)
    assert acts.shape == (2,)
    obs, rew, term, trunc, info = venv.step(acts)
    assert rew.shape == (2,)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_agent.py::test_vector_env_and_act_batch -v`
Expected: FAIL

- [ ] **Step 3: Implementar**

`make_vector_env(n_envs, cfg, opponent, asynchronous=True)`: `AsyncVectorEnv` de `ArenaEnv` (padrão, usado pelo treino); com `asynchronous=False` usa `SyncVectorEnv` (usado pelos testes). `act_batch(states, greedy=False)`: tensor em lote → `argmax` por linha (ou sorteio por linha via módulo `random` quando não guloso). `train.py`: loop vetorizado (obs/actions/rewards/dones em lote); para a transição de fim de episódio usar a `final_observation` dos `infos` quando disponível; `learn()` a cada 4 passos do vetor.

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_agent.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add arena/vector.py train.py dqn/agent.py tests/test_agent.py
git commit -m "feat: treino vetorizado com --envs N (Fase 5)"
```

**Gate Fase 5 (A/B):** adversário fixo `random`; benchmark passos/segundo antes (`--envs 1`) e depois (`--envs 8`); A/B de qualidade com `--envs 8` (3 seeds/150 eps) — G2 pede ≥ 3× de velocidade de parede por episódio (incluindo escrita de checkpoint).

---

### Task 14: Rodada final de medição e relatório

**Files:**
- Nenhum (operacional).

- [ ] **Step 1: Avaliar todas as fases nos 4 orçamentos**

Run: para cada snapshot `checkpoints/budget/ep*.pt` de cada fase/seed: `TORCH_CPP_LOG_LEVEL=ERROR venv/bin/python evaluate.py --checkpoint <p> --episodes 20 --seed 1000`

- [ ] **Step 2: Montar a tabela comparativa**

Colunas: fase, seed, recompensa média em 50/100/150/300 eps, tempo de parede por episódio; registrar em `docs/resultados-fases.md` e no relatório final.

- [ ] **Step 3: Verificar G1/G2 e decidir Fase 6**

G1 (≥2× vs baseline em 150 eps, média das seeds) e G2 (≥3× mais rápido) — se G1 falhar, propor candidatas da Fase 6 (reward shaping potencial, assíncrono) ao autor.

- [ ] **Step 4: Commit do relatório**

```bash
git add docs/resultados-fases.md
git commit -m "docs: resultados do A/B das fases 0-5"
```

---
