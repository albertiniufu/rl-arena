# Risco — jogo de conquista inspirado em War/Risk

Jogo de conquista de territórios com **fases de reforço, ataque e
fortificação**, **combate por dados** e vitória por domínio total. Inclui
motor puro, bot heurístico, ambiente Gymnasium e modo jogável em texto.

## Como rodar

A partir da raiz do repositório:

```bash
python -m games.risk.play                 # humano (J1) contra bot
python -m games.risk.play --players 4     # humano contra 3 bots
python -m games.risk.play --demo          # bots contra bots
pytest -q games/risk
```

Comandos no modo humano:

```
p <territorio>          reforçar (colocar 1 exército)
a <origem> <destino>    atacar um vizinho
f <origem> <destino>    fortificar (mover 1 exército)
fim                     encerrar a fase
ajuda / sair
```

Nomes de territórios aceitam prefixo único (ex.: `ala` → `Alaska`).

## Regras

- Tabuleiro: 6 continentes × 4 territórios = 24 territórios (ver
  `board.py`). Cada continente dá um bônus de reforço.
- Cada jogador recebe `initial_armies` exércitos no início (1 por território
  + distribuição aleatória do restante).
- Reforço por turno: `max(3, territórios // 3)` + bônus de continentes.
- Ataque: origem com 2+ exércitos, destino vizinho inimigo. Atacante rola
  até 3 dados, defensor até 2; dados ordenados são comparados e empate é do
  defensor. O atacante ocupa o território conquistado com os dados vencedores
  (deixando ao menos 1 exército na origem).
- Fortificação: um movimento entre territórios próprios vizinhos por turno.
- Vence quem dominar todos os territórios.

## Ambiente Gymnasium

`games/risk/env.py` expõe `RiskEnv`:

- o agente controla o jogador 0; os demais são `RiskBot`;
- **ação** `Discrete(n² + n + 1)` (ver `RiskGame.decode`):
  - `0` encerra a fase;
  - `1..n` coloca um exército no território `ação-1`;
  - `n+1..n+n²` move/ataca de `src` para `dst` (`src = (ação-n-1) // n`);
- **observação**: one-hot do dono + exércitos normalizados por território,
  one-hot da fase, reforços restantes, se é o turno do agente e o turno;
- `action_masks()` devolve a máscara booleana de ações legais (compatível
  com `sb3-contrib` `ActionMasker`);
- recompensa: variação da fração de territórios do agente, `+10` por vitória
  e `-10` por derrota;
- `reset → (obs, info)` e `step → (obs, reward, terminated, truncated, info)`.

`max_turns` trunca partidas longas. O bot pode ser trocado passando
`opponent=...` ao construtor.

## Estrutura

| Arquivo | Papel |
| --- | --- |
| `board.py` | Territórios, continentes, bônus e adjacências. |
| `game.py` | `RiskGame`: estado, ações, combate e vitória. |
| `agents.py` | `RiskBot`: reforço de fronteira, ataque e fortificação. |
| `env.py` | `RiskEnv`: ambiente Gymnasium com máscara de ações. |
| `play.py` | Modo jogável em texto e demo bot × bot. |
| `tests/test_risk.py` | Tabuleiro, regras, combate, bot e contrato Gym. |
