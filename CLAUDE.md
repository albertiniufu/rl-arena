# CLAUDE.md

Monorepo de jogos/simuladores. Cada jogo tem o seu próprio `CLAUDE.md` com os
detalhes de arquitetura e comandos; o Claude Code carrega o arquivo mais
próximo ao diretório de trabalho.

## Estrutura

- `games/default/` — o RL Arena original (plataforma 2D, `arena/`, `agents/`,
  `algs/`, `train.py`, `evaluate.py`, `play.py`, testes). Leia
  [`games/default/CLAUDE.md`](games/default/CLAUDE.md) antes de mexer nele.
- `games/economy/` — simulador econômico (população, empresas, governo,
  variabilidade macro) + ambiente Gymnasium + demo em texto.
- `games/risk/` — jogo de conquista inspirado em War/Risk (territórios,
  fases, combate por dados) + ambiente Gymnasium + modo jogável em texto.
- `conftest.py` (raiz) — ajusta `sys.path` para os testes de todos os jogos.
- `requirements.txt` (raiz) — dependências comuns (numpy, gymnasium, torch,
  pygame, pytest, tensorboard, imageio).

## Comandos

```bash
pytest -q                     # suíte completa
pytest -q games/economy       # um jogo
cd games/default && python train.py --episodes 300
python -m games.economy.demo --steps 24
python -m games.risk.play
```

Ao rodar o jogo default, entre em `games/default/` (ou passe
`--checkpoint-dir`/caminhos absolutos) para que `checkpoints/` e `logs/`
fiquem junto do jogo.

## Convenções

- Documentação e comentários em português.
- Cada jogo é autocontido; dependências pesadas de RL ficam no default.
- Novos ambientes: contrato Gymnasium (`reset` → `(obs, info)`, `step` →
  `(obs, reward, terminated, truncated, info)`), RNG injetável e seed
  reprodutível.
