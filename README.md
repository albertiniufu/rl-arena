# Games

Coleção de jogos e simuladores para aprendizagem por reforço e estudo de
sistemas. Cada jogo vive em `games/<nome>/` com seu próprio README, código,
testes e entry points.

| Jogo | Diretório | Descrição |
| --- | --- | --- |
| RL Arena | [`games/default/`](games/default/README.md) | Plataforma 2D com DQN, Double DQN/PER, PPO, A2C, IPPO e MAPPO. |
| Economia | [`games/economy/`](games/economy/README.md) | Banco-agente: concessão e precificação por GH com recompensa por RAROC líquido, sobre uma economia com população, empresas, governo e choques macro. Modo governo disponível. |
| Risco | [`games/risk/`](games/risk/README.md) | Jogo de conquista de territórios inspirado em War/Risk, com motor de fases e combate por dados. |

## Instalação

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Testes

```bash
pytest -q                 # todos os jogos
pytest -q games/risk      # um jogo
```

## Jogos novos

Crie `games/<nome>/` com um pacote Python próprio, um `README.md`, testes em
`games/<nome>/tests/` e, se fizer sentido, um `env.py` compatível com
Gymnasium. O `conftest.py` da raiz já coloca a raiz e o jogo default no
`sys.path`; para os demais jogos use imports absolutos (`games.economy...`).
