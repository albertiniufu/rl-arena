# RL Arena — plataforma 2D mínima para DQN

Projeto didático com Python, NumPy, Gymnasium, PyTorch e Pygame. O mapa é fixo; o agente principal aprende com DQN e o adversário usa uma política simples. A observação é uma grade semântica channels-first `(5, H, W)`.

## Instalação

Recomendado: Python 3.11 ou 3.12.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Testes

```bash
pytest -q
```

## Treinar

Sem janela gráfica (padrão):

```bash
python train.py --episodes 300
```

Com visualização:

```bash
python train.py --episodes 100 --render
```

Escolher adversário aleatório:

```bash
python train.py --episodes 300 --opponent random
```

O adversário pode ser `rule` (padrão: aproxima do jogador e às vezes empurra), `random` (sorteia entre as 6 primeiras ações) ou `coins` (persegue a moeda visível):

```bash
python train.py --episodes 300 --opponent coins
```

As políticas ficam em `agents/baseline.py` e são resolvidas pelo environment por nome.

Continuar treinamento a partir de checkpoint:

```bash
python train.py --episodes 300 --resume checkpoints/latest.pt
```

O argumento `--episodes` indica quantos episódios adicionais serão executados. Os checkpoints são salvos em `checkpoints/latest.pt` e `checkpoints/best.pt`; métricas em `logs/training.csv`.

## Avaliação sem atualizar pesos

```bash
python evaluate.py --checkpoint checkpoints/best.pt --episodes 20
```

Com visualização:

```bash
python evaluate.py --checkpoint checkpoints/best.pt --episodes 5 --render
```

## Reproduzir uma política

```bash
python play.py --checkpoint checkpoints/best.pt --render
```

Sem `--render`, o script executa os episódios sem abrir janela.

Também dá para dispensar o checkpoint e controlar o jogador com uma política baseline — por exemplo, `coins` contra `coins` (ambos perseguem moedas, sem empurrões):

```bash
python play.py --player coins --opponent coins --render
```

O `--player` aceita `coins`, `random` ou `human`; sem ele, vale o comportamento padrão (checkpoint DQN guloso).

Para jogar no teclado contra um adversário não humano (a janela abre automaticamente):

```bash
python play.py --player human --opponent coins
```

| Tecla | Ação |
|---|---|
| ← → ou A D | Andar |
| Espaço / ↑ / W | Pular (diagonal se estiver andando) |
| Q | Empurrar para a esquerda (força máxima) |
| E | Empurrar para a direita (força máxima) |

## Visual (skins)

`train.py`, `evaluate.py` e `play.py` aceitam `--skin` para escolher o renderizador:

```bash
python play.py --player human --opponent coins --skin neon
```

Cada skin é um arquivo `arena/renderer_<skin>.py` que define uma classe `ArenaRenderer`; o clássico é `arena/renderer.py`. As opções de `--skin` são descobertas automaticamente pelos arquivos existentes — para criar uma skin nova (ex.: `arena/renderer_dinosaur.py`), basta adicionar o arquivo, sem tocar em mais nenhum código.

## Ações

| Índice | Ação |
|---:|---|
| 0 | Não fazer nada |
| 1–2 | Esquerda / direita |
| 3 | Pular verticalmente |
| 4–5 | Pular diagonal esquerda / direita |
| 6–8 | Empurrar adversário para esquerda com força crescente |
| 9–11 | Empurrar adversário para direita com força crescente |

## Observações e limitações da versão didática

- A simulação usa tempo discreto de `1 / fps` segundos.
- O cenário e a configuração inicial são fixos.
- O episódio termina pelo limite de tempo (truncamento), nunca apenas por morte. Ao morrer, o personagem perde 10 moedas (saldo mínimo zero), fica inativo por 10 segundos simulados e reaparece sobre uma plataforma escolhida aleatoriamente.
- A observação é uma grade semântica simples. Velocidades e temporizadores não estão explicitamente codificados, portanto a observação não é totalmente Markoviana; isso é uma simplificação para a primeira experiência com CNN.
- O crédito por eliminações é deliberadamente simplificado.
- A retomada (`--resume`) é reproduzível bit a bit: o checkpoint persiste pesos, rede-alvo, otimizador, passos, epsilon, replay buffer, o estado de todos os RNGs (Python, NumPy, Torch, engine e ambiente) e um snapshot do ambiente. Com os mesmos argumentos (em especial o mesmo `--seed`), retomar produz exatamente os mesmos episódios, pesos e arquivos de checkpoint de um treino contínuo. Checkpoints antigos, sem esses campos, ainda carregam (sem reproduzibilidade).

## Estrutura

```text
rl-arena/
├── arena/       # configuração, motor, ambiente Gymnasium e renderizador
├── agents/      # políticas baseline usadas como adversário pelo environment
├── dqn/         # CNN, replay buffer, agente e checkpoints
├── tests/
├── train.py
├── evaluate.py
├── play.py
├── requirements.txt
└── README.md
```
