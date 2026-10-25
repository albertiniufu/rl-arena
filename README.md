# RL Arena — plataforma 2D mínima para DQN

Projeto didático para estudar Reinforcement Learning na prática: um agente DQN (PyTorch + Gymnasium + NumPy + Pygame) treina num mapa 2D fixo para coletar moedas e empurrar um adversário de política simples. Este guia leva você do zero até plugar o seu próprio modelo e o seu próprio algoritmo de RL no lugar do DQN.

## Demonstração

[![Dois personagens coletando moedas no mapa fixo (10 s do demo.mp4)](demo.gif)](demo.mp4)

O gif mostra os primeiros 10 segundos; o vídeo completo está em [`demo.mp4`](demo.mp4) (clique para abrir o player).

## Guia de uso

Ambiente: Linux. Pré-requisitos: Python 3.10 ou mais novo e familiaridade básica com terminal.

### 1. Instalar

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Treinar do zero

```bash
python train.py --episodes 300
```

Treina sem abrir janela (padrão). Ao final, os pesos ficam em `checkpoints/latest.pt` e `checkpoints/best.pt`; as métricas por episódio, em `logs/training.csv`.

Dica: se o terminal for inundado por avisos do NNPACK, rode com `TORCH_CPP_LOG_LEVEL=ERROR python train.py ...`.

### 3. Assistir o treino

```bash
python train.py --episodes 100 --render
```

Abre a janela do Pygame mostrando o agente aprendendo (fechar a janela encerra o processo).

### 4. Avaliar sem atualizar pesos

```bash
python evaluate.py --checkpoint checkpoints/best.pt --episodes 20
```

Executa a política gulosa (sem explorar nem atualizar pesos) e grava `logs/evaluation.csv`. Com `--render`, mostra a janela.

### 5. Jogar e assistir

```bash
python play.py --checkpoint checkpoints/best.pt --render
```

Sem `--render`, executa os episódios sem abrir janela.

Também dá para dispensar o checkpoint e rodar políticas baseline, por exemplo `coins` contra `coins` (ambos perseguem moedas, sem empurrões):

```bash
python play.py --player coins --opponent coins --render
```

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

### 6. Experimentar

- Adversário: `--opponent random`, `coins` ou `rule` (o padrão `auto` é um currículo — veja [Adversários](#adversários)).
- Semente: `--seed N` (treinos com a mesma semente são idênticos).
- Mundo: física, recompensas, duração do episódio e contagens ficam em `arena/config.py`.

### 7. Seu próprio modelo e seu próprio algoritmo de RL

O DQN está em duas peças separadas:

- `dqn/model.py` — `DQN(nn.Module)`: a CNN que recebe a observação `(5, H, W)` e devolve 12 Q-valores.
- `dqn/agent.py` — `DQNAgent`: o algoritmo — replay buffer, rede-alvo, epsilon-greedy, Adam + SmoothL1Loss, clip de gradiente — com o contrato usado por `train.py`.

Para colocar o seu no lugar:

- **Só a rede**: copie/edite `dqn/model.py` mantendo entrada `(5, H, W)` e saída 12. O agente, o treino, os checkpoints e a retomada continuam funcionando.
- **O algoritmo inteiro**: escreva sua classe com o mesmo contrato do `DQNAgent` que `train.py` usa (coletar transições, aprender a cada passo, salvar/carregar via `dqn/checkpoint.py`). Ambiente, recompensas, adversário e retomada não mudam.
- **Uma política nova (ex.: para o adversário)**: implemente `act(ctx: PolicyContext)` e registre o nome em `make_policy()`, em `agents/baseline.py`.

### 8. Verificar

```bash
python -m pytest -q
```

São 25 testes (~7 s). Rode depois de instalar e depois de cada mudança sua para confirmar que nada quebrou.

## Referência

### Ações

| Índice | Ação |
|---:|---|
| 0 | Não fazer nada |
| 1–2 | Esquerda / direita |
| 3 | Pular verticalmente |
| 4–5 | Pular diagonal esquerda / direita |
| 6–8 | Empurrar adversário para esquerda com força crescente |
| 9–11 | Empurrar adversário para direita com força crescente |

### Adversários

- `auto` (padrão): currículo — 50 episódios contra `random`, depois alterna `coins`/`rule` e, por fim, joga contra o melhor checkpoint, com 10% dos episódios contra outra política. As fases usam contagens absolutas, então retomar continua o currículo de onde parou.
- `random`: sorteia entre as 6 primeiras ações.
- `rule`: aproxima do jogador e às vezes empurra.
- `coins`: persegue a moeda visível.

As políticas ficam em `agents/baseline.py`.

### Checkpoints e retomada

```bash
python train.py --episodes 300 --resume checkpoints/latest.pt
python train.py --episodes 300 --checkpoint-dir /caminho/para/run1 --resume latest.pt
```

`--episodes` indica episódios **adicionais** ao retomar. O checkpoint é pesos-only: guarda as redes (online e alvo), o otimizador, os passos de treino, o epsilon e o número de episódios consumidos — retomar continua a numeração de episódios e o treino de onde parou. Replay buffer e RNGs **não** são persistidos, então a retomada não é mais reproduzível bit a bit. Checkpoints da arquitetura antiga (`cnn-dqn-v1`) não são mais aceitos.

### Estatísticas e TensorBoard

```bash
python train.py --episodes 300 --stats
```

grava `logs/stats.csv` por episódio: recompensa, perda (média/mín/máx), norma do gradiente antes do clip, Q-valores, epsilon e tamanho do buffer.

```bash
python train.py --episodes 300 --tensorboard
tensorboard --logdir logs/runs
```

mostra as mesmas curvas, histogramas dos pesos da CNN e a observação como imagem; cada seed escreve em `logs/runs/seed<N>`.

### Skins

```bash
python play.py --player human --opponent coins --skin neon
```

Cada skin é um arquivo `arena/renderer_<skin>.py`; as opções de `--skin` são descobertas automaticamente — criar `arena/renderer_dinosaur.py` já habilita `--skin dinosaur`.

### Gravação de vídeo

```bash
python play.py --player coins --opponent coins --record demonstracao.mp4
```

30 fps na resolução da arena, H.264 (`yuv420p`, crf 23, `faststart`) — compatível com WhatsApp e players comuns (~1–2 MB por episódio). Requer `imageio` e `imageio-ffmpeg`. Para gravar sem janela: `SDL_VIDEODRIVER=dummy python play.py ...`.

### Observações e limitações da versão didática

- O cenário e a configuração inicial são fixos.
- O episódio termina pelo limite de tempo (truncamento), nunca apenas por morte. Ao morrer, o personagem perde 10 moedas (saldo mínimo zero), fica inativo por 5 s simulados e reaparece sobre uma plataforma aleatória.
- A observação é uma grade semântica `(5, H, W)`; velocidades e temporizadores não estão codificados, então a observação não é totalmente Markoviana — simplificação deliberada para a primeira experiência com CNN.
- O crédito por eliminações é simplificado.

### Estrutura

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
