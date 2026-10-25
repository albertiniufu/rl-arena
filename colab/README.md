# Treino no Colab (GPU)

## Arquivos

- `rl_arena_colab.ipynb` — notebook para rodar no Colab.
- `requirements-colab.txt` — dependências que faltam na imagem do Colab
  (o PyTorch com CUDA já vem instalado).
- `build_zip.sh` — gera `dist/rl-arena-colab.zip` a partir do HEAD do git
  (somente arquivos rastreados).

## Como usar

1. Gere o pacote na raiz do repositório:

   ```bash
   bash colab/build_zip.sh
   ```

   O arquivo sai em `dist/rl-arena-colab.zip`.

2. No Colab, abra `colab/rl_arena_colab.ipynb` (*Arquivo → Fazer upload de
   notebook*) e escolha *Ambiente de execução → Alterar o tipo de ambiente de
   execução → GPU (T4)*.

3. Rode as células na ordem. A célula de upload pede o
   `rl-arena-colab.zip`; as seguintes extraem em `/content/rl-arena`,
   instalam as dependências, treinam, avaliam e empacotam os resultados.

## O que o notebook faz

- verifica CUDA/GPU;
- extrai o pacote em `/content/rl-arena`;
- instala `colab/requirements-colab.txt`;
- treina com `train.py` (frame stack 4, frame skip 3, n-step 3, Double DQN
  por padrão; tudo ajustável na célula de configuração);
- mostra o TensorBoard (`logs/runs`);
- avalia `best.pt` e os snapshots de orçamento (`budget/ep*.pt`) contra o
  adversário `rule`, no protocolo do A/B (20 episódios, seed 1000);
- plota a curva de treino;
- empacota `checkpoints/` e `logs/` em `rl-arena-resultados.zip`.

## Restrições do modo vetorizado

Com `--envs` > 1, o `train.py` recusa `--opponent auto`, `--render`,
`--stats`, `--tensorboard` e `--n-step` > 1 (o currículo por episódio, a
telemetria por processo e a janela n-step do replay não mapeiam em workers
concorrentes). Para medir velocidade:

```python
ENVS = 8
N_STEP = 1
TENSORBOARD = False
OPPONENT = "random"
```

## Observações

- O checkpoint guarda `frame_stack`/`frame_skip`; `evaluate.py` e `play.py`
  aplicam a config gravada automaticamente.
- A rede é pequena: a GPU pode não acelerar muito o treino, porque o gargalo
  é a simulação em CPU. O modo vetorizado com vários workers costuma ser o
  ganho real.
- O Colab tem limite de tempo de GPU; rode sessões menores com `--resume`
  se precisar (a retomada continua a numeração dos episódios, mas buffer e
  RNGs recomeçam — não é bit a bit).
