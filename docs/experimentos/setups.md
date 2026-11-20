# Setups

Um **setup** é a combinação de ambiente, observação, recompensa, regras e
protocolo de avaliação. Algoritmos e seeds são **experimentos** dentro de um
setup.

## Regra

Crie um novo `Sx` quando mudar algo que altere o problema ou a forma de medir:

- física, mapa, níveis, hazards;
- observação (`frame_stack`, `frame_skip`, canais);
- recompensa ou penalidades;
- cenário multiagente/papéis;
- protocolo de avaliação (seeds, episódios, checkpoint).

**Não** crie um novo setup só por mudar algoritmo, seed, `lr` ou batch —
isso é um experimento dentro do setup.

## Histórico

| Setup | Mudança principal | Como reproduzir | Experimentos |
| --- | --- | --- | --- |
| **S0** | Baseline single-agent do seek_coin: nível 0, `frame_stack=1`, `frame_skip=1`, adversário `auto`, DQN. | `python train.py --arena seek_coin --alg dqn --episodes 300` | execuções iniciais |
| **S1** | Protocolo de comparação controlada: seed 7, 100 episódios, adversário `rule`, `frame_stack=4`, `frame_skip=3`, batch 64, lr 2.5e-4, avaliação gulosa de 20 eps (seed 1000). | [`scripts/compare_onpolicy.sh`](../../arenas/seek_coin/scripts/compare_onpolicy.sh); detalhes no [README da arena](../../arenas/seek_coin/README.md#comparações-controladas) | comparações DQN × Double DQN × PPO × A2C |
| **S2** | Níveis: nível 1 com plataformas móveis; nível 2 com areia/gelo e canais extras de observação. | `python train.py --arena seek_coin --level 1` ou `--level 2`; ver [`arena/levels.py`](../../arenas/seek_coin/arena/levels.py) e [`arena/engine.py`](../../arenas/seek_coin/arena/engine.py) | nível 1/2 (PPO, IPPO, EPPO, PPO-AES, GTR) |
| **S3** | Multiagente homogêneo: `ippo` (crítico local) e `mappo` (crítico centralizado), cenários `all-vs-all` e `teams`, observação egocêntrica de 6 canais. | `python train.py --alg mappo --agents 4 --scenario teams --teams 2`; ver [`arena/scenario.py`](../../arenas/seek_coin/arena/scenario.py) | comparações IPPO × MAPPO |
| **S4** | Papéis heterogêneos: cenário `collectors-vs-pushers`, HAPPO, máscaras de ação por papel, calibração de empurrão e cooldown. | `python train.py --alg happo --agents 4 --scenario collectors-vs-pushers --teams 2`; ver commit `1c87219` | [HAPPO × MAPPO](EXP-20261108-happo-vs-mappo.md) |
| **S5** | Treino vetorizado/GPU: `--envs`, `GpuArenaVecEnv` e `scripts/train_gpu_ppo.py`. | [`scripts/benchmark_gpu_env.py`](../../arenas/seek_coin/scripts/benchmark_gpu_env.py) e [`scripts/train_gpu_ppo.py`](../../arenas/seek_coin/scripts/train_gpu_ppo.py) | benchmarks de throughput |
| **S6** | Nível 3: botão de ativação no chão esquerdo, cancelamento na plataforma central e super moeda no chão direito. | `python train.py --arena seek_coin --level 3`; ver [`arena/levels.py`](../../arenas/seek_coin/arena/levels.py) | BO PPO × EPPO; [confirmação 10 seeds](EXP-20261120-ppo-vs-eppo-level3-10seeds.md) |
| **S7** | Visão local (nível 4): mapa do nível 3 com a mecânica de botões/super moeda **desligada por padrão** (`--super-coin` reativa), observação recortada `view_size × view_size` (padrão 9). Muda o shape da observação e o que o agente pode perceber; recompensa/física/ações são as mesmas de S6. | `python train.py --arena seek_coin --level 4 --view-size 9`; ver [`arena/observations.py`](../../arenas/seek_coin/arena/observations.py) | planejado: nível 3 (S6) × nível 4 (S7) no mesmo orçamento e seeds, medindo recompensa/moedas, passos até o alvo e estabilidade. Como S7 não tem a super moeda por padrão, rode S6 também com `--no-super-coin` (ou S7 com `--super-coin`) para isolar o efeito da visão local. Comece com 3+ seeds, checkpoint e episódios de avaliação idênticos; use frame stack nos dois lados e reporte a variância. |
| **S8** | Mundo procedural (nível 5): chunks determinísticos de 480 px com chão/buracos, plataformas elevadas/móveis, areia/gelo e rolagem contínua; observação local com canal `hole`. | `python train.py --arena seek_coin --level 5 --view-size 9`; ver [`arena/endless.py`](../../arenas/seek_coin/arena/endless.py) | planejado: nível 4 (S7) × nível 5 (S8) em distância percorrida, moedas por 1.000 passos e mortes por queda; avaliar também zero-shot cruzado (checkpoint de S7 em S8 e vice-versa). Reportar por seed e com IC; comparar o custo por passo/memória. |

> S0–S2 são cumulativos no ambiente; S3–S4 mudam o cenário. Sempre registre na
> entrada do experimento qual `Sx` foi usado.

## Como adicionar um setup

1. Copie a linha do setup anterior e descreva só o que mudou.
2. Aponte o arquivo/commit que implementa a mudança.
3. Registre o primeiro experimento que usa o novo setup.
4. Se a mudança for opcional (flag), diga o valor usado no experimento — não
   basta citar o setup.
