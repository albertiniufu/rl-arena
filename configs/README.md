# Configurações YAML

`configs/` guarda presets de execução e espaços de busca.

## Estrutura

```text
configs/
├── seek_coin/
│   ├── base-level3.yaml    # setup comum do nível 3
│   ├── ppo-level3.yaml     # melhor PPO da campanha BO
│   └── eppo-level3.yaml    # melhor EPPO da campanha BO
└── search_spaces/
    ├── ppo.yaml            # espaço de busca do PPO
    └── eppo.yaml           # espaço de busca do EPPO
```

## Usar

O campo `arena` permite omitir `--arena`:

```bash
python train.py --config configs/seek_coin/ppo-level3.yaml
python evaluate.py --config configs/seek_coin/ppo-level3.yaml \
  --checkpoint checkpoints/ppo-level3/latest.pt
```

Precedência: **CLI > YAML > defaults**. Para sobrescrever um campo:

```bash
python train.py --config configs/seek_coin/ppo-level3.yaml \
  --set training.max_hours=0.5 --set seed=42
```

## Formato

```yaml
arena: seek_coin
setup: S6
alg: ppo
level: 3
episode_seconds: 90
frame_stack: 4
frame_skip: 3
opponent: rule
seed: 7

training:
  episodes: 1000000
  max_hours: 6
  checkpoint_dir: checkpoints/ppo-level3
  agent_params:
    lr: 0.00068
    batch_size: 256

evaluation:
  episodes: 50
  seed: 1000
```

`agent_params` vira `--agent-param chave=valor`. Chaves desconhecidas geram
erro com o nome do campo.

## BO genérica

Os espaços em `search_spaces/` são lidos por `scripts/bo_campaign.py`:

```bash
python scripts/bo_campaign.py \
  --algorithms ppo,eppo \
  --base-config configs/seek_coin/base-level3.yaml \
  --space-dir configs/search_spaces \
  --hours-per-alg 6 \
  --out experiments/BO-generica

python scripts/asha_campaign.py \
  --algorithms ppo,eppo \
  --base-config configs/seek_coin/base-level3.yaml \
  --space-dir configs/search_spaces \
  --rungs 5,15,30,60 --initial-trials 16 --eta 2 \
  --out experiments/ASHA-generica
```
