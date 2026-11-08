# Economia — simulador de sistemas econômicos

Simulador didático de uma economia com **população/clientes**, **empresas
concorrentes com produtos**, **governo** (impostos, gastos, juros) e
**variabilidade macroeconômica** (choques de crescimento, inflação e
desemprego).

## Como rodar

A partir da raiz do repositório:

```bash
python -m games.economy.demo --steps 24
python -m games.economy.demo --steps 36 --policy intervention
pytest -q games/economy
```

## O modelo

Cada passo é um período (mês) e a moeda é abstrata.

- **Clientes** (`Customer`): emprego, renda (salário ou seguro-desemprego),
  poupança, preferência por qualidade e propensão a consumir. Compram no
  máximo um produto por passo, escolhendo por utilidade logit
  preço × qualidade.
- **Empresas** (`Firm`/`Product`): empregam, produzem com estoque, pagam
  salários e insumos, ajustam preço conforme estoque/inflação e pagam
  imposto sobre o lucro. Empresa com caixa negativo demite e recebe um
  resgate implícito (simplificação didática).
- **Governo**: arrecada imposto de renda e de lucro, paga seguro-desemprego,
  faz compras públicas (fração do PIB) e paga juros sobre a dívida. A
  política é ajustável a cada passo.
- **Macro**: crescimento AR(1) com choques raros, salários e benefícios
  indexados à inflação, inflação medida pela variação do índice de preços,
  desemprego e Gini calculados a cada passo.

## Ambiente Gymnasium

`games/economy/env.py` expõe `EconomyEnv`, em que o agente é o governo:

- **Ação** (`Box(3,)`): alíquota de imposto de renda `[0, 0.6]`, razão de
  gasto público/PIB `[0.05, 0.5]` e taxa de juros `[0, 0.05]`.
- **Observação** (`Box(11,)`): PIB normalizado, inflação, desemprego, preço
  relativo, dívida/PIB, instrumentos de política, lucro das empresas,
  poupança média e fração do episódio.
- **Recompensa**: bem-estar social (PIB per capita − desvio da inflação −
  desemprego − dívida − desigualdade).
- `reset → (obs, info)`, `step → (obs, reward, terminated, truncated, info)`.

Truncamento em `EconomyConfig.max_steps` (padrão 120). O RNG é próprio do
motor e `reset(seed=...)` torna a simulação reprodutível.

## Estrutura

| Arquivo | Papel |
| --- | --- |
| `config.py` | `EconomyConfig`: todos os parâmetros do mundo. |
| `model.py` | `Economy`: motor puro, sem Gym. |
| `env.py` | `EconomyEnv`: ambiente Gymnasium. |
| `demo.py` | Simulação em texto com tabela por passo. |
| `tests/test_economy.py` | Invariantes, contabilidade do governo, reprodutibilidade e contrato Gym. |

## Ganchos para RL

- O agente pode controlar o governo (como em `EconomyEnv`) ou, em uma
  variante multiagente, cada empresa definir preço/produção.
- As métricas de `metrics()` são um ponto natural para recompensas
  alternativas (bem-estar, igualdade, estabilidade).
