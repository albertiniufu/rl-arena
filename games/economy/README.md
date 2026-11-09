# Economia — banco-agente em um simulador macroeconômico

Simulador didático de uma economia com **população/clientes**, **empresas
concorrentes com produtos**, **governo** e **variabilidade macroeconômica**
(choques de crescimento, inflação e desemprego). O agente principal é um
**banco de crédito** que decide **concessão** e **precificação por grupo
homogêneo de risco (GH)** e busca maximizar o **RAROC líquido**.

## Como rodar

A partir da raiz do repositório:

```bash
python -m games.economy.bank_demo --steps 24 --policy risk-aware
python -m games.economy.bank_demo --steps 24 --policy naive
python -m games.economy.bank_demo --steps 24 --policy random
python -m games.economy.demo --steps 24          # modo governo
pytest -q games/economy
```

## Agente banco (`BankEnv`)

### Grupos homogêneos de risco (GH)

Quatro grupos fixos em `BankConfig` (todos configuráveis):

| GH | PD base | LGD | Exposição | Demanda | Spread de referência |
| --- | --- | --- | --- | --- | --- |
| A — baixo risco | 1% | 35% | 1.000 | 180 | 4% |
| B — médio-baixo | 3% | 45% | 800 | 150 | 7% |
| C — médio-alto | 8% | 55% | 600 | 110 | 11% |
| D — alto risco | 15% | 65% | 400 | 70 | 16% |

A PD efetiva sobe com estresse macro (crescimento negativo, inflação acima
da meta e desemprego alto). A demanda cai quando o spread sobe acima da
referência e quando a economia esfria.

### Ações (por GH, espaço contínuo)

Para cada GH, o agente escolhe:

- **aprovação** `[0, 1]`: fração das propostas aprovadas (0 nega tudo);
- **limite** `[0.1, 1.0]`: multiplicador da exposição por operação;
- **spread** `[0, 0.25]`: taxa sobre o custo de funding.

Ação total: `Box(3 * n_GH)`, por exemplo `[[aprovação, limite, spread], ...]`.

### Recompensa (RAROC líquido)

```
RAROC = (receita líquida de juros - perda esperada - opex)
        / capital econômico
```

- receita líquida de juros = `EAD × spread`;
- perda esperada = `EAD × PD × LGD`;
- opex = custo por operação + custo proporcional ao EAD;
- capital econômico = maior entre capital regulatório (`EAD × 8%`) e perda
  inesperada (`EAD × LGD × sqrt(PD(1-PD)) × 2,326`), com carga extra por
  concentração acima de 40% em um GH;
- a recompensa desconta o **default realizado**, exige capital mínimo e
  penaliza a falência do banco.

O banco começa com `initial_capital = 100.000`. Perdas realizadas reduzem o
capital; se ele chegar a zero, o episódio termina (falência).

### Observação (32 dimensões para 4 GHs)

- macro: PIB normalizado, crescimento, inflação, desemprego e taxa de juros;
- por GH: PD, LGD, fator de demanda, aprovação, limite e spread anteriores;
- banco: capital relativo, último RAROC e taxa de default realizada.

`reset → (obs, info)` e `step → (obs, reward, terminated, truncated, info)`;
`info["bank"]` traz EAD, PD, perdas, capital, RAROC, defaults e concentração.
O RNG do banco é separado do RNG da economia e `reset(seed=...)` torna tudo
reprodutível.

## O modelo econômico

Cada passo é um período (mês) e a moeda é abstrata.

- **Clientes** (`Customer`): emprego, renda (salário ou seguro-desemprego),
  poupança e propensão a consumir. Compram no máximo um produto por passo,
  escolhendo por utilidade logit preço × qualidade.
- **Empresas** (`Firm`/`Product`): empregam, produzem com estoque, pagam
  salários e insumos, ajustam preço e pagam imposto sobre o lucro.
- **Governo**: arrecada impostos, paga seguro-desemprego, faz compras
  públicas e paga juros sobre a dívida.
- **Macro**: crescimento AR(1) com choques raros, salários e benefícios
  indexados à inflação, inflação pelo índice de preços, desemprego e Gini.

## Modo governo (`EconomyEnv`)

O ambiente anterior continua disponível: o agente é o governo e escolhe
alíquota de imposto de renda, gasto público/PIB e taxa de juros, com
recompensa de bem-estar social. Ver `env.py`.

## Estrutura

| Arquivo | Papel |
| --- | --- |
| `config.py` | `EconomyConfig`: parâmetros da economia. |
| `model.py` | `Economy`: motor puro, sem Gym. |
| `bank.py` | `Bank`/`BankConfig`/`RiskGroup`: GHs, crédito, RAROC. |
| `bank_env.py` | `BankEnv`: ambiente Gymnasium do banco. |
| `env.py` | `EconomyEnv`: ambiente Gymnasium do governo. |
| `bank_demo.py` | Políticas de exemplo (risk-aware, naive, random). |
| `demo.py` | Simulação macro em texto. |
| `tests/` | Economia (7 testes) e banco (9 testes). |

## Ganchos para RL

- O espaço de ação é contínuo; dá para usar PPO/SAC ou discretizar
  aprovação/limite/spread por GH.
- Próximos passos naturais: colateral e garantias, provisão/ECL, rating
  interno, metas de concentração e capital por linha, e um banco
  multiagente competindo no mesmo mercado.
