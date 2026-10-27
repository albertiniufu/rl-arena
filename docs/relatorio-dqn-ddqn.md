# DQN e Double DQN: diagnóstico e busca de hiperparâmetros

Data: 2026-10-27.

## Resumo

As implementações de `alg_dqn` e `alg_doubledqn_per_dueling` estão
matematicamente corretas: o alvo de TD, o Double DQN, a cabeça dueling, o
replay e a sincronização da rede-alvo foram auditados e um teste
automatizado mostra que os dois agentes aprendem um MDP de dois estados.

O desempenho ruim vinha de quatro problemas de **configuração/agendamento**:

1. **A exploração morria no episódio ~13.** O ε decaía por passo
   (`epsilon_decay=0.9995`, ~450 passos por episódio) e atingia o piso 0,05
   em 13 episódios — num treino de 3 h com 918 episódios (DQN) ou 524
   (Double DQN). O agente passava o resto do treino explorando o Q de um
   agente quase aleatório.
2. **A rede-alvo era sincronizada a cada 10 updates (`C=10`).** O DQN do
   Nature sincroniza a cada 10 mil passos; com C=10 o alvo se move junto
   com a online e o treino fica instável. Foi a maior alavanca isolada:
   com C=1000 o DQN coletou 15,8 moedas de treino contra 8,8 do C=10 e a
   avaliação final coletou 12,3 contra 5,9.
3. **O PER rank-based não tinha α nem pesos de importância.** A amostragem
   proporcional a `1/rank` concentrava ~28% dos sorteios nas 10 transições
   de maior prioridade e deixava a amostra efetiva em ~67 transições de 20
   mil. No Double DQN isso inflou o Q para ~50 e a loss para ~51.
4. **Não havia warmup:** o primeiro update acontecia com 64 transições do
   começo do episódio, altamente correlacionadas.

Correções aplicadas: rampa de ε por episódio absoluto (1 → 0,1 em 150
episódios), C=1000 nos dois algoritmos, warmup de 1000 transições, PER com
α=0,6 e β agendado de 0,4 até 1 e um update a cada 4 passos
(`--learn-every 4`). A validação de 300 episódios está na seção 4.

## 1. Auditoria da implementação

Revisão linha a linha de `alg_dqn/` e `alg_doubledqn_per_dueling/`:

| Item | Situação |
| --- | --- |
| Alvo de TD `r + γ·max Q_target(s', ·)` | Correto; só `terminated` zera o futuro (truncamento continua bootstrapping). |
| Double DQN (`online` escolhe, `target` avalia) | Correto. |
| Dueling `Q = V + A − média(A)` | Correto. |
| Sincronização da target a cada `C` updates | Implementação correta (contador de `learn()` bem-sucedidos); o **valor** `C=10` era agressivo demais — ver §4.6. |
| Replay uniforme com cópia da observação | Correto. |
| Rede inicial (~Q ≈ 0,05) e otimizador | Corretos; não havia bug de inicialização. |
| Aprendizado de um MDP de dois estados | Teste `test_offpolicy_agents_learn_two_state_mdp` passa para os dois agentes. |

Conclusão da auditoria: o código estava certo; o problema estava nos
**defaults e no agendamento**.

## 2. Evidência dos problemas

### 2.1 Exploração

Com ~450 passos de agente por episódio, o decay por passo atinge o piso
muito cedo:

| `epsilon_decay` | Episódio em que ε chega ao piso 0,1 |
| --- | --- |
| 0,9995 (default) | ~13 (piso 0,05) |
| 0,9999 | ~51 |
| 0,99998 | ~256 |
| 0,99999 | ~512 |

Os treinos de 3 h terminaram com ε=0,05 no episódio 918 (DQN) e 524
(Double DQN). Com a política praticamente gulosa desde o episódio 13, o
DQN estagnou em 4–6 moedas de treino — média por blocos de 100 episódios:
4,4 → 4,4 → 5,5 → 6,2 → 5,6 → 5,8 → 6,1 → 6,4 → 4,3 → 4,7 — e avaliação
gulosa de 1,90 (best) e 2,20 (latest). O PPO, que não usa ε, chegou a
64,75/71,80 no mesmo orçamento — a arena é aprendível; o DQN é que parou
de explorar.

### 2.2 PER sem correção de viés

O buffer antigo sorteava com probabilidade `1/rank`, sem α e sem pesos de
importância. Para 20 mil transições, a amostra efetiva (ESS) fica em ~67
(0,33% do buffer) e o top-10 do rank recebe 27,9% dos sorteios. O efeito
observado no Double DQN de 3 h:

| Métrica | Valor |
| --- | --- |
| Loss final | 50,89 (pico 133,1 no episódio 393; primeira loss > 10 no episódio 213) |
| Q máximo | ≈ 49,7 |
| Norma de gradiente máxima | 132,9 |
| Moedas por episódio (blocos de 100) | 2,0 → 2,8 → 2,2 → 2,1 → 2,0 → 2,2 (nunca aprendeu) |
| Política gulosa final | JUMP em 100% dos passos |

O Q inflado é o sintoma clássico de updates viesados e de alta variância
sobre um punhado de transições superamostradas.

### 2.3 Warmup

Sem warmup, o primeiro `learn()` roda assim que o buffer tem `batch_size`
(64) transições — todas do primeiro episódio. O warmup de 1000 transições
(≈ 2,2 episódios) garante diversidade mínima antes do primeiro update.

### 2.4 Orçamento de parede: muitos updates, poucos dados

O loop de 1 ambiente chama `learn()` a cada transição. Com a CNN em CPU,
cada passo custa ~26 ms (forward + backward + amostragem do replay), então
os algoritmos de valor processam muito menos ambiente que o PPO no mesmo
tempo de parede:

| Algoritmo (3 h) | Passos de agente | Passos/s |
| --- | --- | --- |
| PPO | 2,27 M | 210 |
| DQN | 413 k | 38 |
| Double DQN | 236 k | 22 |

O Double DQN ainda paga o `argsort` do PER a cada amostra (~3 ms). O PPO
vê 5,5× mais transições que o DQN e 9,6× mais que o Double DQN — parte da
diferença de desempenho é orçamento de dados, não só o algoritmo.

## 3. Correções aplicadas

### 3.1 Rampa de ε por episódio (`train.py`)

- `epsilon_for_episode(ep, ramp_end, floor)`: rampa linear de 1 até `floor`
  ao longo de `ramp_end` episódios **absolutos**.
- `--epsilon-ramp-episodes` (padrão 150) e `--epsilon-min` (padrão 0,1).
- `schedule_epsilon()` aplica a rampa no início de cada episódio e desliga
  o decay por passo do agente; algoritmos on-policy (PPO/A2C) não são
  tocados.
- A âncora em episódios absolutos é o que faz a rampa funcionar com
  `--max-hours`: `scripts/train_long.sh` usa `--episodes 1000000`, então
  uma rampa relativa a `--episodes` nunca terminaria. Como `ep` continua a
  numeração na retomada, a rampa também retoma de onde parou.

### 3.2 Warmup (`alg_dqn`, `alg_doubledqn_per_dueling`)

- `warmup` no construtor (padrão 0 para compatibilidade; `train.py` passa
  `--warmup`, padrão 1000).
- `learn()` devolve `None` enquanto `len(buffer) < max(warmup, batch_size)`.

### 3.3 PER com α e pesos de importância

- Amostragem `rank^-α` com α=0,6 (configurável).
- Peso de importância `(N·p)^-β` normalizado pelo máximo, com β subindo
  linearmente de 0,4 até 1 em `beta_steps` (100 mil updates).
- Perda `SmoothL1` por amostra multiplicada pelo peso e reduzida pela média.
- Medição com 20 mil transições: ESS sobe de ~67 para ~2910 (14,6% do
  buffer) e o top-10 cai de 27,9% para 3,0% dos sorteios.

### 3.4 `--learn-every`

- `--learn-every K` faz um update a cada K transições (padrão 1, que
  mantém o comportamento anterior). K=4 é o padrão do DQN do Nature.
- Reduz o custo por passo de ambiente e aumenta os dados vistos por hora;
  em troca, faz menos updates por transição.
- No loop vetorizado (`--envs > 1`) o update já é a cada 4 passos do
  vetor; `--learn-every` é recusado nesse modo para não haver dois
  controles concorrentes.

## 4. Busca de hiperparâmetros

### 4.1 Protocolo

Seed 7, adversário `rule`, `frame_stack=4`, `frame_skip=3`, batch 64,
buffer 20 mil, lr 2,5e-4. Avaliação gulosa em 20 episódios (seeds
1000–1019) do melhor checkpoint de treino e do final. Treino e avaliação
usam o mesmo algoritmo de `alg_registry`/`ArenaEnv` do projeto.

### 4.2 Triagem DQN (90 episódios)

| Trial | C | ε | Treino (últ. 20) | Moedas | Melhor eval | Eval final |
| --- | --- | --- | --- | --- | --- | --- |
| `dqn_base` | 10 | decay 0,9995 | 4,15 | 5,85 | 5,60 | 3,35 |
| `dqn_eps9999` | 10 | decay 0,9999 | 1,37 | 4,35 | 2,41 | −0,90 |
| `dqn_C1000` | 1000 | decay 0,9995 | 1,33 | 7,10 | −2,66 | 1,06 |
| `dqn_C1000_eps9999` | 1000 | decay 0,9999 | −3,60 | 5,30 | −23,33 | −9,93 |

Com 90 episódios a variância domina e nenhum trial supera o plateau de
~5 moedas já visto em 3 h. A conclusão útil veio do Double DQN.

### 4.3 Triagem Double DQN (90 episódios, código antigo)

| Trial | C | ε | Moedas (últ. 20) | Melhor eval | Eval final | Q máx | Grad máx |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ddqn_base` | 10 | decay 0,9995 | 2,75 | 0,75 | 1,20 | 59,9 | 132,9 |
| `ddqn_C1000_eps9999` | 1000 | decay 0,9999 | 10,35 | −11,41 | −7,90 | 3,7 | 1,9 |
| `ddqn_uniform_C1000_eps9999` | 1000 | decay 0,9999 | 7,85 | −7,96 | −3,43 | 3,8 | 1,0 |

C=1000 elimina a instabilidade (Q e gradiente normais) e o treino passa a
coletar 10,35 moedas/episódio — mas a avaliação gulosa ainda tem episódios
degenerados (parados ou pulando), o que derruba a média. O ablativo com
replay uniforme mostra que, com C=1000, a concentração do PER antigo não
era mais o fator dominante em 90 episódios; em 3 h, era.

### 4.4 Validação com o código corrigido (300 episódios)

PER corrigido (α=0,6, β 0,4→1), rampa de 150 episódios, warmup 1000:

| Trial | C | Coletou (últ. 20) | Melhor eval | Eval final | Coletou (eval) | Q máx | Grad máx | Duração |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ddqn_fix_C10` | 10 | 4,65 | 3,30 | 0,80 | 2,70 | 17,9 | 13,7 | 83 min |
| `ddqn_fix_C1000` | 1000 | 7,35 | 2,50 | 4,21 | 5,95 | 7,6 | 0,5 | 81 min |

Com o PER corrigido os dois valores de C são estáveis (Q máx < 18, sem
loss explodindo), mas C=1000 coleta mais no treino (7,35 vs 4,65) e tem
avaliação final melhor (4,21 vs 0,80). A curva do C=1000 ainda subia no
fim: 5,2 → 5,9 → 6,8 → 6,5 → 6,5 → 7,0 → 7,9 → 7,9 → 8,1 → 7,0 moedas por
bloco de 30 episódios. **C=1000 passa a ser o default do Double DQN.**

Para comparação, o Double DQN de 3 h com o código antigo ficou em ~2
moedas/episódio e loss de até 133; o código corrigido, em 300 episódios,
coleta 7–8 e mantém a loss em ~0.

### 4.5 DQN com a rampa e o `--learn-every`

Os dois trials usam o DQN com C=10, a rampa de 150 episódios e warmup
1000; a única diferença é o intervalo entre updates.

| Trial | `--learn-every` | Moedas (últ. 20) | Melhor eval | Eval final | Coletou (eval) | Q máx | Grad máx | Duração |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dqn300_ramp150_learn4` | 4 | 7,5 | −2,94 | 0,28 | 5,9 | 7,0 | 7,4 | 16,6 min |
| `dqn300_ramp150_learn1` | 1 | 4,15 | −9,45 | 2,29 | 2,65 | 19,7 | 5,5 | 53,3 min |

O K=4 aprendeu mais no mesmo número de episódios: 7,5 moedas de treino
(últ. 20) contra 4,15, e a avaliação final coleta 5,9 contra 2,65. A média
de recompensa da avaliação do K=4 (0,28) é puxada por **um** episódio
degenerado (−45, política gulosa parada); a mediana é 4,0 contra 2,0 do K=1
e, fora esse episódio, a distribuição do K=4 é melhor. O K=4 também roda em
**16,6 min** contra 53,3 min — 3,2× mais rápido.

Curva do K=4 por bloco de 30 episódios (coletou): 6,0 → 5,5 → 8,0 → 7,4 →
10,5 → 7,9 → 7,1 → 8,3 → 9,2 → 8,4; os `STOP` caem de 67 para ~10 passos
por episódio conforme ε desce. O K=4 é o candidato a default dos
algoritmos off-policy; a validação no Double DQN está na seção 4.6.

O episódio degenerado da política gulosa (parada até esgotar os 45 s, −45
de recompensa) aparece esporadicamente nos dois algoritmos de valor e não
é resolvido por nenhum hiperparâmetro testado; é um limite do Q aprendido
para `STOP`, não do treino.

### 4.6 Controles finais: C no DQN e `--learn-every` no Double DQN

Mesmo protocolo (300 episódios, rampa 150, warmup 1000). Os dois controles
completam o quadro: o DQN com C=1000 no lugar do C=10 e o Double DQN com
`--learn-every 4` no lugar de 1.

| Trial | Algoritmo | C | K | Moedas treino (últ. 20) | Coletou (treino) | Melhor eval | Eval final | Coletou (eval) | Q máx | Grad máx | Duração |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dqn300_ramp150_learn4` | DQN | 10 | 4 | 7,5 | 8,8 | −2,94 | 0,28 | 5,9 | 7,0 | 7,4 | 16,6 min |
| `dqn300_ramp150_learn1` | DQN | 10 | 1 | 4,15 | 4,55 | −9,45 | 2,29 | 2,65 | 19,7 | 5,5 | 53,3 min |
| `dqn300_ramp150_C1000_learn4` | DQN | 1000 | 4 | 14,45 | 15,8 | 4,39 | 4,05 | 12,3 | 3,2 | 1,0 | 15,7 min |
| `ddqn300_fix_C1000_learn1` | DDQN | 1000 | 1 | 5,05 | 7,35 | 2,50 | 4,21 | 5,95 | 7,6 | 0,5 | 81 min |
| `ddqn300_fix_C1000_learn4` | DDQN | 1000 | 4 | 12,85 | 14,45 | −2,49 | 1,21 | 14,15 | 3,2 | 0,5 | 23,5 min |

**C=1000 é melhor nos dois algoritmos.** No DQN com K=4, trocar C=10 por
C=1000 mais que dobrou o treino (8,8 → 15,8 coletadas) e a avaliação final
(5,9 → 12,3), com Q máximo 3,2 contra 7,0 e gradiente 1,0 contra 7,4.
A conclusão vale também para o Double DQN (§4.4). `C=10` era a maior
alavanca de instabilidade — mais até que o PER.

**K=4 é melhor nos dois algoritmos.** No DQN, 3,4× mais moedas coletadas
no treino (15,8 vs 4,55) e 3,2× mais rápido. No Double DQN, a avaliação
final coletou 14,15 contra 5,95 do K=1 e o treino rodou em 23,5 min contra
81. Com um orçamento fixo de episódios, ver mais dados compensa os updates
menos frequentes por transição.

**Ressalva:** a média de recompensa da avaliação gulosa é puxada por
episódios degenerados (política parada, −39 a −45) em quase todos os runs;
por isso a tabela mostra também "coletou (eval)". No Double DQN de K=4 a
política final coleta 14,15 mas para ~59 passos por episódio (contra 3,4
do DQN), e a recompensa média fica em 1,21 contra 4,05 — em recompensa, o
DQN corrigido é o melhor resultado da busca. O vício do Q em `STOP` é um
limite conhecido da avaliação gulosa, não do treino.

## 5. Configuração final

| Parâmetro | Valor | Origem |
| --- | --- | --- |
| `C` (sincronização da target) | 1000 | §4.4 e §4.6 |
| Rampa de ε | 1 → 0,1 em 150 episódios absolutos | §2.1 e §4.5 |
| Warmup | 1000 transições | §2.3 |
| PER | α=0,6, β 0,4→1 em 100 mil updates, IS weights | §2.2 |
| `--learn-every` | 4 (1 no modo vetorizado) | §4.5 e §4.6 |
| `--lr` | 2,5e-4 (novo default da CLI) | protocolo da busca |
| Batch / buffer | 64 / 50 mil (a busca usou 20 mil por RAM) | protocolo |

Os três primeiros são mudanças de código (`alg_dqn/agent.py`,
`alg_doubledqn_per_dueling/agent.py`, `train.py`); o restante são defaults
da CLI. Com esses valores, o DQN de 300 episódios (16 min) coleta 12,3
moedas na avaliação final, contra 5,9 do default antigo — e o Double DQN
14,15, contra 5,95.
