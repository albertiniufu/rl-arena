# Backlog

Itens de trabalho do repositório, separados entre **pendentes** e
**concluídos**. Marque `[x]`, mova a seção para “Concluídos” e registre onde
o item foi implementado (arquivo + testes) quando terminar.

## Visão geral

| Item | Status | Onde |
| --- | --- | --- |
| Sistema de níveis do seek_coin (0/1) | ✅ concluído | [`arenas/seek_coin/arena/levels.py`](arenas/seek_coin/arena/levels.py) |
| Nível 2 — areia e gelo | ✅ concluído | [`arenas/seek_coin/arena/engine.py`](arenas/seek_coin/arena/engine.py) |
| Metodologia estatística para relatórios | ✅ concluído | [`arenas/common/report.py`](arenas/common/report.py) |
| Episódios de 90 s | ✅ concluído | [`arenas/seek_coin/arena/config.py`](arenas/seek_coin/arena/config.py) |
| Nível 3 — botão + super moeda | ✅ concluído | [`arenas/seek_coin/arena/engine.py`](arenas/seek_coin/arena/engine.py) |
| Campanha BO: PPO × EPPO no nível 3 | ✅ concluído | [`docs/experimentos/EXP-20261115-bo-level3-6h.md`](docs/experimentos/EXP-20261115-bo-level3-6h.md) |
| Otimização bayesiana de hiperparâmetros | ✅ concluído | [`scripts/bo_campaign.py`](scripts/bo_campaign.py) |
| Configurações de execução em YAML | ✅ concluído | [`arenas/common/yaml_config.py`](arenas/common/yaml_config.py) |
| Fila de Experimentos (web) | ✅ concluído | [`docs/fila-de-experimentos.md`](docs/fila-de-experimentos.md) |
| DDPG (implementação + testes) | ✅ concluído | [`arenas/algs/alg_ddpg/`](arenas/algs/alg_ddpg/) |
| TD3 (implementação + testes) | ✅ concluído | [`arenas/algs/alg_td3/`](arenas/algs/alg_td3/) |
| SAC (implementação + testes) | ✅ concluído | [`arenas/algs/alg_sac/`](arenas/algs/alg_sac/) e [`alg_sac_discrete/`](arenas/algs/alg_sac_discrete/) |
| ASHA para campanhas de BO | ✅ concluído | [`scripts/asha_campaign.py`](scripts/asha_campaign.py) |
| Skill `experimenter-helper` (propostas MD + YAML) | ✅ concluído | [`.agents/skills/experimenter-helper/SKILL.md`](.agents/skills/experimenter-helper/SKILL.md) |
| Estudo: Empirical Design in RL (JMLR) | ✅ concluído | [`docs/experimentos/empirical-design.md`](docs/experimentos/empirical-design.md) |
| Log de job local na Fila de Experimentos | ✅ concluído | [`arenas/common/queue_gate.py`](arenas/common/queue_gate.py) |
| Sincronizar status do YAML com a Fila | ✅ concluído | [`arenas/common/job_queue.py`](arenas/common/job_queue.py) |
| Abas de análise e proposta na Fila | ✅ concluído | [`arenas/common/job_queue_ui.py`](arenas/common/job_queue_ui.py) |
| Aba de algoritmos por arena na Fila | ✅ concluído | [`arenas/common/algorithm_catalog.py`](arenas/common/algorithm_catalog.py) |
| Visualização da evolução do BO na Fila | ⏳ pendente | — |
| Nível 4 — visão local limitada | ⏳ pendente | — |
| Nível 5 — mundo contínuo procedural | ⏳ pendente | — |
| Ambiente OOD para validar agentes do seek_coin | ⏳ pendente | — |
| Testes de aprendizado off-policy | ⏳ pendente | — |
| Offline RL (dataset fixo) | ⏳ pendente | — |
| Algoritmos com alta eficiência amostral | ⏳ pendente | — |
| GRPO — PPO sem crítico (baseline relativo ao grupo) | ⏳ pendente | — |
| Investigar arquiteturas seq2seq/Transformer | ⏳ pendente | — |
| Pré-treino com recompensas verificáveis | ⏳ pendente | — |
| The Option Keyboard | ⏳ pendente | — |

## Pendentes

### Visualização da evolução do BO na Fila de Experimentos

- [ ] **Adicionar à Fila de Experimentos uma visualização da evolução do BO de cada algoritmo em cada experimento.**
  - Motivação: os dados brutos (`trials.jsonl`, `campaign.json`, `best.json`) já ficam no diretório do experimento, mas investigar convergência, efeito dos hiperparâmetros e melhoria exige abrir arquivos e montar gráficos manualmente.
  - Acesso: botão/aba **BO** no painel do experimento e/ou seção na aba **Análise de resultados**, filtrável por experimento e algoritmo (`ppo`, `eppo`, etc.).
  - Fonte de dados: ler `campaign.json`, `best.json` e `trials.jsonl` de cada algoritmo; suportar também os `trials.jsonl`/`results.csv` do ASHA. Não recalcular treino nem inventar trials.
  - Tabela de trials: índice, status, score, reward, coins, deaths, duração, parâmetros testados, caminho do log e link para o artefato; ordenável e filtrável por status.
  - **Convergência do BO**:
    - score por trial e melhor score acumulado (*best-so-far*) ao longo dos trials/tempo;
    - reward/coins/deaths do melhor trial ao longo da campanha;
    - marcação do trial escolhido como `best` e da validação final;
    - comparação lado a lado entre algoritmos no mesmo experimento.
  - **Efeito de cada parâmetro**:
    - para parâmetros numéricos: dispersão `parâmetro × score`, com tendência/µ de bins e correlação de Spearman;
    - para parâmetros categóricos: box/violin/strip por escolha;
    - *marginal effect*: score médio por faixa do parâmetro, com n e IC quando houver trials suficientes;
    - ranking de importância dos parâmetros (ex.: Spearman absoluto, variância explicada ou permutation importance) com aviso de que 32 trials é amostra pequena;
    - parallel coordinates dos melhores trials e heatmap de interação para os dois parâmetros mais relevantes.
  - **Melhoria**:
    - comparar o melhor trial com o default/base e com a mediana dos trials;
    - mostrar `best_so_far − baseline` e a curva de melhoria;
    - separar ganho por algoritmo e por faixa de orçamento (trial curto × validação final).
  - Backend: endpoints para listar campanhas BO de um experimento e devolver trials normalizados; validar caminhos dentro do repo e não executar código arbitrário.
  - UI: reutilizar Plotly (como em `report_html.py`), manter filtros/estado na URL e permitir exportar os dados filtrados em CSV/JSON.
  - Estatística: deixar explícito que os plots são **exploratórios**; com poucos trials não declarar causalidade nem “melhor parâmetro” sem incerteza. Reportar trials falhos e não esconder outliers.
  - Testes: parser de `trials.jsonl`/ASHA, normalização de parâmetros, best-so-far, agregação por bin, parâmetro categórico, trial falho, campanha ausente e API/rota.
  - Documentar em `docs/fila-de-experimentos.md` e no README de `scripts/bo_campaign.py`/`asha_campaign.py`.

### Nível 4 — visão local limitada

- [ ] **Criar o nível 4 do seek_coin com observação parcial**: o agente não vê a tela inteira, apenas um quadrado limitado ao redor dele.
  - Motivação: testar percepção local, generalização e necessidade de memória; o nível 3 ainda entrega o mapa inteiro, o que facilita planejamento global.
  - Observação: recorte `(view_size, view_size)` da grade semântica centrado no agente (ex.: 7×7 ou 9×9 células). Fora dos limites do mapa, preencher com um valor fixo (ex.: 0 ou canal de borda). O agente fica sempre no centro do recorte.
  - Configuração: `view_size`/`view_radius` em `ArenaConfig`, ativo no nível 4; gravar no checkpoint e validar na retomada. Decidir se o default dos níveis 0–3 continua a tela inteira.
  - Entidades: moedas, adversários, botões e super moeda só aparecem se estiverem dentro do recorte. Não vazar informação global por `info`/renderer de treino.
  - Multiagente: cada agente recebe o próprio recorte centrado em si (visão egocêntrica); shapes iguais entre agentes.
  - Algoritmos: começar com frame stack; documentar que a observação é parcialmente observável e que memória recorrente (RNN/Transformer) fica como evolução futura.
  - Render: manter o renderer da tela inteira para humanos e adicionar um modo de depuração mostrando o recorte do agente (ou uma máscara de névoa).
  - Testes: shape do recorte, agente no centro, padding nas bordas, entidades fora do recorte invisíveis, ausência de canais globais, seed, checkpoint com `view_size` e smoke single/multiagente.
  - Avaliação: comparar nível 3 (visão global) × nível 4 (visão local) no mesmo orçamento e seeds; medir queda de desempenho e necessidade de memória.
  - Documentar no README/CLAUDE, `arena/levels.py` e [`docs/experimentos/`](docs/experimentos/README.md).
  - Fora de escopo inicial: campo de visão cônico, oclusão por paredes/plataformas e arquitetura recorrente dedicada.

### Nível 5 — mundo contínuo procedural

- [ ] **Criar o nível 5: um ambiente continuamente gerado e em movimento**, que posiciona novos elementos aleatoriamente (moedas, plataformas, areia, gelo e buracos) enquanto o agente avança.
  - Motivação: sair do mapa fixo e testar generalização contínua, percepção local e controle em um mundo que não termina; é o passo natural depois do nível 4 (visão local).
  - Mundo: chunks/segmentos gerados proceduralmente à frente e descartados atrás; a câmera acompanha o agente (ou o mundo rola continuamente). A posição do agente pode ser relativa ao chunk atual.
  - Elementos: moedas, plataformas estáticas/móveis, areia, gelo e **buracos** (novo hazard: cair custa morte/respawn ou termina o episódio, a decidir). Taxas de spawn e dificuldade crescem com a distância/tempo.
  - Determinismo: geração por seed e por índice de chunk; o mesmo seed produz a mesma sequência de chunks. Não usar RNG global fora do gerador do ambiente.
  - Observação: usar a visão local do nível 4 (recorte ao redor do agente) para manter shape fixo; o mundo infinito não cabe na tela. Canais extras para areia/gelo/buraco e, se necessário, distância/velocidade.
  - Física: plataformas e buracos precisam interagir com o engine atual; buracos podem ser regiões sem colisão de chão ou zonas fatais. Garantir que moedas não nasçam dentro de buracos/paredes.
  - Memória: remover chunks/elementos fora da visão para manter memória e custo por passo limitados; medir isso em um episódio longo.
  - Render: câmera rolante, desenhar só os chunks visíveis e indicar buracos com clareza.
  - Testes: geração determinística por seed, spawn/despawn de chunks, limites de memória, colisão com buraco, areia/gelo, observação local, checkpoint com nível 5, evaluate/play e smoke de episódio longo.
  - Avaliação: comparar com nível 4 em distância até a morte/queda e moedas por 1.000 passos; reportar por seed.
  - Fora de escopo inicial: geração infinita multiagente, editor de mapas e curriculum automático de dificuldade.

### Ambiente OOD para validar agentes do seek_coin

- [ ] **Criar um ambiente similar ao seek_coin, mas com objetos, moedas e dinâmica de movimento diferentes**, para validar agentes treinados no seek_coin fora da distribuição de treino.
  - Motivação: um agente que só funciona no mapa/física de treino pode ter decorado o cenário. A validação OOD mede generalização de percepção, controle e planejamento.
  - Forma: preferir uma variante dentro de `arenas/seek_coin/` (`--variant shift` ou `level 5`) para reutilizar engine/ambiente e manter o contrato de observação/ação; se o acoplamento ficar artificial, criar `arenas/seek_coin_ood/` e extrair o que for comum.
  - Diferenças mínimas: gravidade/atrito/pulo, plataformas móveis com trajetórias diferentes, moedas que se movem ou têm lifetime distinto, obstáculos/objetos novos e layout alterado. Manter o mesmo `Discrete(6)` e o mesmo shape de observação.
  - Configuração: `ArenaConfig` com um bloco/seção de variante, gravado no checkpoint; `evaluate.py`/`play.py` devem reconstruir a variante correta.
  - Protocolo: treinar no seek_coin normal, avaliar **zero-shot** na variante com seeds/episódios fixos; opcionalmente fine-tune curto para medir adaptação.
  - Métricas: queda de reward/moedas em relação ao seek_coin, taxa de sucesso em objetivos novos, mortes e tempo até a primeira coleta; reportar por seed.
  - Testes: variante determinística com seed, shape de observação/ação, diferenças de dinâmica, checkpoint com variante, evaluate/play e ausência de vazamento de estado global.
  - Documentar no README/CLAUDE e em [`docs/experimentos/`](docs/experimentos/README.md).
  - Fora de escopo inicial: mudar o espaço de ações ou a interface de observação; primeiro validar generalização com o mesmo contrato.

### Testes de aprendizado off-policy

- [ ] **Criar uma suíte de testes que prove que os algoritmos off-policy aprendem de fato** (`dqn`, `doubledqn_per_dueling` e `mka`), sem depender dos algoritmos on-policy.
  - Motivação: os testes atuais cobrem contrato, checkpoints, CLI e os loops, mas o aprendizado em si é validado principalmente por execuções manuais. Falta uma regressão rápida para replay, rede-alvo, ε-greedy, `--warmup` e `--learn-every`.
  - Escopo: off-policy **online** (o agente continua interagindo com o ambiente). Aprender de um dataset fixo (offline RL) seria um item separado.
  - Ambiente: usar um cenário pequeno e determinístico (ex.: `ArenaConfig` reduzido ou um gridworld/bandit de teste) em que a política ótima seja conhecida e a recompensa tenha sinal claro.
  - Casos: DQN, Double DQN + PER + dueling e MKA; treino single-agent e vetorizado (`--envs`); `--save-every 0`; retomada com `--resume`.
  - Asserções: a recompensa média da política gulosa melhora em relação a uma política aleatória e/ou ao checkpoint inicial; usar seed fixa e tolerância para a variância.
  - Performance: suíte curta (ideal < 60 s), sem depender de GPU; marcar como `slow` se necessário.
  - Separar claramente do caminho on-policy: PPO/A2C continuam com os testes atuais e não entram nessa suíte.
  - Documentar em [`docs/como-contribuir.md`](docs/como-contribuir.md) como rodar só essa suíte e como adicionar um novo algoritmo off-policy.

### Offline RL (aprender de um dataset fixo)

- [ ] **Adicionar suporte a treino offline**: aprender somente de transições gravadas, sem interagir com o ambiente durante o treino.
  - Motivação: separar o custo de coleta do custo de aprendizado, permitir reprocessar o mesmo dataset e comparar algoritmos off-policy sob os mesmos dados.
  - Relação com o item anterior: aqui o foco é **offline** (dataset congelado), não off-policy online. Os dois itens podem compartilhar o replay e os testes de aprendizado.
  - Coleta: script/comando para gravar `(obs, action, reward, next_obs, terminated)` a partir de políticas baseline, checkpoints ou execuções exploratórias; formato versionado (ex.: `.npz` ou memmap), seed e metadados da arena.
  - Treino: opção `--offline-dataset PATH` (nome a definir) que carrega o replay do disco e chama `learn()` sem `env.step`; manter o caminho online atual intacto.
  - Algoritmos: começar por `dqn`, `doubledqn_per_dueling` e `mka`; documentar por que PPO/A2C (on-policy) não se aplicam sem interação ou dados logged.
  - Avaliação: ambiente separado, seeds fixas; comparar com a política que gerou o dataset e com o mesmo algoritmo treinado online no mesmo orçamento de updates.
  - Checkpoint: gravar `algorithm`, identificador/versão do dataset e config da arena; retomada exige o mesmo dataset.
  - Testes: dataset pequeno e determinístico; overfit de um lote (loss cai e a política reproduz o alvo); retomada; dataset incompatível falha com mensagem clara.
  - CLI/docs: exemplos no README da arena e em [`docs/algoritmos.md`](docs/algoritmos.md); deixar explícita a diferença entre offline e off-policy online.

### Algoritmos com alta eficiência amostral

- [ ] **Pesquisar e comparar algoritmos que aprendem com menos amostras**, medindo custo em passos de ambiente (não em episódios ou horas de parede).
  - Motivação: o seek_coin é caro de simular; melhorar a eficiência amostral rende mais que acelerar o update. O objetivo é responder “qual algoritmo chega a uma política útil com menos `env.step`?”.
  - Candidatos a avaliar (priorizar 2–3):
    - **Discreto:** Rainbow (Double DQN + PER + dueling + n-step + distribucional + Noisy Nets); DQN com `n-step` e/ou `Noisy Nets` isolados.
    - **Contínuo:** SAC e TD3 (a economy usa `Box` e já tem `ppo_continuous`, `ddpg`, `td3` e `sac`).
    - **Offline:** CQL/IQL, se o item de offline RL avançar.
    - **Model-based:** DreamerV3/MCTS simples, se houver orçamento para um item separado.
  - Protocolo: mesmas seeds, mesma `ArenaConfig` e mesmo orçamento em **passos de ambiente**; avaliação gulosa periódica com seeds fixas; reportar média e desvio em 3+ seeds.
  - Métricas: passos até atingir um alvo de recompensa/moedas; área sob a curva; recompensa final no mesmo orçamento; tempo de update e memória como custo secundário.
  - Implementação: começar por um algoritmo isolado no contrato `arenas/algs/alg_<nome>` (ex.: `n-step` ou `Noisy Nets` dentro do DQN) antes de um Rainbow completo; manter o checkpoint e a CLI atuais.
  - Entregáveis: relatório comparativo em `docs/` (ou `arenas/seek_coin/docs/`), pelo menos um algoritmo novo com testes, e uma tabela de “quando usar” em [`docs/algoritmos.md`](docs/algoritmos.md).
  - Testes: reprodutibilidade por seed, contrato de checkpoint, curva de aprendizado em ambiente pequeno e comparação com o `dqn` no mesmo orçamento de passos.
  - Fora de escopo inicial: tuning extensivo de hiperparâmetros e implementação de world models completos.

### GRPO — PPO sem crítico (vantagem relativa ao grupo)

- [ ] **Implementar o GRPO (Group Relative Policy Optimization), que mantém o recorte (clipping) do PPO e dispensa a rede de valor**, usando a vantagem relativa dentro de um grupo de rollouts da mesma condição inicial.
  - Ideia central ([DeepSeekMath](https://arxiv.org/abs/2402.03300), Shao et al., 2024): para cada condição inicial ("prompt" no paper) amostra-se um grupo de `G` trajetórias, calcula-se a recompensa de cada uma e normaliza-se dentro do grupo, `A_i = (r_i − mean(r)) / std(r)`; o update é o objetivo recortado do PPO, `min(ρ_i A_i, clip(ρ_i, 1−ε, 1+ε) A_i)`, opcionalmente com penalidade KL contra uma política de referência congelada. Não há crítico, GAE nem `value_coef`: metade das redes, checkpoint menor, updates mais rápidos e nenhuma dependência de uma função de valor aprendida (que é justamente a parte mais frágil em horizonte longo e recompensa esparsa).
  - Motivação no repositório: o `eppo` já explora variações do crítico; o GRPO testa o extremo oposto (nenhum crítico), é barato de treinar e é o algoritmo padrão em pipelines de RL com recompensa verificável — conecta com o item [Pré-treino com recompensas verificáveis](#pré-treino-com-recompensas-verificáveis).
  - Pacote: `arenas/algs/alg_grpo/` no mesmo contrato de [`alg_ppo`](arenas/algs/alg_ppo/agent.py) (`agent.py`, `model.py`, `rollout_buffer.py`), reaproveitando o encoder de [`encoders.py`](arenas/algs/encoders.py) para imagem `(C,H,W)` e vetor `(N,)`. Descoberta em [`alg_registry.py`](arenas/algs/alg_registry.py) é automática (sufixo do diretório → `--alg grpo`); falta registrar em [`algorithm_catalog.py`](arenas/common/algorithm_catalog.py) e em [`agent_factory`](arenas/common/agent_factory.py).
  - Definição do grupo (a decisão de projeto mais importante, porque a arena não tem "prompts"):
    - **por episódio**, `G` ambientes com a **mesma seed de episódio** e o mesmo adversário, cada um com uma política levemente diferente (amostragem do ator) → grupo natural, mas caro: episódios de 90 s × `G`;
    - **por segmento**, `G` cópias do mesmo estado obtido com `env.get_state()`/`set_state()` (já suportados pelo `ArenaEnv`) e `k` passos cada, agregando a recompensa do segmento → grupos baratos e comparáveis em passos de ambiente. Começar por esta opção e manter a de episódio como variante.
  - Parâmetros (`--agent-param`): `group_size` (ex.: 4–8), `clip`, `epochs`, `lr`, `kl_coef` (0 desliga; usar o estimador k3), `segment_steps` e `adv_eps` para o caso degenerado de grupo homogêneo (`std ≈ 0` → vantagens nulas, nunca `NaN`).
  - Checkpoint: gravar `algorithm: grpo`, `group_size`, `kl_coef` e `segment_steps` na `config`; **sem pesos de crítico**; se houver KL, guardar a política de referência de forma explícita (ou recriá-la a partir do primeiro snapshot da política, documentando a escolha). `evaluate.py`/`play.py` devem reconstruir pelo [`checkpoint`](arenas/common/checkpoint.py) como nos outros algoritmos, e `CheckpointPolicy` precisa aceitar um agente só com ator (não assumir `value` no `state_dict`).
  - Protocolo de comparação: `ppo` × `grpo` no nível 3 (S6), mesmas seeds e mesmo orçamento — declarar **dois orçamentos**: horas de parede (o do `rigorous_validation.py`) e **passos de ambiente**, porque `G` rollouts por update mudam a amostra consumida por hora; a análise primária usa a unidade `train_seed` (Wilcoxon pareado + Holm na família `{reward, coins}`), como no `EXP-20261120-ppo-vs-eppo-level3`.
  - Métricas: recompensa/moedas gulosa (primárias), custo por update, memória e tamanho do checkpoint (secundárias), variância das vantagens dentro do grupo e taxa de grupos homogêneos (diagnóstico) — o diagnóstico mostra quanto sinal o GRPO está realmente extraindo.
  - Riscos: vantagem relativa ao grupo **descarta a escala absoluta da recompensa** (não distingue "todos ruins" de "todos bons" no mesmo update); grupos homogêneos zeram o gradiente; sem crítico não há *bootstrapping*, então horizonte de 90 s e recompensa esparsa podem exigir grupos maiores ou `kl_coef` maior. Registrar esses casos no relatório em vez de tunar até esconder.
  - Entregáveis: pacote `alg_grpo` com testes, entrada em `docs/algoritmos.md` + catálogo da Fila, uma proposta em `docs/experimentos/` (via skill `experimenter-helper`) e um YAML de experimento enfileirável com `depends_on`.
  - Testes: vantagens com média 0 e desvio 1 dentro do grupo; `std = 0` não gera `NaN`; ausência de crítico no checkpoint; `save`/`load` + `CheckpointPolicy`; reprodutibilidade por seed; smoke de treino no nível 3; regressão garantindo que `ppo`/`eppo` não mudaram.
  - Fora de escopo inicial: rollout distribuído entre máquinas, GRPO para RLHF com modelo de recompensa treinado e refinamentos de token-level KL do paper original.

### Investigar arquiteturas seq2seq/Transformer para agentes

- [ ] **Investigar como arquiteturas baseadas em seq2seq estilo Transformer podem produzir agentes melhores**, especialmente em observação parcial e histórico longo.
  - Motivação: os algoritmos atuais usam frame stack fixo e CNN/MLP; isso limita memória e generalização para níveis 4/5 (visão local e mundo contínuo). Um modelo de sequência pode aprender a integrar histórico variável.
  - Linhas a estudar: Decision Transformer, Trajectory Transformer, Gato-style sequence models e, mais simples, um encoder Transformer causal sobre a sequência de observações/ações em um agente PPO/DQN.
  - Escopo inicial: implementar um encoder Transformer causal (ou GRU como baseline) que recebe uma janela de observações e devolve um embedding para o ator/crítico, mantendo o contrato `Agent` e o espaço de ações existente.
  - Comparações: frame stack fixo × janela Transformer × GRU, em nível 4 (visão local), com o mesmo orçamento em passos e seeds.
  - Pontos de atenção: custo O(T²), memória da sequência, masking causal, variável comprimento de episódio, batching de sequências, normalização e posição/tempo.
  - Offline/Behavior Cloning: avaliar se o modelo seq2seq ajuda a aprender de trajetórias gravadas (liga com offline RL e pré-treino com recompensas verificáveis).
  - Checkpoint: gravar arquitetura, `context_length`, `d_model`, número de camadas/cabeças e versão; validar shape e retomada.
  - Testes: causality do mask (não vê o futuro), shape/variável de contexto, batch de sequências, checkpoint, seed e smoke de treino no nível 4.
  - Documentação: [`docs/algoritmos.md`](docs/algoritmos.md), README da arena e aula HTML se o resultado for adotado.
  - Fora de escopo inicial: implementar um Gato completo multimodal ou treinar um Transformer de trajetória em larga escala; primeiro medir se o encoder causal ajuda em um ambiente pequeno.

### Pré-treino com recompensas verificáveis

- [ ] **Usar pré-treino com recompensas verificáveis (RLVR) para inicializar agentes antes do treino principal**, aproveitando tarefas cuja recompensa pode ser checada por código.
  - Motivação: tarefas com verificação objetiva (chegou à plataforma, coletou N moedas, capturou o território, respeitou o capital mínimo) dão sinal limpo e barato para aprender representações e habilidades antes do objetivo final, que costuma ser esparso.
  - Definir por arena um conjunto de **tarefas verificáveis**:
    - **seek_coin:** alcançar uma coordenada/plataforma, coletar N moedas em T passos, empurrar o adversário para fora de uma região, sobreviver sem morrer.
    - **risk:** executar uma sequência legal de reforço/ataque/fortificação, capturar um continente, terminar o turno sem ação inválida.
    - **economy:** manter capital acima do mínimo, respeitar limite regulatório, atingir um RAROC alvo em um cenário controlado.
  - Verificador: função pura `verify(episode_info, task) -> {success, score, details}` por tarefa, testada e versionada; a recompensa de pré-treino vem do verificador, não de shaping denso.
  - Pipeline: gerar tarefas/currículo → pré-treinar com a recompensa verificável → salvar checkpoint `pretrain.pt` → fine-tune no ambiente principal carregando só os pesos.
  - Contrato: mesma observação/ação da arena; gravar `pretrain_task`, `verifier_version` e config no checkpoint; o fine-tune não deve depender do verificador em tempo de execução.
  - Avaliação: comparar from-scratch × pré-treinado com o mesmo orçamento de fine-tune, mesmas seeds e tarefas de teste held-out; medir passos até o alvo e desempenho final.
  - Riscos: reward hacking do verificador, distribuição de tarefas estreita demais e custo de geração. Mitigar com tarefas held-out, verificadores independentes e limite de orçamento.
  - Entregáveis: verificadores por arena, script de pré-treino, pelo menos um experimento comparativo em [`docs/experimentos/`](docs/experimentos/README.md), documentação e testes dos verificadores.
  - Fora de escopo inicial: usar LLMs ou modelos de linguagem como verificadores; o foco é verificação programática e determinística.

### The Option Keyboard

- [ ] **Implementar ambiente e algoritmo do paper "The Option Keyboard: Combining Skills in Reinforcement Learning"** (Barreto, Borsa, Quan, Schaul, Silver, Hessel, Mankowitz, Zidek, Munos — NeurIPS 2019).
  - Ideia central: aprender *successor features* `ψ(s, a)` e um conjunto de opções para vetores de pesos de cumulantes `w`; a política combinada usa GPI sobre `Q_w(s, a) = ψ(s, a) · w`; o "teclado de opções" combina convexamente as opções aprendidas para generalizar a tarefas novas sem treinar do zero.
  - Ambiente: usar `arenas/seek_coin` com cumulantes configuráveis (moedas, empurrões, sobrevivência) e `w` amostrado por episódio; se o acoplamento com o Arena ficar artificial, criar `arenas/option_keyboard/` com um gridworld dedicado e recompensa `r = φ(s, a) · w`.
  - Algoritmo: `arenas/algs/alg_option_keyboard/` seguindo o contrato de `arenas/algs/alg_dqn/agent.py` — aprender `ψ` e as opções-base (ex.: vértices/cantos do simplex de `w`), calcular o Q combinado por GPI no momento de agir e escolher/combinar opções para um `w` alvo.
  - Avaliação: tarefas com `w` retidos (fora do conjunto de treino), comparando com treinar cada `w` do zero; medir a generalização do teclado.
  - Checkpoint: gravar `algorithm`, `arch_version`, conjunto de opções, vetores `w` de treino e `obs_shape`; retomada com o mesmo conjunto.
  - Testes: reprodutibilidade por seed, GPI ≥ cada opção individual no conjunto de treino, formato de observação/ação e retomada.
  - CLI: `--alg option_keyboard`, opções para pesos/cumulantes de treino e `--eval-weights` para as tarefas retidas.

## Concluídos

> Os detalhes abaixo preservam o plano original; use-os como registro do
> que foi implementado, não como tarefas em aberto.

### Aba de algoritmos por arena na Fila de Experimentos

> Implementado em
> [`arenas/common/algorithm_catalog.py`](arenas/common/algorithm_catalog.py)
> (lista por arena, metadados e links), `arenas/registry.py`
> (`ArenaSpec.algorithms` e `algorithms_for_arena`),
> [`arenas/common/job_queue_ui.py`](arenas/common/job_queue_ui.py) (aba
> **Algoritmos**) e `scripts/job_queue_server.py` (`GET /api/algorithms` e
> `/docs/<caminho>`, que renderiza Markdown). Documentação em
> [`docs/fila-de-experimentos.md`](docs/fila-de-experimentos.md), com prévia em
> [`docs/fila-de-algoritmos.png`](docs/fila-de-algoritmos.png). Testes:
> `test_algorithm_catalog.py` e `test_queue_workbench_api.py`.

- [x] **Adicionar à Fila uma aba com os algoritmos implementados por arena e links para a documentação de cada algoritmo de treino.**
  - Lista derivada do `--alg` de cada arena: seek_coin descobre os pacotes
    não contínuos em `arenas/algs/`; risk e economy usam a lista explícita de
    `ArenaSpec.algorithms`.
  - Cada linha mostra tipo (on/off-policy, discreto/contínuo, multiagente) e
    links para a aula HTML do algoritmo, o catálogo `docs/algoritmos.md` e o
    README da arena.
  - O endpoint `/docs/<caminho>` serve documentos do repositório; Markdown é
    renderizado com tabelas, blocos de código e âncoras, e caminhos fora da
    raiz são recusados.

### Abas de análise e proposta na Fila de Experimentos

> Implementado em [`arenas/common/job_queue_ui.py`](arenas/common/job_queue_ui.py)
> (abas Fila, Análise de resultados e Proposta),
> [`arenas/common/experiment_analysis.py`](arenas/common/experiment_analysis.py)
> (descoberta de resultados, unidade primária/secundária e relatório via
> `report.py`/`report_html.py`),
> [`arenas/common/experiment_proposal.py`](arenas/common/experiment_proposal.py)
> (geração/validação MD+YAML, compartilhada com a skill) e
> [`scripts/job_queue_server.py`](scripts/job_queue_server.py) (endpoints).
> Documentação em [`docs/fila-de-experimentos.md`](docs/fila-de-experimentos.md)
> e prévia em [`docs/fila-de-proposta.png`](docs/fila-de-proposta.png).
> Testes: `test_experiment_analysis.py`, `test_experiment_proposal.py` e
> `test_queue_workbench_api.py`.

- [x] **Adicionar duas abas à interface da Fila de Experimentos: “Análise de resultados” e “Proposta de experimentos”.**
  - Motivação: hoje a Fila executa e mostra YAML/artefatos, mas a análise estatística e a criação de propostas ficam em scripts e arquivos separados. As abas aproximam o ciclo completo: propor → executar → analisar → decidir.
  - Aba **Análise de resultados**:
    - selecionar um experimento/job e carregar `results.csv`, `results-by-train-seed.csv` e artefatos;
    - mostrar descritivas, IC, tamanho de efeito, p-valor corrigido e avisos de pseudo-replicação;
    - permitir escolher métrica, baseline, unidade independente, correção e `--aggregate`;
    - reutilizar `arenas/common/report.py` e `report_html.py` em vez de duplicar estatística;
    - exibir gráficos de médias/IC, distribuição por seed, diferença A−B e efeito × p corrigido;
    - deixar explícito quando a análise é primária (seed de treino) ou secundária (episódios).
  - Aba **Proposta de experimentos**:
    - formulário guiado pelos campos de [`docs/experimentos/TEMPLATE.md`](docs/experimentos/TEMPLATE.md) e da skill `experimenter-helper`;
    - perguntar antes de executar: pergunta, hipótese, falsificação, métrica primária, baseline, seeds, orçamento, tuning, avaliação, estatística e critérios de decisão;
    - validar o YAML com o validador da skill, que delega para `experiment_proposal.py`;
    - gerar prévia do Markdown e do YAML, com opção de salvar em `docs/experimentos/EXP-AAAAMMDD-slug.{md,yaml}` sem sobrescrever;
    - opcionalmente enfileirar a proposta via `POST /api/experiments`.
  - Backend: endpoints para validar/gerar proposta, listar resultados de um experimento e montar o relatório; validar caminhos dentro do repositório e não aceitar shell arbitrário.
  - UX: manter a aba atual de fila/logs; usar abas separadas para não misturar operação com análise/proposta; preservar deep link `?job=<id>`.
  - Testes: validação de proposta inválida/válida, geração de MD+YAML, recusa de sobrescrita, caminho fora do repo, relatório de análise com dados sintéticos, troca de métrica/baseline e renderização HTML.
  - Fora de escopo inicial: editor WYSIWYG, colaboração multiusuário e execução automática sem confirmação.
  - Documentar em `docs/fila-de-experimentos.md` e atualizar a prévia da interface.

### Sistema de níveis do seek_coin (0 e 1)

> Implementado: `arenas/seek_coin/arena/levels.py` (nível 0 = mapa histórico;
> nível 1 = plataformas móveis), `--level` nos entry points e nível
> gravado/validado no checkpoint; o nível 1 tem duas plataformas móveis (a
> mais alta, vertical, e a central, horizontal).

- [x] **Sistema de níveis** — transformar o mapa fixo em níveis selecionáveis. O nível 0 é o mapa atual (`arena-v1`, plataformas fixas em `arenas/seek_coin/arena/engine.py`); o nível 1 introduz uma **plataforma flutuante móvel**.
  - Extrair a lista `self.platforms` (hoje fixa em `ArenaEngine.__init__`) para definições de nível, ex.: `arenas/seek_coin/arena/levels.py` com `LEVELS = {0: ..., 1: ...}`.
  - Adicionar `level: int = 0` em `ArenaConfig` (ou evoluir `map_version`, que hoje só existe como metadado) e expor `--level` em `arenas/seek_coin/train.py`, `evaluate.py` e `play.py`.
  - Gravar o nível no checkpoint e recusar retomada com nível diferente, como já é feito com `n_agents`/`team_ids`.
  - **Nível 1 — plataforma flutuante móvel**: plataforma que se desloca (horizontal e/ou vertical) por uma trajetória; atualizar a posição a cada substep de física, antes da colisão.
  - Física: personagem em cima da plataforma deve ser carregado junto (herdar o deslocamento horizontal) e continuar colidindo/aterrissando normalmente; moedas não devem nascer em posição inválida sobre a plataforma móvel.
  - Observação/render: o canal 0 de `render_grid` e todas as skins (`arena/renderer*.py`) já leem `engine.platforms`, mas hoje como tuplas estáticas — garantir que leiam a posição atualizada por substep.
  - Testes: mapa do nível 0 idêntico ao atual (regressão); nível 1 com plataforma móvel (movimento, colisão, caronagem e reprodução com seed).
  - Documentar os níveis e o `--level` no `CLAUDE.md` e no `README.md`.

### Nível 2 — areia e gelo

> Implementado: `--level 2`, hazards de areia/gelo com spawn aleatório, canais
> de observação extras e renderer default. Testes e smoke PPO/IPPO ok; treino
> noturno em `logs/overnight_l2`.

- [x] **Nível 2 com hazards temporários** — áreas de areia e gelo aparecem em posições aleatórias sobre as plataformas por alguns segundos e desaparecem/renascem.
  - Areia: personagem que passa por cima tem a aceleração/velocidade máxima reduzida (ex.: 30%).
  - Gelo: ao pisar, escorrega na horizontal numa direção aleatória por ~1/5 da largura da tela (128 px), com cooldown para não re-disparar.
  - Engine: `Hazard` (tipo, retângulo, timer, plataforma), spawn/respawn com RNG, acompanhar plataformas móveis e serializar em `get_state`/`set_state`.
  - Observação: canais extras só no nível 2 (areia/gelo) para não mudar o shape dos níveis 0/1; renderer default desenha os hazards.
  - Testes: spawn/timers, efeito da areia, distância do escorregão, canais de observação, shapes e checkpoint com `--level 2`.
  - Documentar no README/CLAUDE.

### Metodologia estatística para relatórios de experimentos

> Implementado em [`arenas/common/report.py`](arenas/common/report.py) e
> [`scripts/report.py`](scripts/report.py); documentação em
> [`docs/experimentos/README.md`](docs/experimentos/README.md).

- [x] **Criar uma metodologia padrão para todo experimento**, de modo que cada combinação cenário/nível + algoritmos produza um relatório com evidência estatística das diferenças de performance.
  - Motivação: média de uma seed não é evidência. O relatório precisa dizer **qual diferença é real, qual é ruído e qual é o tamanho do efeito**, com um protocolo repetível.
  - Desenho mínimo: pelo menos 3 seeds por algoritmo (ideal 5+), mesmo orçamento em passos de ambiente, mesma política de avaliação e mesmo conjunto de episódios de teste.
  - Métricas por execução: recompensa/moedas/RAROC, eficiência amostral (passos até o alvo), estabilidade (desvio/IQR) e taxa de falha/abortos.
  - Estatística descritiva: média, mediana, desvio-padrão, IQR e intervalo de confiança (bootstrap ou t, deixando explícito o método).
  - Teste de hipótese: comparar algoritmos com teste pareado por seed quando possível (ex.: Wilcoxon signed-rank) ou não pareado (Mann-Whitney); reportar estatística, p-valor e tamanho de efeito (Cohen's d ou Cliff's delta).
  - Múltiplas comparações: corrigir com Holm-Bonferroni ou Benjamini-Hochberg e deixar claro qual família de hipóteses foi corrigida.
  - Curvas: aprender com bandas de confiança; comparar eficiência amostral por passos, não por horas de parede.
  - Integração: `arenas/common/report.py` + `scripts/report.py` leem CSVs/JSONL de `experiments/<ID>/` e geram `report.md`; formato canônico documentado em [`docs/experimentos/README.md`](docs/experimentos/README.md).
  - Rigor: definir a métrica primária **antes** de rodar, não escolher depois; reportar todas as seeds, inclusive as ruins; não descartar outliers sem critério declarado.
  - Entregáveis: módulo de estatística + CLI + template de relatório + um experimento piloto com pelo menos 3 seeds; testes com dados sintéticos (diferença conhecida e ausência de diferença).
  - Critério de aceite: um comando lê uma pasta de experimento e produz um relatório com tabela por algoritmo, IC, teste, p-valor corrigido, tamanho de efeito e conclusão textual e interface HTML interativa; o relatório é reproduzível a partir dos CSVs.
  - Fora de escopo inicial: inferência bayesiana completa, modelos hierárquicos e meta-análise entre setups diferentes.

### Episódios de 90 s (horizonte mais longo)

> Implementado: `ArenaConfig.episode_seconds=90.0`, `--episode-seconds`
> em train/evaluate/play, duração gravada e aplicada pelo checkpoint.

- [x] **Aumentar a duração dos episódios do seek_coin de 45 s para 90 s**, de forma configurável e registrada no checkpoint.
  - Motivação: dar tempo para sequências mais longas (ex.: nível 3) e para o agente se recuperar de erros; ao mesmo tempo, medir o custo em eficiência amostral e em crédito de longo prazo.
  - Configuração: `--episode-seconds` em `train.py`, `evaluate.py` e `play.py`; o default virou 90. O valor é gravado no checkpoint e aplicado no evaluate/play.
  - Recalibrar o que depende do tempo: `coin_lifetime` (60 s), timers de hazard, respawn, penalidades por passo, currículo do `AutoOpponent` e orçamentos por `--max-hours`/episódios.
  - On-policy: verificar se `rollout_steps` (PPO 1024) e `n_steps` (A2C) continuam adequados para episódios de 90 s; ajustar o protocolo se necessário.
  - Avaliação: comparar 45 s × 90 s com o mesmo orçamento em **passos de ambiente** e as mesmas seeds; medir recompensa final, eficiência amostral e estabilidade. Registrar em [`docs/experimentos/`](docs/experimentos/README.md).
  - Testes: truncamento em 90 s, contagem de passos, `--max-hours`, checkpoint com `episode_seconds`, retomada com duração diferente e reprodutibilidade por seed.
  - Documentar no README/CLAUDE e em `ArenaConfig`.
  - Fora de escopo inicial: mudar o `frame_skip` ou a física para compensar o tempo; primeiro medir o efeito de só alongar o episódio.

### Nível 3 — ações sequenciais distantes no espaço e no tempo

> Implementado: `arena/levels.py` (botões/super moeda), `arena/engine.py`
> (ativação/cancelamento/coleta/expiração), canais extras em
> `arena/observations.py` e testes em `tests/test_level3.py`.

- [x] **Adicionar o nível 3 ao seek_coin**: recompensa alta exige uma sequência de ações distantes no espaço e no tempo, com um adversário competindo pela mesma oportunidade.
  - Exemplo de mecânica: um botão de ativação em um lado do mapa liga uma **super moeda** no lado oposto; o agente precisa atravessar o mapa e coletá-la antes que o adversário aperte o botão de cancelamento ou colete a super moeda primeiro. O reward da super moeda deve ser bem maior que o de uma moeda comum.
  - Estado do nível: botão de ativação, botão de cancelamento, super moeda e janela de tempo. Apenas uma super moeda ativa por vez; definir cooldown após coleta/cancelamento/expiração.
  - Interação: preferir ativação **por contato** (pisar no botão) para manter `Discrete(6)` e não quebrar checkpoints. Se for necessário um comando explícito, adicionar `INTERACT`, versionar a rede e o checkpoint e documentar o novo espaço de ações.
  - Adversário: a política baseline (`rule`/`auto`) também precisa saber ativar, cancelar e coletar a super moeda; no multiagente, todos aprendem a disputar. Sem isso, o nível vira um objetivo solitário e perde a parte adversarial.
  - Observação: canais extras só no nível 3 (botão ativo/inativo, super moeda, timer restante) para não mudar o shape dos níveis 0/1/2. Com `frame_stack`, o agente ainda precisa inferir a dinâmica; avaliar se o timer precisa de canal próprio.
  - Recompensa: valor alto e configurável para a super moeda (`super_coin_reward`), com opção de bônus para quem cancela/nega a jogada do adversário. Evitar shaping denso que entregue a resposta pronta; medir com e sem PBRS.
  - Configuração: `ArenaConfig.level=3`, parâmetros de posição/duração/valor/cooldown, validação no checkpoint e recusa de retomada com nível diferente.
  - Render: desenhar botões, estado ativo/inativo, super moeda e timer no renderer default (e nas skins que fizer sentido).
  - Testes: ativação por contato, spawn da super moeda no lado oposto, expiração da janela, cancelamento pelo adversário, coleta pelo agente e pelo adversário, canais de observação, shapes, seed reproduzível e checkpoint com `--level 3`.
  - Métricas: taxa de sucesso da sequência, tempo até a coleta, quantas vezes o adversário cancelou/coletou e reward médio com/sem super moeda. Registrar em [`docs/experimentos/`](docs/experimentos/README.md).
  - Documentar no README/CLAUDE e no [`arena/levels.py`](arenas/seek_coin/arena/levels.py).
  - Fora de escopo inicial: cadeias com 3+ botões e salas/portas; começar com uma ativação + uma super moeda + um cancelamento.

### Campanha BO: PPO × EPPO no nível 3

> Concluída em `experiments/BO-level3-6h-20261115/`; relatório em
> [`docs/experimentos/EXP-20261115-bo-level3-6h.md`](docs/experimentos/EXP-20261115-bo-level3-6h.md).

- [x] **Procurar os melhores hiperparâmetros do PPO e do EPPO para o nível 3 do seek_coin com busca bayesiana**, com orçamento de **6 h de parede para cada algoritmo** (12 h no total).
  - Depende de: item de otimização bayesiana e de configurações em YAML (o `best_params` vira um preset em `configs/seek_coin/`).
  - Condição: seek_coin nível 3, adversário primário `rule`; repetir os melhores com `coins` como validação de generalização. Seed de treino 7 nos trials; top configurações validadas com 3+ seeds.
  - Orçamento: 6 h por algoritmo para a campanha; usar multi-fidelidade (ex.: 10–15 min por trial, top 5 com 30–60 min) para caber 24–36 trials por algoritmo. Rodar um algoritmo por vez para não compartilhar GPU.
  - PPO — espaço de busca inicial: `lr` 1e-4..1e-3 (log), `batch_size` {64,128,256}, `rollout_steps` {512,1024,2048}, `epochs` {2,4,8}, `clip` 0.1..0.3, `entropy_coef` 0..0.02, `gae_lambda` 0.9..0.99, `max_grad_norm` 0.5..10.
  - EPPO — espaço de busca inicial: `lr` 1e-4..1e-3 (log), `batch_size` {64,128,256}, `rollout_steps` {1024,2048,4096}, `epochs` {4,10}, `clip` 0.1..0.3, `entropy_coef` 0..0.02, `gae_lambda` 0.9..0.99, `max_grad_norm` 0.5..10, `ucb_radius` 0..0.05, `evidential_coef` 0..0.05, `gae_variant` {`cor`,`ind`}.
  - Fixos: nível 3, `episode_seconds=90`, `frame_stack=4`, `frame_skip=3`, `seed=7` nos trials. Objetivo: reward médio da política gulosa em 50 episódios (seeds 1000..1049), reportando também moedas, mortes e eficiência em passos até o alvo.
  - Registro: trial log completo (config, orçamento, métrica, status), melhor config, incerteza da busca e artefatos; entrada em [`docs/experimentos/`](docs/experimentos/README.md) com relatório estatístico.
  - Entregáveis: `configs/seek_coin/ppo-level3.yaml` e `eppo-level3.yaml`, campanha reprodutível, relatório comparando PPO e EPPO **com as configs tunadas**, e uma conclusão sobre qual foi mais eficiente no nível 3.
  - Critério de aceite: as duas campanhas somam ~12 h, cada melhor config é reproduzível por seed, e a comparação final usa 3+ seeds e o relatório estatístico (IC, teste pareado e tamanho de efeito).
  - Riscos: BO overfittar o conjunto de avaliação, ruído de wall-clock e concorrência de GPU; mitigar com held-out de seeds, multi-fidelidade e execução sequencial.

### Otimização bayesiana de hiperparâmetros por cenário

> Implementado em `scripts/bo_campaign.py`: GP-EI genérico, espaços YAML
> em `configs/search_spaces/<alg>.yaml`, configuração base por arena e
> trials via `train.py --config`. Multi-fidelidade simples (trials + final).
> ASHA e validação multi-seed ficam no fluxo de validação rigorosa.

- [x] **Implementar busca bayesiana de hiperparâmetros por cenário**, para aumentar a confiança de que cada algoritmo foi bem explorado e que uma comparação não foi decidida por azar de uma configuração sub-ótima.
  - Motivação: comparar algoritmos com um único `lr`/batch/rollout é frágil. O objetivo é dar a cada algoritmo um orçamento de tuning semelhante e reportar a melhor configuração encontrada com uma medida de incerteza.
  - Escopo: um espaço de busca por algoritmo e por cenário (seek_coin níveis 0/1/2, risk, economy), com bounds e restrições (ex.: `lr`, batch, `gamma`, `epsilon`, rollout/épocas, `frame_stack`/`frame_skip` quando fizer sentido).
  - Algoritmo de busca: começar com um GP + Expected Improvement ou uma biblioteca leve (ex.: Optuna, BoTorch ou Ax) atrás de uma dependência opcional. Registrar a versão e o espaço de busca usado.
  - Orçamento: limitar por número de trials × passos de ambiente; usar multi-fidelidade (successive halving/ASHA) para descartar configurações ruins cedo.
  - Objetivo: métrica de avaliação gulosa (recompensa/moedas/RAROC), com 3+ seeds e held-out de seeds para evitar overfitting ao conjunto de tuning.
  - Integração: script `scripts/tune.py` (ou `arenas/common/tuning.py`) que gera trials, roda `train.py`/`train_agent` e grava `best_params.json`; permitir carregar a melhor configuração no treino (`--params`/`--config`).
  - Registro: cada campanha vira um experimento em [`docs/experimentos/`](docs/experimentos/README.md), com espaço de busca, trials, melhor config, orçamento e conclusão.
  - Entregáveis: módulo de tuning, CLI, uma campanha piloto em pelo menos um cenário, documentação em `docs/` e testes com um objetivo sintético (sem treinar rede de verdade).
  - Riscos: custo alto, ruído entre seeds e overfitting ao conjunto de avaliação. Mitigar com multi-fidelidade, held-out de seeds e limites explícitos de orçamento.
  - Critério de aceite: dado o mesmo orçamento, reportar para cada algoritmo a melhor configuração, a média/desvio no held-out e a incerteza da busca; a comparação final usa as configurações tunadas, não as de default.

### Configurações de execução em YAML

> Implementado em `arenas/common/yaml_config.py`, integrado a
> `train.py`/`evaluate.py`/`play.py` das três arenas; presets em
> [`configs/`](configs/README.md) e espaços de busca em
> `configs/search_spaces/`.

- [x] **Carregar a configuração de cada execução (arena + algoritmo) de um arquivo YAML**, em vez de depender de dezenas de flags repetidas na linha de comando.
  - Motivação: experimentos comparáveis precisam do mesmo conjunto de parâmetros registrado e reproduzível; hoje os comandos ficam espalhados em scripts/READMEs e é fácil esquecer uma flag.
  - Estrutura sugerida:
    ```yaml
    arena: seek_coin
    setup: S4
    algorithm: ppo
    seed: 7
    episode_seconds: 90
    frame_stack: 4
    frame_skip: 3
    training:
      episodes: 1000000
      max_hours: 3
      batch_size: 64
      lr: 2.5e-4
    evaluation:
      episodes: 50
      seed: 1000
      checkpoint: latest
    ```
  - CLI: `--config caminho.yaml` e `--set chave=valor` em `train.py`/`evaluate.py`/`play.py`; precedência **CLI > YAML > defaults do ArenaConfig**.
  - Organização: `configs/seek_coin/{ppo,eppo,mappo,...}.yaml`, `configs/risk/...`, `configs/economy/...`; um preset por protocolo (ex.: `ppo-30min.yaml`, `ppo-level3.yaml`).
  - Validação: schema explícito, chaves desconhecidas com erro, tipos/intervalos validados e mensagem apontando o campo; não aceitar segredos/caminhos absolutos no preset.
  - Reprodutibilidade: gravar no checkpoint o YAML resolvido (ou hash + caminho) e o `run_name`; experimentos em [`docs/experimentos/`](docs/experimentos/README.md) apontam para o preset usado.
  - Integração com BO: o espaço de busca pode ser declarado no mesmo YAML (`search_space`) e o `best_params.json` vira um preset.
  - Testes: parsing, precedência CLI/YAML/default, chaves desconhecidas, tipos inválidos, `--set`, round-trip no checkpoint e execução de um smoke test com YAML.
  - Documentar no README da raiz e em `docs/como-contribuir.md`; exemplos por arena.
  - Fora de escopo inicial: orquestrador de múltiplos jobs, templates Jinja e configuração remota.

### Fila de Experimentos

> Implementada em `arenas/common/job_queue.py`, `job_queue_ui.py`,
> `scripts/job_queue_server.py` e `scripts/experiment.py`; documentação em
> [`docs/fila-de-experimentos.md`](docs/fila-de-experimentos.md). Além de
> jobs, a UI carrega o YAML do experimento e lista artefatos: logs,
> relatórios, vídeos, estatísticas e documentos.

- [x] **Criar uma Fila de Experimentos web para acompanhar e administrar os próximos processos**, integrada aos experimentos e scripts do repositório.
  - Motivação: treinos longos, campanhas de BO e validações rodam em sequência/paralelo; hoje o controle é manual (`nohup`, `ps`, `tail`), sem visão central de fila, status, logs e recursos.
  - Backend: um processo/serviço local que mantém a fila em JSON/SQLite, inicia jobs com `subprocess`/`Popen`, captura PID, exit code, stdout/stderr e timestamps, e permite pausar/cancelar/reordenar.
  - Modelo de job: comando, diretório de trabalho, algoritmo/arena, seed, orçamento (`--max-hours`), prioridade, dependências (ex.: só rodar EPPO depois do PPO), concorrência máxima e recurso (CPU/GPU).
  - Interface: lista com status (`planejado`, `rodando`, `concluído`, `falhou`, `cancelado`), progresso (episódios/horas), logs ao vivo, botões de iniciar/pausar/cancelar/reordenar, filtros por arena/algoritmo/seed e ETA.
  - Integração: carregar jobs de YAML/CSV (ver item de configurações em YAML), apontar para `train.py`, `scripts/bo_campaign.py` e `scripts/rigorous_validation.py`; ao terminar, exibir caminho dos checkpoints e relatórios e permitir gerar `report.html`.
  - Persistência e recuperação: sobreviver a reinício do serviço, detectar processos órfãos, não duplicar jobs e registrar histórico.
  - Concorrência e segurança: limite de jobs simultâneos, lock por GPU, bind em `127.0.0.1` por padrão, sem execução arbitrária de comandos pela web sem autenticação/local-only.
  - Testes: fila unitária (ordem, dependências, cancelamento), simulação com jobs curtos, persistência, API/CLI do backend e smoke da UI (sem depender de browser real).
  - Entregáveis: backend + UI, documentação em `docs/`, exemplo de fila YAML e integração com pelo menos um script real.
  - Fora de escopo inicial: multiusuário/autenticação complexa, agendamento distribuído em várias máquinas e Kubernetes.

### DDPG — implementação e testes

> Implementado em `arenas/algs/alg_ddpg/` e testado na `economy`
> (`arenas/common/tests/test_continuous_algorithms.py`).

- [x] **Implementar e testar o DDPG (Deep Deterministic Policy Gradient)** como algoritmo off-policy para ações contínuas, seguindo o contrato de `arenas/algs/alg_*`.
  - Motivação: a arena `economy` usa `Box`; o DDPG é a referência off-policy determinística e a base didática para TD3/SAC.
  - Pacote: `arenas/algs/alg_ddpg/` com `actor.py`, `critic.py`, `agent.py` e `__init__.py`; `Agent` deve expor `algorithm="ddpg"`, `arch_version`, `obs_shape`, `n_actions`, `online`/`target` (ator e crítico), `optimizer`, `buffer`, `steps`, `epsilon` (ou campo de ruído) e os métodos do contrato (`act`, `act_batch`, `learn`, `q_stats`, `head_weights`).
  - Algoritmo: ator determinístico `μ(s)`; crítico `Q(s,a)`; alvos separados para ator e crítico (Polyak/soft update `tau`); replay uniforme; ruído de exploração (ex.: Ornstein-Uhlenbeck ou ruído gaussiano com escala decrescente); `terminated` zera o bootstrap, `truncated` mantém.
  - Ações contínuas: `continuous = True`; recortar ações nos limites do `Box`; `set_action_bounds` como no `ppo_continuous`.
  - CLI: `--alg ddpg` deve funcionar em `train.py`/`evaluate.py`/`play.py` da arena `economy`; expor `--agent-param` para `tau`, `noise_std`, `noise_decay`, `policy_delay`, `warmup` etc.
  - Checkpoint: mesmo formato pesos-only; validar arquitetura, `obs_shape`, `n_actions` e config da arena; retomada restaura ator/crítico, alvos e otimizadores.
  - Testes: contrato de ação/observação, `act` dentro dos limites, `learn` com batch pequeno, soft update dos alvos, bootstrap com `terminated` vs `truncated`, checkpoint round-trip, reprodutibilidade por seed e smoke de 1–2 episódios na `economy`.
  - Documentação: [`docs/algoritmos.md`](docs/algoritmos.md), [`arenas/algs/README.md`](arenas/algs/README.md), README da `economy` e uma aula HTML opcional.
  - Fora de escopo inicial: TD3 e SAC têm itens próprios. Primeiro entregar DDPG correto e testado.

### TD3 — Twin Delayed DDPG (implementação e testes)

> Implementado em `arenas/algs/alg_td3/` com testes de `min(Q1,Q2)`,
> target smoothing, atraso do ator, checkpoint e smoke na `economy`.

- [x] **Implementar e testar o TD3 (Twin Delayed DDPG)** como evolução do DDPG para ações contínuas, seguindo o contrato de `arenas/algs/alg_*`.
  - Dependência: o item de DDPG deve estar pronto (TD3 reutiliza ator, replay, soft update e checkpoint).
  - Pacote: `arenas/algs/alg_td3/` com `actor.py`, `critic.py` (dois críticos), `agent.py` e `__init__.py`; `Agent` expõe `algorithm="td3"`, `arch_version`, `obs_shape`, `n_actions`, `online`/`target`, `optimizer`, `buffer`, `steps` e o contrato completo.
  - Três correções sobre o DDPG:
    - **Clipped double Q:** dois críticos `Q1`/`Q2`; o alvo usa `min(Q1, Q2)` para reduzir superestimação.
    - **Target policy smoothing:** ruído gaussiano recortado na ação do ator-alvo, com `noise_clip`.
    - **Delayed policy updates:** o ator e os alvos atualizam a cada `policy_delay` updates do crítico.
  - Ações contínuas: `continuous = True`; recorte nos limites do `Box`; `set_action_bounds`; exploração por ruído no ator.
  - CLI: `--alg td3` na `economy` (e qualquer arena `Box`), com `--agent-param` para `tau`, `policy_delay`, `noise_std`, `noise_clip`, `target_noise`, `target_noise_clip` etc.
  - Checkpoint: pesos-only com ator/críticos online+target, otimizadores e config; validar `obs_shape`/`n_actions`; retomada reproduzível por seed (com a ressalva de RNG não persistido).
  - Testes: `min(Q1,Q2)` no alvo, smoothing dentro de `noise_clip`, ator só atualiza no `policy_delay`, soft update, limites de ação, bootstrap `terminated` vs `truncated`, checkpoint round-trip, seed e smoke de 1–2 episódios na `economy`.
  - Comparação: DDPG × TD3 no mesmo orçamento de passos e seeds; reportar estabilidade e variância, não só média.
  - Documentação: [`docs/algoritmos.md`](docs/algoritmos.md), [`arenas/algs/README.md`](arenas/algs/README.md), README da `economy` e aula HTML opcional.
  - Fora de escopo inicial: SAC (outra linha de evolução) e tuning extensivo; primeiro entregar TD3 correto e testado.

### SAC — Soft Actor-Critic (implementação e testes)

> Implementado em `arenas/algs/alg_sac/` (contínuo, `economy`) e
> `arenas/algs/alg_sac_discrete/` (discreto, `seek_coin`); testes em
> `test_continuous_algorithms.py` e `test_sac_discrete.py`.

- [x] **Implementar e testar o SAC (Soft Actor-Critic)** como algoritmo off-policy para ações contínuas, seguindo o contrato de `arenas/algs/alg_*`.
  - Motivação: é a referência de eficiência amostral para `Box` e permite comparar com `ppo_continuous`, DDPG e TD3 na `economy`.
  - Pacote: `arenas/algs/alg_sac/` com ator gaussiano (`tanh`), dois críticos, `agent.py` e `__init__.py`; `Agent` expõe `algorithm="sac"`, `arch_version`, `obs_shape`, `n_actions`, `online`/`target`, `optimizer`, `buffer`, `steps` e o contrato completo.
  - Algoritmo: objetivo entropia-regularizado `Q(s,a) - α log π(a|s)`; dois críticos com `min(Q1,Q2)`; ator estocástico com reparametrization trick; alvos com soft update (`tau`); replay uniforme; `terminated` zera o bootstrap e `truncated` mantém.
  - Temperatura: começar com `alpha` fixo e depois implementar ajuste automático (`--agent-param alpha=auto` ou `target_entropy`), com log de `alpha` no stats/TensorBoard.
  - Ações contínuas: `continuous = True`; aplicar `tanh` e escala para o `Box`; `set_action_bounds`; guardar `log_prob` corrigido pelo jacobiano do `tanh`.
  - CLI: `--alg sac` na `economy` (e qualquer arena `Box`), com `--agent-param` para `tau`, `gamma`, `alpha`, `target_entropy`, `policy_lr`, `q_lr`, `batch_size`, `hidden_dim` etc.
  - Checkpoint: pesos-only com ator/críticos online+target, otimizadores, `alpha` (se automático) e config; validar `obs_shape`/`n_actions`; retomada reproduzível por seed (RNG não persistido).
  - Testes: `tanh` dentro dos limites, `log_prob` finito, `min(Q1,Q2)` no alvo, soft update, atualização do ator, ajuste de `alpha`, bootstrap `terminated` vs `truncated`, checkpoint round-trip, seed e smoke de 1–2 episódios na `economy`.
  - Comparação: SAC × `ppo_continuous` (e DDPG/TD3, quando existirem) no mesmo orçamento de passos e seeds; reportar eficiência amostral, estabilidade e variância.
  - Documentação: [`docs/algoritmos.md`](docs/algoritmos.md), [`arenas/algs/README.md`](arenas/algs/README.md), README da `economy` e aula HTML opcional.
  - Fora de escopo inicial: SAC discreto, distributional SAC e tuning extensivo; primeiro entregar o SAC contínuo correto e testado.

### ASHA para campanhas de BO

> Implementado em `scripts/asha_campaign.py`: rungs configuráveis, eta,
> promoção/poda assíncrona, propostas GP-EI para o rung 0, YAML base e
> `train.py --config`. Testes em `arenas/common/tests/test_asha_campaign.py`.

- [x] **Adicionar ASHA (Asynchronous Successive Halving Algorithm) às campanhas de BO**, para podar configurações ruins cedo e promover as boas para orçamentos maiores.
  - Rungs configuráveis (`--rungs 5,15,30,60`), `eta`, mínimo de completions antes de promover/podar e limite de paralelismo.
  - Trials usam os mesmos YAMLs de base/espaço da BO; promoção reutiliza o melhor config e aumenta o orçamento.
  - Propostas de novas configs no rung 0 usam GP-EI após os primeiros trials.
  - Persistência em `trials.jsonl`, `results.csv` e `best.json`.
  - Fora de escopo inicial: agendamento distribuído e UI dedicada (a fila web pode executar o script).

### Estudo: Empirical Design in Reinforcement Learning (JMLR)

> Implementado em [`docs/experimentos/empirical-design.md`](docs/experimentos/empirical-design.md),
> no [`TEMPLATE.md`](docs/experimentos/TEMPLATE.md), na skill
> [`experimenter-helper`](.agents/skills/experimenter-helper/SKILL.md) e nas
> ferramentas de relatório/BO/validação. A auditoria dos experimentos
> existentes ficou deliberadamente fora de escopo nesta rodada.

- [x] **Estudar o paper “Empirical Design in Reinforcement Learning”** (Patterson, Neumann, White & White, JMLR; [arXiv:2304.01315](https://arxiv.org/abs/2304.01315)) e transformar as recomendações em prática no repositório.
  - Motivação: os experimentos do repo já usam seeds, IC e testes, mas faltava uma revisão sistemática de desenho empírico: escolha de métricas, orçamento, baselines, tuning, seleção de checkpoint, relato de variância e comparação justa.
  - Entregável 1 — notas de leitura em [`docs/experimentos/empirical-design.md`](docs/experimentos/empirical-design.md), com os princípios que se aplicam a RL empírico e o que não se aplica ao escopo didático do repo.
  - Entregável 2 — checklist acionável no [`TEMPLATE.md`](docs/experimentos/TEMPLATE.md): hipótese, falsificação, métrica primária, orçamento em passos, seeds, tuning, seleção de checkpoint, episódios de avaliação, IC/efeito, custo computacional e resultados negativos.
  - Entregável 3 — auditoria dos experimentos existentes: **fora de escopo nesta rodada**.
  - Entregável 4 — ajustes em ferramentas: `report.py` ganhou agregação por unidade independente (`--aggregate`) e aviso de seed repetida; `bo_campaign.py` ganhou métrica de seleção explícita e validação final com seed separada; `rigorous_validation.py` registra a unidade independente e separa análise primária/secundária.
  - Entregável 5 — guia curto em [`docs/como-contribuir.md`](docs/como-contribuir.md) sobre como projetar e reportar um experimento sem inflar significância.
  - Fora de escopo inicial: reproduzir os benchmarks do paper ou implementar novos algoritmos; o foco é metodologia e auditoria do repo.

### Log de job local na Fila de Experimentos

> Investigação e correção em [`arenas/common/queue_gate.py`](arenas/common/queue_gate.py);
> testes em `arenas/common/tests/test_queue_gate.py`.

- [x] **Verificar por que o log de uma execução local não aparecia na Fila de Experimentos.**
  - Causa: jobs `local` são processos externos; o servidor não tem acesso ao
    `stdout`/`stderr` deles. O `log_path` era criado no estado do job, mas
    ninguém escrevia no arquivo. Só jobs `managed` (iniciados pelo servidor)
    tinham `Popen(stdout=handle)`.
  - Correção: `queue_gate` resolve o `log_path` devolvido pela fila e duplica
    `stdout`/`stderr` para o terminal e para o arquivo enquanto o job local
    espera permissão e treina.
  - O log inclui mensagens da fila, espera, início, saída do treino e erros.
  - Teste de unidade: `test_queue_gate.py` cobre o `Tee`, a resolução do
    caminho e a captura de `stdout`/`stderr`.
  - Teste manual: `train.py --arena risk --alg dqn --episodes 1` via fila
    temporária gerou log no arquivo e apareceu em `/api/jobs/<id>/log`.
  - Limitação: jobs locais iniciados **antes** da correção não têm log
    retroativo.

### Sincronizar status do YAML com a Fila

> Implementado em [`arenas/common/job_queue.py`](arenas/common/job_queue.py),
> [`arenas/common/job_queue_ui.py`](arenas/common/job_queue_ui.py) e
> [`scripts/job_queue_server.py`](scripts/job_queue_server.py); testes em
> `arenas/common/tests/test_job_queue.py`.

- [x] **Manter o `status` do YAML de experimento coerente com o job na Fila de Experimentos.**
  - A UI mostra o status declarado no YAML e o **status efetivo** derivado do job: `queued→planned`, `running→running`, `done→done`, `failed/canceled→aborted`.
  - Divergências aparecem como `YAML: planned · efetivo: done (divergente)`.
  - O botão **Atualizar status do YAML** faz uma atualização explícita, atômica e restrita à linha `status:`, preservando comentários e ordem.
  - Endpoint: `POST /api/jobs/<id>/experiment/status`.
  - Testes cobrem status efetivo, sincronização, preservação de comentários e cancelamento → `aborted`.
  - Limitação: a sincronização é explícita; a fila não edita o YAML automaticamente.
