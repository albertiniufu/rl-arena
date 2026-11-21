# Checklist de Empirical Design

Use este checklist junto com `SKILL.md`. Ele é intencionalmente conservador:
prefira um experimento menor e bem controlado a um experimento grande e
confundido.

## A. Pergunta e claim

- [ ] A pergunta cabe em uma frase.
- [ ] A hipótese prevê uma direção/efeito.
- [ ] Existe um resultado que refutaria a hipótese.
- [ ] O claim máximo permitido está escrito.
- [ ] A pergunta não depende de uma métrica escolhida depois.

## B. Evidência existente

- [ ] Procurei em `docs/experimentos/registro.md`.
- [ ] Verifiquei `setups.md`.
- [ ] Li o README da arena e do algoritmo.
- [ ] Sei se o resultado existente tem 1 seed, tuning desigual ou checkpoint escolhido no teste.
- [ ] Se a lacuna é só analítica, não vou treinar de novo.

## C. Desenho

- [ ] Baseline forte incluído.
- [ ] Ablação incluída quando o claim é “X ajuda”.
- [ ] Mesmo setup para todos os braços.
- [ ] Mesmo orçamento em passos de ambiente (ou compute declarado).
- [ ] Mesma política de avaliação para todos.
- [ ] Controle de confounds: GPU compartilhada, ordem de execução, cache, horário.
- [ ] Se o ambiente mudou, novo setup registrado.

## D. Tuning

- [ ] Espaço de busca documentado.
- [ ] Orçamento de tuning igual para todos.
- [ ] Conjunto de validação separado do teste.
- [ ] BO/ASHA usam seeds de validação fixas.
- [ ] O melhor config é reproduzível por YAML.
- [ ] Não escolhi config olhando o resultado final.

## E. Seeds e poder

- [ ] 3+ seeds de treino; ideal 5+.
- [ ] Sei que 3 seeds dão pouco poder estatístico.
- [ ] Episódios de avaliação não são tratados como runs independentes.
- [ ] Vou reportar a análise primária por seed de treino.
- [ ] Vou reportar a análise secundária por episódio com a ressalva de pseudo-replicação.

## F. Avaliação

- [ ] Métrica primária definida antes.
- [ ] Métricas secundárias definidas.
- [ ] Número de episódios de avaliação definido.
- [ ] Seeds de avaliação definidas.
- [ ] Checkpoint escolhido por validação, não por teste.
- [ ] Política gulosa.
- [ ] Truncamento e `terminated` tratados corretamente.
- [ ] Episódios de avaliação não usam ε nem atualizam pesos.

## G. Análise estatística

- [ ] IC bootstrap ou método declarado.
- [ ] Teste pareado por seed quando possível.
- [ ] Teste não pareado quando o pareamento não existe.
- [ ] Tamanho de efeito reportado (Cohen's d ou Cliff's delta).
- [ ] Correção para múltiplas comparações (Holm/BH).
- [ ] Distribuição/mediana além da média.
- [ ] Runs falhas e resultados negativos incluídos.
- [ ] Nenhum outlier removido sem critério.

## H. Custo e reprodutibilidade

- [ ] Horas por run.
- [ ] Número de runs.
- [ ] GPU/CPU e memória.
- [ ] Trials de tuning.
- [ ] Versões de dependências.
- [ ] Comando exato.
- [ ] YAML/commit.
- [ ] Seed e checkpoint.

## I. Registro

- [ ] Entrada `docs/experimentos/EXP-...md` criada a partir do template.
- [ ] Entrada `docs/experimentos/EXP-...yaml` criada e validada.
- [ ] Markdown e YAML concordam (mesmas seeds, métricas, orçamento e critérios).
- [ ] Linha no `registro.md`.
- [ ] Setup novo em `setups.md`, se aplicável.
- [ ] Backlog atualizado, se for tarefa de engenharia.
- [ ] Relatório em Markdown/HTML apontado.

## Red flags

- “Vou rodar 1 seed e ver no que dá.”
- “Depois eu escolho a métrica que ficou melhor.”
- “Vou comparar com o default do baseline.”
- “Vou usar o melhor checkpoint no teste.”
- “Vou rodar 10 configurações e reportar a melhor.”
- “O p-valor deu significativo, então o efeito é grande.”
- “Não precisa registrar as runs que falharam.”
- “Vou comparar PPO com 6 h contra DQN com 30 min.”
- “O resultado foi bom no treino, então a política é boa.”
