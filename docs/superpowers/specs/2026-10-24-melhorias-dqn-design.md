# Design: DQN aprendendo mais rápido (branch `melhorias-dqn`)

Data: 2026-10-24 · Status: em revisão (builder check)

## Intenção

Fazer o agente DQN do RL Arena aprender **mais rápido** — alcançar a mesma
qualidade de política com menos episódios (eficiência de amostra) — e reduzir o
**tempo de treino** (velocidade de parede). O projeto continua didático no uso,
mas as melhorias em si podem ser complexas.

Decisões do autor do projeto (Marcelo):
- Qualidade = aprender rápido (menos episódios para o mesmo resultado).
- Pode ficar mais complexo (sem restrição de simplicidade didática).
- Pode mexer no ambiente (observação, recompensas).
- O trabalho acontece no branch `melhorias-dqn`.

## Objetivos

- **G1**: recompensa de avaliação (política gulosa, `evaluate.py`) em N
  episódios de treino ≥ 2× a do baseline atual para o mesmo N.
- **G2**: tempo de parede por episódio de treino menor (alvo: ≥ 3× mais rápido).
- **G3**: tudo mensurável por A/B com seeds fixas — nenhuma melhoria entra sem
  ganho medido.

## Não-objetivos

- Não mudar a semântica do jogo (mapa, física, regras de morte/moeda).
- Não trocar a stack (Gymnasium + PyTorch + NumPy).
- Não buscar pontuação máxima absoluta — a métrica é velocidade de aprendizado.
- Não manter retrocompatibilidade de checkpoint com pesos antigos (ver
  Restrições).

## Anti-objetivos (o que tornaria o trabalho um fracasso mesmo "funcionando")

- Melhoria que ganha só em uma seed ou contra um adversário (precisa valer em
  várias).
- Overfitting ao currículo `auto` (explorar fraqueza de um adversário
  específico em vez de aprender a política geral).
- Ganho que só aparece com 1000+ episódios — precisa aparecer cedo.

## Restrições

- CPU apenas (8 núcleos; `torch.cuda.is_available() == False`).
- Python 3.10+, dependências atuais.
- CLI (`train.py`, `evaluate.py`, `play.py`) mantém as mesmas flags
  (`--stats`, `--tensorboard`, `--record`, `--resume`, `--opponent`).
- Testes atualizados junto (a observação muda de forma).
- Checkpoint: bump de `arch_version` em `dqn/checkpoint.py`; checkpoints antigos
  deixam de carregar (documentado no commit).

## Diagnóstico (por que o treino atual aprende devagar)

1. **Exploração morre cedo demais.** `epsilon *= 0.9995` a cada passo de
   aprendizado → o piso de 0.05 chega no episódio ~5 (~6,7 mil passos). Daí em
   diante o agente é 95% guloso com uma política de qualidade quase aleatória.
   É a causa número 1 de "aprender devagar" hoje.
2. **Observação não-Markoviana.** A grade `(5, 32, 20)` não tem velocidades nem
   temporizadores; o agente não distingue "caindo" de "subindo".
3. **Crédito demorado.** Recompensas +1 (moeda), +3 (eliminação, rara), −2
   (morte); com alvo de 1 passo e γ=0.99, o crédito se propaga lentamente.
4. **Superestimação.** `max` direto sobre a rede-alvo, sem Double DQN.
5. **1 ambiente sequencial, sem frame skip.** 1350 passos/episódio a 30 fps,
   um `learn()` por passo, tudo em Python na CPU.

## Proposta (fases; cada uma medida por A/B antes de seguir)

- **Fase 0 — instrumentação**: salvar checkpoint a cada orçamento fixo
  (50/100/150/300 episódios) para avaliar depois; script de avaliação por
  orçamento. Sem isso, não há como comparar curvas.
- **Fase 1 — exploração e warmup** (só agente): epsilon linear de 1.0 até o
  piso em ~20% dos episódios (em vez de decaimento por passo); piso 0.1;
  warmup de aprendizado (não aprende até o buffer ter ≥ 1000 transições).
  Esperado: maior ganho por linha de código do plano.
- **Fase 2 — ambiente**: empilhamento de 4 frames na observação `(20, 32, 20)`
  e frame skip de 3 (repete a ação por 3 passos, soma recompensas; episódio
  cai para ~450 passos). Atualiza testes, `obs_to_image` (mostra o frame
  atual), `evaluate.py` e `play.py`.
- **Fase 3 — algoritmo**: Double DQN (a ação do alvo é escolhida na rede
  online) + retorno n-step (n=3, γ³). Mudanças pequenas, alto impacto.
- **Fase 4 — replay priorizado + dueling + ajuste fino**: PER rank-based com
  α/β agendados; cabeça dueling (V + A); varredura de hiperparâmetros
  (lr 2.5e-4 vs 1e-3, batch 64/128, target update 500/1000 vs soft τ=0.005).
- **Fase 5 — velocidade de parede**: ambientes vetorizados (8 processos, um
  por núcleo), transições em lote, `learn()` a cada K passos; benchmark de
  passos/segundo antes/depois. GPU fica de fora: não há.

Fora do escopo desta rodada (minhas escolhas): reward shaping potencial
(distância à moeda mais próxima) e aprendizado assíncrono — candidatos a
fase 6 se 1–5 não bastarem.

## Protocolo de medição

- **Métrica**: recompensa média por episódio na política gulosa
  (`evaluate.py --seed 1000 --episodes 20`) nos orçamentos 50, 100, 150 e 300
  episódios de treino.
- **A/B**: 3 seeds (7, 42, 123) por configuração; adversário fixo `random`
  como primário (isola o aprendizado do agente); curva `auto` acompanhada como
  cenário real. Ganho = média das seeds supera a fase anterior em ≥ 15% no
  orçamento principal (150 eps).
- **Custo**: CPU, ~1 min por episódio hoje → 150 eps ≈ 2,5 h por corrida;
  3 seeds ≈ 7,5 h por configuração. Primeiro filtro com 100 eps; 300 eps só
  para as melhores candidatas. A Fase 5 barateia as corridas seguintes
  (investimento que se paga).

## Decisões do autor vs. escolhas minhas

- Do autor: aprender rápido é o objetivo; complexidade liberada; ambiente pode
  mudar; branch `melhorias-dqn`.
- Minhas (podem ser questionadas): ordem das fases; frame stack 4 / skip 3;
  n-step 3; PER rank-based; vetorização in-process; métrica de avaliação
  gulosa em orçamentos fixos; piso de epsilon 0.1; quebra de compatibilidade
  de checkpoints antigos; A/B com adversário fixo `random` como primário.
