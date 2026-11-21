# Atalhos para as tarefas mais comuns do monorepo.
#
#   make            # lista os alvos
#   make install    # cria .venv e instala as dependências
#   make test       # suíte completa
#   make smoke      # treino curtíssimo em cada arena
#
# Variáveis: ARENA (padrão seek_coin), ALG (padrão do registry),
# EPISODES (padrão 300), CHECKPOINT (padrão checkpoints/<arena>/best.pt).
# O interpretador é o `python` do ambiente ativo; use
# `make test PYTHON=.venv/bin/python` para apontar outro.

PYTHON ?= python
ARENA ?= seek_coin
ALG ?= dqn
EPISODES ?= 300
CHECKPOINT ?= checkpoints/$(ARENA)/best.pt

.PHONY: help install test test-arena smoke train evaluate play docs queue clean

help:
	@echo "Alvos disponíveis:"
	@echo "  install       cria .venv e instala requirements.txt"
	@echo "  test          pytest -q (suíte completa)"
	@echo "  test-arena    pytest -q arenas/\$$ARENA"
	@echo "  smoke         treino de 2 episódios em seek_coin, risk e economy"
	@echo "  train         python train.py --arena \$$ARENA --alg \$$ALG --episodes \$$EPISODES"
	@echo "  evaluate      python evaluate.py --arena \$$ARENA --checkpoint \$$CHECKPOINT"
	@echo "  play          python play.py --arena \$$ARENA --checkpoint \$$CHECKPOINT"
	@echo "  docs          serve as aulas HTML em http://localhost:8000"
	@echo "  queue         inicia a Fila de Experimentos em http://127.0.0.1:8765"
	@echo "  clean         remove __pycache__, .pytest_cache e .pyc"

install:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt
	@echo "Pronto. Ative com: source .venv/bin/activate"

test:
	$(PYTHON) -m pytest -q

test-arena:
	$(PYTHON) -m pytest -q arenas/$(ARENA)

smoke:
	$(PYTHON) train.py --no-queue --arena seek_coin --episodes 2 --checkpoint-dir checkpoints/smoke/seek_coin
	$(PYTHON) train.py --no-queue --arena risk --episodes 2 --checkpoint-dir checkpoints/smoke/risk
	$(PYTHON) train.py --no-queue --arena economy --alg ppo_continuous --episodes 2 --checkpoint-dir checkpoints/smoke/economy

train:
	$(PYTHON) train.py --arena $(ARENA) --alg $(ALG) --episodes $(EPISODES)

evaluate:
	$(PYTHON) evaluate.py --arena $(ARENA) --checkpoint $(CHECKPOINT)

play:
	$(PYTHON) play.py --arena $(ARENA) --checkpoint $(CHECKPOINT)

docs:
	@echo "Abrindo http://localhost:8000/arenas/seek_coin/docs/index.html"
	$(PYTHON) -m http.server 8000 --bind 127.0.0.1

queue:
	$(PYTHON) scripts/job_queue_server.py --state experiments/queue/queue.json

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
	find . -type f -name '*.pyc' -delete
