# Contribuindo

O guia completo está em [`docs/como-contribuir.md`](docs/como-contribuir.md).

Resumo rápido:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

- Documentação e comentários em **português**.
- Uma arena não importa outra; o compartilhado fica em `arenas/common/` e
  `arenas/algs/`.
- Novo algoritmo: crie `arenas/algs/alg_<nome>/agent.py` — o `--alg <nome>`
  passa a funcionar sozinho.
- Nova arena: crie `arenas/<nome>/` e registre em `arenas/registry.py`.
- Não versione `checkpoints/`, `logs/`, `experiments/` nem `videos/`.

Antes de abrir um PR, rode `make test` (ou `pytest -q`) e um `make smoke`
para o caminho real de treino.
