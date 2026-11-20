# `arenas/` — o núcleo do monorepo

Aqui ficam as arenas e tudo que elas compartilham.

```text
arenas/
├── registry.py     # catálogo ArenaSpec: nome, pacote, entry points, ações
├── cli.py          # despacho de --arena para o main() da arena
├── common/         # trainer, evaluator, checkpoint, fábrica de agentes
├── algs/           # algoritmos compartilhados (alg_<nome>)
├── seek_coin/      # arena 1: plataforma 2D
├── risk/           # arena 2: conquista de territórios
└── economy/        # arena 3: macroeconomia + banco
```

## Regra de ouro

- Código específico de uma arena fica **dentro** dela.
- O que serve a mais de uma arena vai para `common/` ou `algs/`.
- Uma arena **nunca** importa outra arena.

## Por onde começar

- Entender a arquitetura: [`docs/arquitetura.md`](../docs/arquitetura.md).
- Entender os algoritmos: [`docs/algoritmos.md`](../docs/algoritmos.md) e
  [`algs/README.md`](algs/README.md).
- Usar a infraestrutura comum: [`common/README.md`](common/README.md).
- Escolher uma arena: [`seek_coin`](seek_coin/README.md),
  [`risk`](risk/README.md), [`economy`](economy/README.md).
- Adicionar uma arena: [`docs/como-contribuir.md`](../docs/como-contribuir.md).
