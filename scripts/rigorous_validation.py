#!/usr/bin/env python3
"""Validação multi-seed de PPO/EPPO (nível configurável).

Por padrão lê os presets fixos de `configs/seek_coin/{ppo,eppo}-level3.yaml`;
também aceita `--bo-root` para carregar os `best.json` de uma campanha BO.
O nível, o `view_size` e o estado da super moeda vêm do preset (ou de
`--level`/`--view-size`/`--super-coin`), então o mesmo script serve para os
níveis 3, 4 (visão local) e 5 (mundo procedural).

Roda `--hours` por seed para os dois algoritmos, **um treino por vez**
(PPO, depois EPPO, seed a seed): a máquina é compartilhada, então dois
treinos simultâneos disputariam CPU/GPU e memória. Avalia 50 episódios por
run. Gera:

- `results.csv` — uma linha por episódio de avaliação (análise secundária);
- `results-by-train-seed.csv` — média por seed de treino (unidade
  independente; análise primária);
- `family-tests.md`/`.json` — Wilcoxon pareado por métrica e **Holm entre as
  métricas da família** declarada em `--family-metrics` (padrão `reward,coins`);
- relatórios estatísticos por seed e por episódio.

A análise primária emparelha por seed de treino. A análise por episódio é
secundária e não trata episódios da mesma run como runs independentes.

Uso:
    python scripts/rigorous_validation.py --hours 2 \
        --seeds 7,42,123 --eval-episodes 50 --eval-seed 2000 \
        --out experiments/EXP-20261120-ppo-vs-eppo-level3-10seeds

    python scripts/rigorous_validation.py --hours 2 --level 4 \
        --ppo-config configs/seek_coin/ppo-level4.yaml \
        --eppo-config configs/seek_coin/eppo-level4.yaml \
        --out experiments/EXP-20261121-ppo-vs-eppo-level4-7seeds

Pode ser retomado: runs com `DONE` são pulados.
"""
from __future__ import annotations

import argparse
import os
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_params(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("best_params") or payload


def _load_config_params(path: str | Path) -> dict:
    """Lê os hiperparâmetros fixos de um config YAML do treino."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    params = (payload.get("training") or {}).get("agent_params") or {}
    if not isinstance(params, dict) or not params:
        raise SystemExit(f"{path}: training.agent_params vazio")
    return params


def _load_config_scenario(path: str | Path) -> dict:
    """Lê nível/visão/super moeda de um preset (todos opcionais)."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    scenario = {}
    for key in ("level", "view_size"):
        if payload.get(key) is not None:
            scenario[key] = int(payload[key])
    if payload.get("super_coin") is not None:
        scenario["super_coin"] = bool(payload["super_coin"])
    return scenario


def _resolve_scenario(args, scenarios: dict) -> tuple[int, int | None,
                                                      bool | None]:
    """Prioriza CLI; depois exige que os dois presets concordem."""
    def unanimous(key):
        values = {scenario[key] for scenario in scenarios.values()
                  if key in scenario}
        if len(values) > 1:
            raise SystemExit(
                f"presets discordam em {key}: {sorted(values)}; use a flag "
                "da CLI para definir")
        return values.pop() if values else None

    level = args.level if args.level is not None else unanimous("level")
    if level is None:
        level = 3
    view_size = (args.view_size if args.view_size is not None
                 else unanimous("view_size"))
    if args.super_coin is not None:
        super_coin = bool(args.super_coin)
    else:
        super_coin = unanimous("super_coin")
    return int(level), view_size, super_coin


def _scenario_args(level: int, view_size: int | None,
                   super_coin: bool | None) -> list[str]:
    args = ["--level", str(level)]
    if view_size is not None:
        args += ["--view-size", str(view_size)]
    if super_coin is not None:
        args.append("--super-coin" if super_coin else "--no-super-coin")
    return args


def _run_command(command, cwd: Path, log_path: Path, timeout: float):
    with log_path.open("w", encoding="utf-8") as handle:
        try:
            return subprocess.run(
                command, cwd=cwd, stdout=handle,
                stderr=subprocess.STDOUT, check=False, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None


def _done_marker(run_dir: Path) -> Path:
    return run_dir / "DONE"


def run_one(algorithm: str, seed: int, params: dict, out: Path,
            hours: float, eval_episodes: int, eval_seed: int,
            level: int = 3, view_size: int | None = None,
            super_coin: bool | None = None):
    run_dir = out / f"seed-{seed}" / algorithm
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = _done_marker(run_dir)
    if marker.exists():
        print(f"[skip] {algorithm} seed={seed} já concluído", flush=True)
        return
    train_log = run_dir / "train.log"
    eval_log = run_dir / "eval.log"
    command = [
        sys.executable, str(ROOT / "train.py"),
        "--arena", "seek_coin", "--alg", algorithm,
        *_scenario_args(level, view_size, super_coin),
        "--episodes", "1000000", "--max-hours", str(hours),
        "--opponent", "rule", "--seed", str(seed),
        "--frame-stack", "4", "--frame-skip", "3",
        "--save-every", "10", "--checkpoint-dir", "checkpoints",
    ]
    for key, value in params.items():
        command += ["--agent-param", f"{key}={value}"]
    print(f"[run] {algorithm} seed={seed} ({hours:g} h)", flush=True)
    result = _run_command(command, run_dir, train_log,
                          timeout=hours * 3600.0 + 600.0)
    if result is None or result.returncode != 0:
        print(f"[erro] treino {algorithm} seed={seed} "
              f"rc={None if result is None else result.returncode}",
              flush=True)
        return
    eval_command = [
        sys.executable, str(ROOT / "evaluate.py"),
        "--arena", "seek_coin",
        "--checkpoint", "checkpoints/latest.pt",
        "--episodes", str(eval_episodes), "--seed", str(eval_seed),
    ]
    result = _run_command(eval_command, run_dir, eval_log,
                          timeout=eval_episodes * 30.0 + 600.0)
    if result is None or result.returncode != 0:
        print(f"[erro] avaliação {algorithm} seed={seed}", flush=True)
        return
    marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    print(f"[ok] {algorithm} seed={seed}", flush=True)


def _read_eval(run_dir: Path, algorithm: str, train_seed: int,
               eval_seed: int):
    path = run_dir / "logs" / "evaluation.csv"
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episode = int(row["episode"])
            rows.append({
                "algorithm": algorithm,
                "train_seed": train_seed,
                "eval_seed": eval_seed + episode - 1,
                "episode": episode,
                "seed": f"{train_seed}-{eval_seed + episode - 1}",
                "reward": float(row["reward"]),
                "coins": float(row["coins"]),
                "deaths": float(row["deaths"]),
                "duration_steps": int(row["duration_steps"]),
                "duration_seconds": float(row["duration_seconds"]),
            })
    return rows


def _write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_by_train_seed(rows: list[dict]):
    grouped = {}
    for row in rows:
        key = (row["algorithm"], row["train_seed"])
        grouped.setdefault(key, []).append(row)
    out = []
    for (algorithm, train_seed), values in sorted(grouped.items()):
        out.append({
            "algorithm": algorithm,
            "seed": train_seed,
            "n_eval": len(values),
            "reward": float(np.mean([v["reward"] for v in values])),
            "coins": float(np.mean([v["coins"] for v in values])),
            "deaths": float(np.mean([v["deaths"] for v in values])),
        })
    return out


def _report(metric: str, input_path: Path, out_path: Path):
    command = [
        sys.executable, "-m", "arenas.common.report",
        "--input", str(input_path), "--metric", metric,
        "--paired", "--correction", "holm", "--out", str(out_path),
    ]
    return subprocess.run(command, cwd=ROOT, check=False)


FAMILY_METRICS = ("reward", "coins")


def _pair_metrics(path: Path, metrics, baseline: str, candidate: str):
    """Lê o CSV por seed e pareia baseline × candidato seed a seed."""
    data = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            data[(row["algorithm"], int(row["seed"]))] = row
    seeds = sorted(
        seed for (algorithm, seed) in data
        if algorithm == baseline and (candidate, seed) in data)
    if not seeds:
        return {}
    return {
        metric: (
            [float(data[(baseline, seed)][metric]) for seed in seeds],
            [float(data[(candidate, seed)][metric]) for seed in seeds],
            seeds,
        )
        for metric in metrics
    }


def family_tests(out: Path, metrics=FAMILY_METRICS, baseline: str = "ppo",
                 candidate: str = "eppo", alpha: float = 0.05,
                 n_boot: int = 10000, rng: int = 7):
    """Wilcoxon pareado por métrica + **Holm na família declarada**.

    `report.py` corrige dentro de um relatório (aqui, um único par de
    algoritmos), então a correção entre métricas precisa ser feita aqui: é a
    família de hipóteses do experimento (`reward`, `coins`). Grava
    `family-tests.json` e `family-tests.md` no diretório do experimento.
    """
    from arenas.common.report import (
        bootstrap_diff_ci,
        cohens_d,
        holm_bonferroni,
        wilcoxon_signed_rank,
    )

    metric_list = [m.strip() for m in metrics if m and m.strip()]
    paired = _pair_metrics(out / "results-by-train-seed.csv", metric_list,
                           baseline, candidate)
    if not paired or min(len(a) for a, _, _ in paired.values()) < 2:
        print("[aviso] família de testes: pares insuficientes; nada gerado",
              flush=True)
        return None

    tests = []
    for metric in metric_list:
        values_a, values_b, seeds = paired[metric]
        statistic, p_value = wilcoxon_signed_rank(values_a, values_b)
        # IC e efeito na direção candidato − baseline (EPPO − PPO).
        ci_low, ci_high = bootstrap_diff_ci(values_b, values_a, paired=True,
                                            n_boot=n_boot, rng=rng)
        tests.append({
            "metric": metric,
            "n_pairs": len(seeds),
            "seeds": seeds,
            "baseline_mean": float(np.mean(values_a)),
            "candidate_mean": float(np.mean(values_b)),
            "diff": float(np.mean(values_b) - np.mean(values_a)),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "statistic": float(statistic),
            "p_raw": float(p_value),
            "effect_size": float(cohens_d(values_b, values_a, paired=True)),
        })

    adjusted = holm_bonferroni([test["p_raw"] for test in tests])
    for test, p_adjusted in zip(tests, adjusted):
        test["p_holm"] = float(min(1.0, p_adjusted))
        test["reject"] = bool(test["p_holm"] < alpha)

    payload = {
        "family": metric_list,
        "baseline": baseline,
        "candidate": candidate,
        "correction": "holm",
        "alpha": float(alpha),
        "independent_unit": "train_seed",
        "test": "wilcoxon_signed_rank",
        "tests": tests,
    }
    (out / "family-tests.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "family-tests.md").write_text(
        _family_markdown(payload), encoding="utf-8")
    return payload


def _family_markdown(payload: dict) -> str:
    tests = payload["tests"]
    lines = [
        "# Testes da família declarada",
        "",
        f"Família: {', '.join(f'`{m}`' for m in payload['family'])} · "
        f"correção {payload['correction']} · α = {payload['alpha']:g}",
        f"Comparação pareada por seed de treino: `{payload['baseline']}` "
        f"(baseline) × `{payload['candidate']}` · Wilcoxon signed-rank.",
        "",
        "| métrica | n | baseline | candidato | candidato − baseline | "
        "IC 95% | p bruto | p Holm | Cohen's d | decisão |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for test in tests:
        decision = "rejeita H0" if test["reject"] else "não rejeita H0"
        lines.append(
            f"| `{test['metric']}` | {test['n_pairs']} | "
            f"{test['baseline_mean']:.3f} | {test['candidate_mean']:.3f} | "
            f"{test['diff']:+.3f} | [{test['ci_low']:.3f}, "
            f"{test['ci_high']:.3f}] | {test['p_raw']:.4f} | "
            f"{test['p_holm']:.4f} | {test['effect_size']:.3f} | "
            f"{decision} |")
    lines += [
        "",
        f"n = {tests[0]['n_pairs']} seeds de treino; com essa família de "
        f"{len(tests)} métricas o menor p Holm possível é "
        f"{len(tests)} × {2 / 2 ** tests[0]['n_pairs']:.5f} = "
        f"{len(tests) * 2 / 2 ** tests[0]['n_pairs']:.5f}.",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--level", type=int, default=None,
                        help="nível (padrão: o do preset; sem preset, 3)")
    parser.add_argument("--view-size", type=int, default=None,
                        help="view_size do nível 4/5 (padrão: o do preset)")
    parser.add_argument("--super-coin", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="liga/desliga botões + super moeda "
                             "(padrão: o do preset; nível 4 desligado)")
    parser.add_argument("--bo-root", default=None,
                        help="opcional: diretório da BO com ppo/eppo best.json")
    parser.add_argument("--ppo-config",
                        default="configs/seek_coin/ppo-level3.yaml")
    parser.add_argument("--eppo-config",
                        default="configs/seek_coin/eppo-level3.yaml")
    parser.add_argument("--out",
                        default="experiments/EXP-20261116-validacao-rigorosa")
    parser.add_argument("--family-metrics", default="reward,coins",
                        help="métricas na família de testes corrigida por "
                             "Holm (padrão: reward,coins; as demais ficam "
                             "descritivas)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="nível de significância da família")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    # Os filhos treinam localmente; não passam pelo portão da fila.
    os.environ.setdefault("RL_ARENA_QUEUE_BYPASS", "1")
    args = parse_args(argv)

    if args.bo_root:
        bo_root = Path(args.bo_root)
        params = {
            "ppo": _load_params(bo_root / "ppo" / "best.json"),
            "eppo": _load_params(bo_root / "eppo" / "best.json"),
        }
        params_source = str(bo_root)
    else:
        params = {
            "ppo": _load_config_params(args.ppo_config),
            "eppo": _load_config_params(args.eppo_config),
        }
        params_source = f"{args.ppo_config}, {args.eppo_config}"
    scenarios = {
        "ppo": _load_config_scenario(args.ppo_config),
        "eppo": _load_config_scenario(args.eppo_config),
    }
    level, view_size, super_coin = _resolve_scenario(args, scenarios)
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "hours": args.hours,
        "seeds": seeds,
        "eval_episodes": args.eval_episodes,
        "eval_seed": args.eval_seed,
        "independent_unit": "train_seed",
        "secondary_unit": "eval_episode",
        "correction": "holm",
        "params_source": params_source,
        "ppo_config": args.ppo_config,
        "eppo_config": args.eppo_config,
        "level": level,
        "view_size": view_size,
        "super_coin": super_coin,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": params,
    }
    (out / "campaign.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    for seed in seeds:
        print(f"=== seed {seed}: PPO e EPPO em sequência ===", flush=True)
        # Um treino por vez: nada de PPO+EPPO simultâneos dividindo a máquina.
        for algorithm in ("ppo", "eppo"):
            run_one(algorithm, seed, params[algorithm], out, args.hours,
                    args.eval_episodes, args.eval_seed, level, view_size,
                    super_coin)

    rows = []
    for seed in seeds:
        for algorithm in ("ppo", "eppo"):
            run_dir = out / f"seed-{seed}" / algorithm
            if not (run_dir / "logs" / "evaluation.csv").exists():
                print(f"[aviso] sem avaliação: {run_dir}", flush=True)
                continue
            rows += _read_eval(run_dir, algorithm, seed, args.eval_seed)
    if not rows:
        raise SystemExit("nenhuma avaliação encontrada")
    _write_csv(out / "results.csv", rows)
    by_seed = _aggregate_by_train_seed(rows)
    _write_csv(out / "results-by-train-seed.csv", by_seed)

    family = family_tests(
        out, metrics=[m.strip() for m in args.family_metrics.split(",")],
        alpha=args.alpha)
    if family:
        for test in family["tests"]:
            decision = "rejeita" if test["reject"] else "não rejeita"
            print(f"[família] {test['metric']}: p={test['p_raw']:.4f} -> "
                  f"Holm={test['p_holm']:.4f} ({decision} H0 em "
                  f"α={family['alpha']:g})", flush=True)

    for metric in ("reward", "coins", "deaths"):
        _report(metric, out / "results-by-train-seed.csv",
                out / f"report-by-seed-{metric}.md")
        _report(metric, out / "results.csv",
                out / f"report-per-episode-{metric}.md")
    print("resultados e relatórios gerados em", out, flush=True)
    print("família de testes: " + ", ".join(FAMILY_METRICS) +
          " (Holm; ver family-tests.md)", flush=True)
    print("análise primária: report-by-seed-*.md (unidade = seed de treino)",
          flush=True)
    print("análise secundária: report-per-episode-*.md "
          "(episódios não são runs independentes)", flush=True)


if __name__ == "__main__":
    main()
