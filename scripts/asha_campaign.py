#!/usr/bin/env python3
"""Campanha ASHA (Successive Halving assíncrono) para hiperparâmetros.

Usa os mesmos YAMLs de `bo_campaign.py`:
- configuração base (`--base-config`);
- espaço de busca (`--space-dir/<alg>.yaml`);
- cada trial é um `train.py --config trial.yaml` seguido de `evaluate.py`.

Rungs padrão: 5, 15, 30 e 60 minutos. A cada rung, as melhores `1/eta`
configurações são promovidas; as demais são podadas. A decisão é refeita
conforme novos trials terminam, sem esperar todos os workers.

Exemplo:

    python scripts/asha_campaign.py --algorithms ppo,eppo \
        --base-config configs/seek_coin/base-level3.yaml \
        --space-dir configs/search_spaces \
        --rungs 5,15,30,60 --initial-trials 16 --eta 2 \
        --max-parallel 1 --budget-hours 6 \
        --out experiments/ASHA-level3

As saídas (`trials.jsonl`, `results.csv`, `best.json` e `campaign.json`) são
lidas pela aba **BO** da Fila de Experimentos; a faixa de orçamento usa os
rungs para separar o ganho por orçamento.
"""
from __future__ import annotations

import argparse
import csv
import os
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bo_campaign import (  # noqa: E402
    encode_params,
    load_space,
    random_params,
    run_trial,
    suggest,
)


def should_promote(scores: list[float], score: float, eta: int) -> bool:
    """Mantém as melhores `ceil(n / eta)` pontuações do rung."""
    if not scores:
        return True
    keep = max(1, math.ceil(len(scores) / int(eta)))
    threshold = sorted(scores, reverse=True)[keep - 1]
    return score >= threshold


@dataclass
class Task:
    params: dict
    rung: int
    index: int
    directory: Path


@dataclass
class Record:
    task: Task
    score: float
    reward: float
    coins: float
    deaths: float
    status: str
    resolved: bool = False


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def _write_results(path: Path, records: list[Record]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "index", "rung", "score", "reward", "coins", "deaths",
            "status", "params"])
        writer.writeheader()
        for record in records:
            writer.writerow({
                "index": record.task.index,
                "rung": record.task.rung,
                "score": record.score,
                "reward": record.reward,
                "coins": record.coins,
                "deaths": record.deaths,
                "status": record.status,
                "params": json.dumps(record.task.params, ensure_ascii=False),
            })


def _propose(space, completed, rng):
    if len(completed) < 5:
        return random_params(space, rng)
    X = [encode_params(space, record.task.params)
         for record in completed if record.status == "ok"]
    y = [record.score for record in completed if record.status == "ok"]
    return suggest(space, X, y, rng)


def run_algorithm(algorithm: str, space, base_config: dict, args) -> dict:
    out = Path(args.out) / algorithm
    out.mkdir(parents=True, exist_ok=True)
    rungs = [float(part) for part in args.rungs.split(",") if part.strip()]
    if not rungs:
        raise SystemExit("--rungs não pode ser vazio")
    eta = int(args.eta)
    initial = int(args.initial_trials)
    budget = args.budget_hours * 3600.0
    start = time.monotonic()
    rng = np.random.default_rng(args.seed)
    records: list[Record] = []
    by_rung: dict[int, list[Record]] = {i: [] for i in range(len(rungs))}
    scheduled = 0
    base_scheduled = 0
    lock = threading.Lock()
    results_path = out / "trials.jsonl"
    results_path.write_text("", encoding="utf-8")

    def remaining():
        return budget - (time.monotonic() - start)

    def make_task(params, rung, index):
        directory = out / f"trial_{index:03d}_r{rung}"
        return Task(params=params, rung=rung, index=index, directory=directory)

    def schedule(executor, task):
        nonlocal scheduled
        if task.rung >= len(rungs):
            return None
        if getattr(args, "rung_minutes_guard", True):
            needed = rungs[task.rung] * 60.0
            if remaining() < needed:
                return None
        scheduled += 1
        return executor.submit(
            run_trial, algorithm, task.params, task.directory,
            rungs[task.rung], args.eval_episodes, args.eval_seed,
            base_config)

    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {}
        # Começa com poucos trials; os próximos usam GP-EI sobre os
        # resultados do rung 0. Isso combina BO com a promoção/poda do ASHA.
        for _ in range(min(initial, max(1, args.max_parallel))):
            if remaining() <= rungs[0] * 60.0:
                break
            params = random_params(space, rng)
            task = make_task(params, 0, scheduled)
            future = schedule(executor, task)
            if future is not None:
                futures[future] = task
                base_scheduled += 1

        while futures:
            for future in as_completed(list(futures)):
                task = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"[{algorithm}] trial {task.index} falhou: {exc}",
                          file=sys.stderr, flush=True)
                    continue
                result.index = task.index
                record = Record(
                    task=task, score=(result.reward_mean
                                      + 0.05 * result.coins_mean
                                      if result.status == "ok" else float("-inf")),
                    reward=result.reward_mean, coins=result.coins_mean,
                    deaths=result.deaths_mean, status=result.status)
                records.append(record)
                by_rung[task.rung].append(record)
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "index": task.index, "rung": task.rung,
                        "params": task.params,
                        "status": result.status,
                        "reward": result.reward_mean,
                        "coins": result.coins_mean,
                        "deaths": result.deaths_mean,
                        "score": record.score,
                        "minutes": rungs[task.rung],
                    }, ensure_ascii=False) + "\n")
                print(f"[{algorithm}] r{task.rung} trial {task.index}: "
                      f"score={record.score:.3f} status={result.status}",
                      flush=True)

                # Resolve promoções deste rung quando há amostras suficientes.
                current = by_rung[task.rung]
                if len(current) < max(eta, args.min_completions):
                    continue
                scores = [r.score for r in current]
                for candidate in current:
                    if candidate.resolved:
                        continue
                    candidate.resolved = True
                    if candidate.task.rung + 1 >= len(rungs):
                        continue
                    if should_promote(scores, candidate.score, eta):
                        next_task = make_task(
                            candidate.task.params, candidate.task.rung + 1,
                            scheduled)
                        future = schedule(executor, next_task)
                        if future is not None:
                            futures[future] = next_task
                            print(f"[{algorithm}] promovido "
                                  f"{candidate.task.index} -> rung "
                                  f"{candidate.task.rung + 1}", flush=True)
                    else:
                        print(f"[{algorithm}] podado "
                              f"{candidate.task.index} no rung "
                              f"{candidate.task.rung}", flush=True)

                # Enquanto houver orçamento e meta de trials-base, propõe
                # uma nova config para o rung 0 usando GP-EI.
                if (base_scheduled < initial
                        and remaining() > rungs[0] * 60.0):
                    params = _propose(
                        space, [r for r in records if r.task.rung == 0], rng)
                    new_task = make_task(params, 0, scheduled)
                    future = schedule(executor, new_task)
                    if future is not None:
                        futures[future] = new_task
                        base_scheduled += 1

    _write_results(out / "results.csv", records)
    ok = [record for record in records if record.status == "ok"]
    if not ok:
        raise SystemExit(f"[{algorithm}] nenhum trial válido")
    best = max(ok, key=lambda record: record.score)
    summary = {
        "algorithm": algorithm,
        "best_params": best.task.params,
        "best_rung": best.task.rung,
        "best_minutes": rungs[best.task.rung],
        "best_score": best.score,
        "best_reward": best.reward,
        "best_coins": best.coins,
        "best_deaths": best.deaths,
        "n_trials": len(records),
        "rungs_minutes": rungs,
        "eta": eta,
        "elapsed_hours": (time.monotonic() - start) / 3600.0,
    }
    _write_json(out / "best.json", summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithms", default="ppo,eppo")
    parser.add_argument("--base-config",
                        default="configs/seek_coin/base-level3.yaml")
    parser.add_argument("--space-dir", default="configs/search_spaces")
    parser.add_argument("--rungs", default="5,15,30,60",
                        help="orçamentos em minutos, separados por vírgula")
    parser.add_argument("--eta", type=int, default=2)
    parser.add_argument("--initial-trials", type=int, default=16)
    parser.add_argument("--min-completions", type=int, default=2,
                        help="mínimo de trials por rung antes de promover/podar")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--budget-hours", type=float, default=6.0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="experiments/ASHA-generica")
    parser.add_argument("--no-rung-guard", action="store_true",
                        help="não exige orçamento restante para iniciar um rung")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    # Os trials são filhos deste script; não passam pelo portão da fila.
    os.environ.setdefault("RL_ARENA_QUEUE_BYPASS", "1")
    args.rung_minutes_guard = not args.no_rung_guard
    algorithms = [name.strip() for name in args.algorithms.split(",")
                  if name.strip()]
    base_config = yaml.safe_load(
        Path(args.base_config).read_text(encoding="utf-8")) or {}
    if not isinstance(base_config, dict) or "arena" not in base_config:
        raise SystemExit(
            f"{args.base_config} precisa ser um mapa com o campo 'arena'")
    spaces = {}
    for algorithm in algorithms:
        path = Path(args.space_dir) / f"{algorithm}.yaml"
        if not path.exists():
            raise SystemExit(f"espaço de busca não encontrado: {path}")
        spaces[algorithm] = load_space(path)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(
            f"{out} já existe e não está vazio; use outro --out")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "algorithms": algorithms,
        "base_config": str(args.base_config),
        "space_dir": str(args.space_dir),
        "rungs": args.rungs,
        "eta": args.eta,
        "initial_trials": args.initial_trials,
        "min_completions": args.min_completions,
        "max_parallel": args.max_parallel,
        "budget_hours": args.budget_hours,
        "eval_episodes": args.eval_episodes,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_json(out / "campaign.json", manifest)
    results = {}
    for algorithm in algorithms:
        print(f"=== ASHA {algorithm} ===", flush=True)
        results[algorithm] = run_algorithm(
            algorithm, spaces[algorithm], base_config, args)
        _write_json(out / "campaign.json", {**manifest, "results": results})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
