#!/usr/bin/env python3
"""Validação multi-seed de PPO/EPPO no nível 3.

Por padrão lê os presets fixos de `configs/seek_coin/{ppo,eppo}-level3.yaml`;
também aceita `--bo-root` para carregar os `best.json` de uma campanha BO.

Roda `--hours` por seed para os dois algoritmos, em pares PPO+EPPO por seed,
e avalia 50 episódios por run. Gera:

- `results.csv` — uma linha por episódio de avaliação (análise secundária);
- `results-by-train-seed.csv` — média por seed de treino (unidade
  independente; análise primária);
- relatórios estatísticos por seed e por episódio.

A análise primária emparelha por seed de treino. A análise por episódio é
secundária e não trata episódios da mesma run como runs independentes.

Uso:
    python scripts/rigorous_validation.py --hours 2 \
        --seeds 7,42,123 --eval-episodes 50 --eval-seed 2000 \
        --out experiments/EXP-20261120-ppo-vs-eppo-level3-10seeds

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
            hours: float, eval_episodes: int, eval_seed: int):
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
        "--arena", "seek_coin", "--alg", algorithm, "--level", "3",
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--bo-root", default=None,
                        help="opcional: diretório da BO com ppo/eppo best.json")
    parser.add_argument("--ppo-config",
                        default="configs/seek_coin/ppo-level3.yaml")
    parser.add_argument("--eppo-config",
                        default="configs/seek_coin/eppo-level3.yaml")
    parser.add_argument("--out",
                        default="experiments/EXP-20261116-validacao-rigorosa")
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
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": params,
    }
    (out / "campaign.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    for seed in seeds:
        print(f"=== seed {seed}: PPO + EPPO em paralelo ===", flush=True)
        processes = []
        for algorithm in ("ppo", "eppo"):
            run_dir = out / f"seed-{seed}" / algorithm
            run_dir.mkdir(parents=True, exist_ok=True)
            if _done_marker(run_dir).exists():
                continue
            command = [
                sys.executable, str(ROOT / "train.py"),
                "--arena", "seek_coin", "--alg", algorithm, "--level", "3",
                "--episodes", "1000000", "--max-hours", str(args.hours),
                "--opponent", "rule", "--seed", str(seed),
                "--frame-stack", "4", "--frame-skip", "3",
                "--save-every", "10", "--checkpoint-dir", "checkpoints",
            ]
            for key, value in params[algorithm].items():
                command += ["--agent-param", f"{key}={value}"]
            log = (run_dir / "train.log").open("w", encoding="utf-8")
            process = subprocess.Popen(
                command, cwd=run_dir, stdout=log,
                stderr=subprocess.STDOUT)
            processes.append((algorithm, run_dir, process, log))
        for algorithm, run_dir, process, log in processes:
            rc = process.wait()
            log.close()
            if rc != 0:
                print(f"[erro] treino {algorithm} seed={seed} rc={rc}",
                      flush=True)
                continue
            eval_command = [
                sys.executable, str(ROOT / "evaluate.py"),
                "--arena", "seek_coin",
                "--checkpoint", "checkpoints/latest.pt",
                "--episodes", str(args.eval_episodes),
                "--seed", str(args.eval_seed),
            ]
            eval_log = (run_dir / "eval.log").open("w", encoding="utf-8")
            result = subprocess.run(
                eval_command, cwd=run_dir, stdout=eval_log,
                stderr=subprocess.STDOUT, check=False,
                timeout=args.eval_episodes * 30.0 + 600.0)
            eval_log.close()
            if result.returncode != 0:
                print(f"[erro] avaliação {algorithm} seed={seed}", flush=True)
                continue
            _done_marker(run_dir).write_text(
                time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
            print(f"[ok] {algorithm} seed={seed}", flush=True)

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

    for metric in ("reward", "coins", "deaths"):
        _report(metric, out / "results-by-train-seed.csv",
                out / f"report-by-seed-{metric}.md")
        _report(metric, out / "results.csv",
                out / f"report-per-episode-{metric}.md")
    print("resultados e relatórios gerados em", out, flush=True)
    print("análise primária: report-by-seed-*.md (unidade = seed de treino)",
          flush=True)
    print("análise secundária: report-per-episode-*.md "
          "(episódios não são runs independentes)", flush=True)


if __name__ == "__main__":
    main()
