#!/usr/bin/env python3
"""Campanha de busca bayesiana de hiperparâmetros por algoritmo.

Implementa um GP com kernel RBF e Expected Improvement, sem dependências
além de NumPy e PyYAML. Cada trial gera um YAML temporário (base + params),
treina via `train.py --config`, avalia via `evaluate.py --config` e devolve
um score. O melhor trial pode ser reexecutado no fim da campanha.

Os espaços ficam em `configs/search_spaces/<alg>.yaml` e a configuração base
em `configs/<arena>/base-*.yaml`. Exemplo:

    python scripts/bo_campaign.py --algorithms ppo,eppo \
        --base-config configs/seek_coin/base-level3.yaml \
        --space-dir configs/search_spaces \
        --hours-per-alg 6 --trial-minutes 10 --final-minutes 30 \
        --eval-episodes 20 --final-eval-episodes 50 \
        --out experiments/BO-level3-6h

As saídas (`trials.jsonl`, `best.json` e `campaign.json`, uma subpasta por
algoritmo) são lidas pela aba **BO** da Fila de Experimentos, que mostra
convergência, efeito dos parâmetros, melhoria e a tabela de trials. Nada é
recalculado: a aba apenas lê os arquivos.
"""
from __future__ import annotations

import argparse
import os
import copy
import csv
import json
import math
import yaml
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Espaço de busca


@dataclass(frozen=True)
class Param:
    name: str
    kind: str  # float | int | choice
    low: float | None = None
    high: float | None = None
    choices: tuple = ()
    log: bool = False

    def encode(self, value) -> float:
        if self.kind == "choice":
            index = self.choices.index(value)
            return 0.5 if len(self.choices) == 1 else index / (len(self.choices) - 1)
        if self.log:
            return ((math.log(float(value)) - math.log(self.low))
                    / (math.log(self.high) - math.log(self.low)))
        return (float(value) - self.low) / (self.high - self.low)

    def decode(self, x: float):
        x = min(1.0, max(0.0, float(x)))
        if self.kind == "choice":
            index = min(len(self.choices) - 1,
                        int(round(x * (len(self.choices) - 1))))
            return self.choices[index]
        if self.log:
            value = math.exp(math.log(self.low)
                             + x * (math.log(self.high) - math.log(self.low)))
        else:
            value = self.low + x * (self.high - self.low)
        if self.kind == "int":
            return int(round(value))
        return float(value)

    def sample(self, rng) -> object:
        if self.kind == "choice":
            return self.choices[int(rng.integers(len(self.choices)))]
        if self.log:
            value = math.exp(rng.uniform(math.log(self.low),
                                         math.log(self.high)))
        else:
            value = rng.uniform(self.low, self.high)
        if self.kind == "int":
            return int(round(value))
        return float(value)


def load_space(path: str | Path) -> tuple[Param, ...]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"espaço de busca inválido: {path}")
    params = []
    for item in data.get("params", []):
        if not isinstance(item, dict) or "name" not in item:
            raise ValueError(f"parâmetro inválido em {path}: {item!r}")
        kind = str(item.get("type", "float"))
        if kind not in ("float", "int", "choice"):
            raise ValueError(f"tipo desconhecido em {path}: {kind!r}")
        choices = tuple(item.get("choices", ()))
        if kind == "choice" and not choices:
            raise ValueError(f"choice sem choices em {path}: {item!r}")
        params.append(Param(
            name=str(item["name"]), kind=kind,
            low=item.get("low"), high=item.get("high"),
            choices=choices, log=bool(item.get("log", False))))
    if not params:
        raise ValueError(f"espaço de busca vazio: {path}")
    return tuple(params)


def encode_params(space, params: dict) -> np.ndarray:
    return np.array([param.encode(params[param.name]) for param in space],
                    dtype=float)


def decode_vector(space, vector) -> dict:
    return {param.name: param.decode(x) for param, x in zip(space, vector)}


def random_params(space, rng) -> dict:
    return {param.name: param.sample(rng) for param in space}


# ---------------------------------------------------------------------------
# GP + Expected Improvement


def _normal_pdf(z):
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _normal_cdf(z):
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def _kernel(a, b, length_scale=0.30):
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    diff = a[:, None, :] - b[None, :, :]
    return np.exp(-0.5 * np.sum((diff / length_scale) ** 2, axis=2))


def suggest(space, X, y, rng, n_candidates=4000, xi=0.01) -> dict:
    """Sugere parâmetros; usa GP-EI depois de 4 trials, senão aleatório."""
    if len(X) < 4 or len(set(float(v) for v in y)) < 2:
        return random_params(space, rng)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    y_mean, y_std = float(y.mean()), float(y.std())
    if y_std < 1e-12:
        return random_params(space, rng)
    y_norm = (y - y_mean) / y_std

    K = _kernel(X, X)
    K += np.eye(len(X)) * 1e-6
    jitter = 1e-8
    for _ in range(8):
        try:
            L = np.linalg.cholesky(K + np.eye(len(X)) * jitter)
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        return random_params(space, rng)

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
    candidates = rng.random((n_candidates, len(space)))
    Ks = _kernel(candidates, X)
    mu = Ks @ alpha
    v = np.linalg.solve(L, Ks.T)
    variance = np.maximum(1e-9, 1.0 - np.sum(v * v, axis=0))
    sigma = np.sqrt(variance)
    best = float(y_norm.max())
    improvement = mu - best - xi
    z = improvement / sigma
    ei = improvement * _normal_cdf(z) + sigma * _normal_pdf(z)

    # Evita repetir exatamente um ponto já testado.
    seen = {tuple(np.round(row, 8)) for row in X}
    order = np.argsort(ei)[::-1]
    for index in order:
        candidate = candidates[index]
        if tuple(np.round(candidate, 8)) not in seen:
            return decode_vector(space, candidate)
    return random_params(space, rng)


# ---------------------------------------------------------------------------
# Execução de trials


@dataclass
class Trial:
    algorithm: str
    index: int
    params: dict
    reward_mean: float
    coins_mean: float
    deaths_mean: float
    score: float
    seconds: float
    status: str
    trial_dir: str
    error: str = ""
    extra: dict = field(default_factory=dict)


def selection_value(trial: Trial, metric: str = "score") -> float:
    """Valor usado para escolher o melhor trial (métrica primária da BO)."""
    if metric == "reward":
        return trial.reward_mean
    if metric == "coins":
        return trial.coins_mean
    return trial.score


def _mean_from_eval(path: Path) -> tuple[float, float, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV de avaliação vazio: {path}")
    rewards = [float(row["reward"]) for row in rows]
    coins = [float(row["coins"]) for row in rows]
    deaths = [float(row["deaths"]) for row in rows]
    return (float(np.mean(rewards)), float(np.mean(coins)),
            float(np.mean(deaths)))


def run_trial(algorithm: str, params: dict, trial_dir: Path,
              minutes: float, eval_episodes: int, eval_seed: int,
              base_config: dict, timeout_extra: float = 300.0) -> Trial:
    trial_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    train_log = trial_dir / "train.log"
    eval_log = trial_dir / "eval.log"

    train_cfg = copy.deepcopy(base_config)
    train_cfg["alg"] = algorithm
    training = train_cfg.setdefault("training", {})
    training.update({
        "episodes": 1_000_000,
        "max_hours": minutes / 60.0,
        "save_every": 10,
        "checkpoint_dir": "checkpoints",
        "agent_params": dict(params),
    })
    (trial_dir / "trial.yaml").write_text(
        yaml.safe_dump(train_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8")

    command = [sys.executable, str(ROOT / "train.py"),
               "--config", "trial.yaml"]
    error = ""
    with train_log.open("w", encoding="utf-8") as handle:
        try:
            subprocess.run(
                command, cwd=trial_dir, stdout=handle,
                stderr=subprocess.STDOUT, check=False,
                timeout=minutes * 60.0 + timeout_extra)
        except subprocess.TimeoutExpired:
            error = "timeout no treino"

    reward = coins = deaths = float("-inf")
    status = "ok"
    if not error:
        eval_cfg = copy.deepcopy(base_config)
        eval_cfg.pop("alg", None)
        eval_cfg.pop("algorithm", None)
        eval_cfg.pop("training", None)
        evaluation = eval_cfg.setdefault("evaluation", {})
        evaluation.update({
            "checkpoint": "checkpoints/latest.pt",
            "episodes": eval_episodes,
            "seed": eval_seed,
        })
        (trial_dir / "eval.yaml").write_text(
            yaml.safe_dump(eval_cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8")
        eval_command = [sys.executable, str(ROOT / "evaluate.py"),
                        "--config", "eval.yaml"]
        with eval_log.open("w", encoding="utf-8") as handle:
            try:
                subprocess.run(
                    eval_command, cwd=trial_dir, stdout=handle,
                    stderr=subprocess.STDOUT, check=False,
                    timeout=eval_episodes * 30.0 + 300.0)
            except subprocess.TimeoutExpired:
                error = "timeout na avaliação"
        if not error:
            try:
                reward, coins, deaths = _mean_from_eval(
                    trial_dir / "logs" / "evaluation.csv")
            except Exception as exc:  # noqa: BLE001
                error = f"avaliação inválida: {exc}"
    if error:
        status = "erro"
    score = (reward + 0.05 * coins) if status == "ok" else float("-inf")
    return Trial(
        algorithm=algorithm, index=-1, params=params,
        reward_mean=reward, coins_mean=coins, deaths_mean=deaths,
        score=score, seconds=time.monotonic() - started,
        status=status, trial_dir=str(trial_dir), error=error,
    )


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def run_algorithm(algorithm: str, space, base_config: dict,
                  args) -> dict:
    out = Path(args.out) / algorithm
    out.mkdir(parents=True, exist_ok=True)
    trials_path = out / "trials.jsonl"
    trials: list[Trial] = []
    X: list[np.ndarray] = []
    y: list[float] = []
    rng = np.random.default_rng(args.seed)
    start = time.monotonic()
    budget = args.hours_per_alg * 3600.0
    reserve = (args.final_minutes * 60.0 + 120.0
               if args.final_minutes > 0 else 0.0)
    printed_header = False

    while True:
        elapsed = time.monotonic() - start
        remaining = budget - elapsed
        if remaining < args.trial_minutes * 60.0 + 120.0 + reserve:
            break
        if args.max_trials and len(trials) >= args.max_trials:
            break
        params = suggest(space, X, y, rng)
        index = len(trials)
        if not printed_header:
            print(f"[{algorithm}] início: {budget/3600:.2f} h, "
                  f"trial={args.trial_minutes:g} min", flush=True)
            printed_header = True
        print(f"[{algorithm}] trial {index:02d}: {params}", flush=True)
        trial = run_trial(
            algorithm, params, out / f"trial_{index:03d}",
            args.trial_minutes, args.eval_episodes, args.eval_seed,
            base_config)
        trial.index = index
        trials.append(trial)
        if trial.status == "ok":
            X.append(encode_params(space, params))
            y.append(trial.score)
        with trials_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trial.__dict__, ensure_ascii=False) + "\n")
        print(f"[{algorithm}] trial {index:02d}: status={trial.status} "
              f"reward={trial.reward_mean:.3f} coins={trial.coins_mean:.2f} "
              f"score={trial.score:.3f}", flush=True)

    ok = [trial for trial in trials if trial.status == "ok"]
    if not ok:
        raise SystemExit(f"[{algorithm}] nenhum trial válido")
    best = max(ok, key=lambda trial: selection_value(
        trial, args.selection_metric))
    result = {
        "algorithm": algorithm,
        "best_trial": best.index,
        "best_params": best.params,
        "selection_metric": args.selection_metric,
        "selection_eval_seed": args.eval_seed,
        "best_low_fidelity": {
            "reward_mean": best.reward_mean,
            "coins_mean": best.coins_mean,
            "deaths_mean": best.deaths_mean,
            "score": best.score,
        },
        "n_trials": len(trials),
        "budget_hours": args.hours_per_alg,
        "elapsed_hours": (time.monotonic() - start) / 3600.0,
    }

    final_eval_seed = (args.final_eval_seed
                       if args.final_eval_seed is not None
                       else args.eval_seed + 1000)
    result["final_eval_seed"] = final_eval_seed
    if final_eval_seed == args.eval_seed:
        print(f"[{algorithm}] aviso: validação final repete a seed de "
              "seleção; use --final-eval-seed diferente", flush=True)
    remaining = budget - (time.monotonic() - start)
    if (args.final_minutes > 0
            and remaining >= args.final_minutes * 60.0 + 120.0):
        print(f"[{algorithm}] validação final de {args.final_minutes:g} min "
              f"(seed={final_eval_seed})", flush=True)
        final = run_trial(
            algorithm, best.params, out / "final",
            args.final_minutes, args.final_eval_episodes, final_eval_seed,
            base_config)
        final.index = len(trials)
        trials.append(final)
        with trials_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(final.__dict__, ensure_ascii=False) + "\n")
        result["final"] = {
            "status": final.status,
            "reward_mean": final.reward_mean,
            "coins_mean": final.coins_mean,
            "deaths_mean": final.deaths_mean,
            "params": final.params,
            "minutes": args.final_minutes,
            "eval_seed": final_eval_seed,
            "held_out": final_eval_seed != args.eval_seed,
        }
        print(f"[{algorithm}] final: {result['final']}", flush=True)
    _write_json(out / "best.json", result)
    return result


# ---------------------------------------------------------------------------
# CLI


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithms", default="ppo,eppo",
                        help="lista separada por vírgula")
    parser.add_argument("--base-config",
                        default="configs/seek_coin/base-level3.yaml",
                        help="YAML base (arena, nível, física, avaliação)")
    parser.add_argument("--space-dir", default="configs/search_spaces",
                        help="diretório com <alg>.yaml")
    parser.add_argument("--hours-per-alg", type=float, default=6.0,
                        help="orçamento de parede por algoritmo (padrão: 6)")
    parser.add_argument("--trial-minutes", type=float, default=10.0,
                        help="duração de cada trial de baixa fidelidade")
    parser.add_argument("--final-minutes", type=float, default=30.0,
                        help="duração da validação final do melhor trial; 0 desliga")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="episódios de avaliação por trial")
    parser.add_argument("--final-eval-episodes", type=int, default=50,
                        help="episódios de avaliação da validação final")
    parser.add_argument("--eval-seed", type=int, default=1000,
                        help="seed de avaliação usada para selecionar trials")
    parser.add_argument("--final-eval-seed", type=int, default=None,
                        help="seed de avaliação final (padrão: eval-seed + 1000); "
                             "use uma seed separada da seleção")
    parser.add_argument("--selection-metric", default="score",
                        choices=["score", "reward", "coins"],
                        help="métrica primária para escolher o melhor trial")
    parser.add_argument("--seed", type=int, default=7,
                        help="seed do BO e do treino dos trials")
    parser.add_argument("--max-trials", type=int, default=0,
                        help="limite de trials por algoritmo; 0 = só orçamento")
    parser.add_argument("--out", default="experiments/BO-generica",
                        help="diretório da campanha")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    # Os filhos treinam localmente; não passam pelo portão da fila.
    os.environ.setdefault("RL_ARENA_QUEUE_BYPASS", "1")
    args = parse_args(argv)
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
        "hours_per_alg": args.hours_per_alg,
        "trial_minutes": args.trial_minutes,
        "final_minutes": args.final_minutes,
        "eval_episodes": args.eval_episodes,
        "final_eval_episodes": args.final_eval_episodes,
        "eval_seed": args.eval_seed,
        "final_eval_seed": (args.final_eval_seed
                            if args.final_eval_seed is not None
                            else args.eval_seed + 1000),
        "selection_metric": args.selection_metric,
        "seed": args.seed,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_json(out / "campaign.json", manifest)
    results = {}
    for algorithm in algorithms:
        results[algorithm] = run_algorithm(
            algorithm, spaces[algorithm], base_config, args)
        _write_json(out / "campaign.json", {**manifest, "results": results})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
