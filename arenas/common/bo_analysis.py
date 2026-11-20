"""Leitura e análise exploratória de campanhas de BO/ASHA para a Fila.

Este módulo **não recalcula treino nem inventa trials**: lê os artefatos que
`scripts/bo_campaign.py` e `scripts/asha_campaign.py` já gravam
(`trials.jsonl`, `campaign.json`, `best.json`, `results.csv`) e devolve uma
visão normalizada para a interface web.

Layout suportado (um diretório por algoritmo):

    experiments/<campanha>/
      campaign.json          # manifesto (algorithms, space_dir, base_config)
      ppo/
        trials.jsonl         # um JSON por linha (BO ou ASHA)
        best.json
        results.csv          # ASHA
      eppo/
        ...

Todos os caminhos são validados contra a raiz do repositório; o módulo nunca
executa código dos artefatos. A estatística aqui é **exploratória**: com
poucos trials, correlações e efeitos marginais não autorizam causalidade.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import yaml

from .experiment_analysis import (
    AnalysisError,
    relative_to_root,
    resolve_under_root,
)

MAX_TRIALS = 5000
MAX_PARAM_POINTS = 400
CI_Z = 1.959963984540054


class BoAnalysisError(ValueError):
    """Campanha de BO/ASHA ausente ou inválida."""


# ---------------------------------------------------------------------------
# Leitura de arquivos


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path, warnings: list[str]) -> list[dict]:
    records: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8",
                               errors="replace").splitlines()
    except OSError as exc:
        warnings.append(f"não foi possível ler {relative_to_root(path)}: {exc}")
        return []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            warnings.append(
                f"{relative_to_root(path)}:{number}: linha JSON inválida "
                "(ignorada)")
            continue
        if not isinstance(item, dict):
            warnings.append(
                f"{relative_to_root(path)}:{number}: registro não é objeto "
                "(ignorado)")
            continue
        records.append(item)
        if len(records) >= MAX_TRIALS:
            warnings.append(
                f"{relative_to_root(path)}: lendo apenas os primeiros "
                f"{MAX_TRIALS} trials")
            break
    return records


def _read_results_csv(path: Path, warnings: list[str]) -> list[dict]:
    import csv

    records: list[dict] = []
    try:
        with path.open(encoding="utf-8", errors="replace",
                       newline="") as handle:
            for row in csv.DictReader(handle):
                params = row.get("params")
                if isinstance(params, str):
                    try:
                        row["params"] = json.loads(params)
                    except json.JSONDecodeError:
                        row["params"] = {"_raw": params}
                records.append(dict(row))
                if len(records) >= MAX_TRIALS:
                    warnings.append(
                        f"{relative_to_root(path)}: lendo apenas os primeiros "
                        f"{MAX_TRIALS} resultados")
                    break
    except OSError as exc:
        warnings.append(
            f"não foi possível ler {relative_to_root(path)}: {exc}")
    return records


def _clean(value):
    """Converte inf/nan e tipos numpy em JSON estrito (recursivo)."""
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# ---------------------------------------------------------------------------
# Descoberta


def _find_trial_files(root: Path, max_depth: int = 4) -> list[Path]:
    """Procura `trials.jsonl`/`results.csv` dentro de `root` (sem symlinks)."""
    found: list[Path] = []
    root = root.resolve()

    def walk(directory: Path, depth: int):
        try:
            children = sorted(directory.iterdir())
        except OSError:
            return
        for child in children:
            if child.is_symlink():
                continue
            if child.is_dir() and depth < max_depth:
                if child.name in {"node_modules", "__pycache__", ".git"}:
                    continue
                walk(child, depth + 1)
            elif child.name in {"trials.jsonl", "results.csv"}:
                found.append(child)
    walk(root, 0)
    return found


def _root_from_trial_file(path: Path) -> Path:
    """Diretório da campanha a partir de `.../<alg>/trials.jsonl`."""
    if path.name == "results.csv":
        return path.parent.parent
    return path.parent.parent


def _candidate_roots(path: Path) -> list[Path]:
    if path.is_file():
        if path.name in {"trials.jsonl", "results.csv"}:
            return [_root_from_trial_file(path)]
        if path.name == "campaign.json":
            return [path.parent]
        if path.name == "best.json":
            return [path.parent.parent if path.parent.parent != path.parent
                    else path.parent]
        return [path.parent]
    return [path]


def _campaign_name(root: Path) -> str:
    return root.name or str(root)


# ---------------------------------------------------------------------------
# Normalização dos trials


def _pick(record: dict, *names):
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return None


def _trial_kind(record: dict, trial_dir: str | None) -> str:
    if str(record.get("kind") or "") == "final":
        return "final"
    if trial_dir and Path(str(trial_dir)).name == "final":
        return "final"
    return "trial"


def _resolve_artifact(trial_dir: str | None, alg_dir: Path,
                      index: int, rung: int | None) -> tuple[str | None, str | None]:
    """Devolve `(diretório relativo, log relativo)` do trial."""
    candidates: list[Path] = []
    if trial_dir:
        raw = Path(str(trial_dir))
        candidates.extend([raw, alg_dir.parent / raw, alg_dir / raw])
    if rung is not None and index is not None and index >= 0:
        candidates.append(alg_dir / f"trial_{int(index):03d}_r{int(rung)}")
    chosen = None
    for candidate in candidates:
        try:
            resolved = resolve_under_root(candidate)
        except AnalysisError:
            continue
        if resolved.is_dir():
            chosen = resolved
            break
    if chosen is None:
        return None, None
    log = chosen / "train.log"
    return (relative_to_root(chosen),
            relative_to_root(log) if log.is_file() else None)


def _normalize_trial(record: dict, algorithm: str, source: str,
                     alg_dir: Path) -> dict:
    index = record.get("index")
    try:
        index = int(index)
    except (TypeError, ValueError):
        index = -1
    rung = record.get("rung")
    try:
        rung = int(rung) if rung is not None else None
    except (TypeError, ValueError):
        rung = None
    status = str(record.get("status") or "desconhecido")
    score = _finite(record.get("score"))
    reward = _finite(_pick(record, "reward_mean", "reward"))
    coins = _finite(_pick(record, "coins_mean", "coins"))
    deaths = _finite(_pick(record, "deaths_mean", "deaths"))
    seconds = _finite(record.get("seconds"))
    minutes = _finite(record.get("minutes"))
    if seconds is None and minutes is not None:
        seconds = minutes * 60.0
    trial_dir = record.get("trial_dir")
    artifact, log = _resolve_artifact(
        str(trial_dir) if trial_dir else None, alg_dir, index, rung)
    params = record.get("params")
    if not isinstance(params, dict):
        params = {}
    kind = _trial_kind(record, str(trial_dir) if trial_dir else None)
    return {
        "algorithm": algorithm,
        "source": source,
        "kind": kind,
        "index": index,
        "rung": rung,
        "status": status,
        "ok": status == "ok" and score is not None,
        "score": score,
        "reward": reward,
        "coins": coins,
        "deaths": deaths,
        "duration_seconds": seconds,
        "minutes": minutes,
        "params": params,
        "artifact": artifact,
        "log": log,
        "error": str(record.get("error") or ""),
    }


def _detect_source(records: list[dict], alg_dir: Path) -> str:
    if (alg_dir / "results.csv").is_file():
        return "asha"
    if any("rung" in record for record in records):
        return "asha"
    return "bo"


# ---------------------------------------------------------------------------
# Estatística exploratória


def _mean(values) -> float | None:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else None


def _sd(values) -> float | None:
    values = [float(value) for value in values]
    if len(values) < 2:
        return None
    media = sum(values) / len(values)
    return math.sqrt(sum((v - media) ** 2 for v in values)
                     / (len(values) - 1))


def _ci95(values) -> list[float] | None:
    values = [float(value) for value in values]
    if not values:
        return None
    media = sum(values) / len(values)
    if len(values) < 2:
        return [media, media]
    erro = _sd(values) / math.sqrt(len(values))
    return [media - CI_Z * erro, media + CI_Z * erro]


def _ranks(values) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while (end + 1 < len(order)
               and values[order[end + 1]] == values[order[position]]):
            end += 1
        rank = (position + end) / 2.0 + 1.0
        for i in range(position, end + 1):
            ranks[order[i]] = rank
        position = end + 1
    return ranks


def _pearson(x, y):
    if len(x) < 2 or len(x) != len(y):
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if denom == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denom


def _spearman(pairs) -> float | None:
    if len(pairs) < 3:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    return _pearson(_ranks(xs), _ranks(ys))


def _is_numeric(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool))


def _infer_kind(values) -> str:
    if not values:
        return "unknown"
    if all(_is_numeric(value) for value in values):
        unique = {float(value) for value in values}
        if len(unique) <= 8 and len(unique) * 2 < len(values):
            return "categorical"
        return "numeric"
    return "categorical"


def _space_kinds(manifest: dict, algorithm: str,
                 warnings: list[str]) -> dict[str, str]:
    """Tipos dos parâmetros a partir do espaço de busca YAML, se existir."""
    space_dir = manifest.get("space_dir") or "configs/search_spaces"
    path = Path(space_dir) / f"{algorithm}.yaml"
    try:
        resolved = resolve_under_root(path)
        data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except (AnalysisError, OSError, yaml.YAMLError):
        warnings.append(
            f"espaço de busca {path} não encontrado; a classificação "
            "numérico/categórico foi inferida dos trials")
        return {}
    kinds = {}
    for item in data.get("params", []) if isinstance(data, dict) else []:
        if not isinstance(item, dict) or "name" not in item:
            continue
        kind = str(item.get("type", "float"))
        kinds[str(item["name"])] = (
            "categorical" if kind == "choice" else "numeric")
    return kinds


def _bins(pairs, max_bins: int = 4) -> list[dict]:
    """Faixas por ordem de valor, com n, média e IC de 95%."""
    if not pairs:
        return []
    ordered = sorted(pairs, key=lambda pair: pair[0])
    groups = [group for group in np.array_split(
        np.arange(len(ordered)), min(max_bins, len(ordered))) if len(group)]
    bins = []
    for group in groups:
        indices = [int(i) for i in group]
        xs = [ordered[i][0] for i in indices]
        scores = [ordered[i][1] for i in indices]
        bins.append({
            "lo": float(min(xs)),
            "hi": float(max(xs)),
            "n": len(scores),
            "mean": _mean(scores),
            "ci95": _ci95(scores),
        })
    return bins


def _numeric_param(name: str, pairs: list[tuple[float, float]],
                   importance: float | None) -> dict:
    points = [{"x": float(x), "score": float(y)}
              for x, y in pairs][:MAX_PARAM_POINTS]
    return {
        "name": name,
        "kind": "numeric",
        "n": len(pairs),
        "points": points,
        "bins": _bins(pairs),
        "spearman": _spearman(pairs),
        "importance": importance,
    }


def _categorical_param(name: str, groups: dict) -> dict:
    rows = []
    for key in sorted(groups, key=str):
        scores = groups[key]
        rows.append({
            "value": key,
            "n": len(scores),
            "mean": _mean(scores),
            "sd": _sd(scores),
            "ci95": _ci95(scores),
            "scores": [float(v) for v in scores][:MAX_PARAM_POINTS],
        })
    return {
        "name": name,
        "kind": "categorical",
        "n": sum(row["n"] for row in rows),
        "groups": rows,
        "importance": None,
    }


def _eta_squared(groups: dict) -> float | None:
    all_scores = [value for values in groups.values() for value in values]
    if len(all_scores) < 2:
        return None
    overall = sum(all_scores) / len(all_scores)
    total = sum((value - overall) ** 2 for value in all_scores)
    if total == 0:
        return None
    between = sum(len(values) * (sum(values) / len(values) - overall) ** 2
                  for values in groups.values())
    return between / total


def _parameter_stats(trials: list[dict], kinds: dict[str, str],
                     warnings: list[str]) -> list[dict]:
    ok = [trial for trial in trials
          if trial["ok"] and trial["score"] is not None]
    names = sorted({name for trial in ok for name in trial["params"]})
    stats = []
    for name in names:
        values = [(trial["params"][name], trial["score"]) for trial in ok
                  if name in trial["params"]]
        if not values:
            continue
        kind = kinds.get(name) or _infer_kind([v for v, _ in values])
        if kind == "numeric":
            pairs = [(float(v), float(s)) for v, s in values
                     if _is_numeric(v)]
            if len(pairs) < 2:
                continue
            rho = _spearman(pairs)
            stats.append(_numeric_param(name, pairs,
                                        abs(rho) if rho is not None else None))
        else:
            groups: dict = {}
            for value, score in values:
                groups.setdefault(str(value), []).append(float(score))
            entry = _categorical_param(name, groups)
            entry["eta_squared"] = _eta_squared(groups)
            entry["importance"] = entry["eta_squared"]
            stats.append(entry)
    stats.sort(key=lambda item: (
        -(item.get("importance") or 0.0), item["name"]))
    for position, item in enumerate(stats, start=1):
        item["rank"] = position
    if not stats:
        warnings.append("nenhum parâmetro com trials válidos suficiente "
                        "para estimar efeitos")
    return stats


def _best_so_far(trials: list[dict]) -> list[dict]:
    ordered = sorted(trials, key=lambda trial: trial["index"])
    best = None
    series = []
    for trial in ordered:
        if trial["ok"] and trial["score"] is not None:
            best = trial["score"] if best is None else max(best, trial["score"])
        series.append({
            "index": trial["index"],
            "score": trial["score"],
            "best": best,
            "status": trial["status"],
            "kind": trial["kind"],
            "rung": trial["rung"],
        })
    return series


def _interaction(trials: list[dict], stats: list[dict]) -> dict | None:
    numeric = [item for item in stats if item["kind"] == "numeric"]
    if len(numeric) < 2:
        return None
    first, second = numeric[0]["name"], numeric[1]["name"]
    points = [(trial["params"].get(first), trial["params"].get(second),
               trial["score"]) for trial in trials
              if trial["ok"] and trial["score"] is not None
              and _is_numeric(trial["params"].get(first))
              and _is_numeric(trial["params"].get(second))]
    if len(points) < 4:
        return None
    if len({float(p[0]) for p in points}) < 3:
        return None
    if len({float(p[1]) for p in points}) < 3:
        return None

    def terciles(values):
        ordered = sorted(values)
        return [ordered[int(len(ordered) * k / 3)]
                for k in (0, 1, 2)] if ordered else []

    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x_edges = terciles(xs)
    y_edges = terciles(ys)

    def bucket(value, edges):
        if value <= edges[0]:
            return 0
        if value <= edges[1]:
            return 1
        return 2

    means = [[None] * 3 for _ in range(3)]
    counts = [[0] * 3 for _ in range(3)]
    sums = [[0.0] * 3 for _ in range(3)]
    for x, y, score in points:
        i, j = bucket(float(x), y_edges), bucket(float(y), x_edges)
        sums[j][i] += float(score)
        counts[j][i] += 1
    for j in range(3):
        for i in range(3):
            if counts[j][i]:
                means[j][i] = sums[j][i] / counts[j][i]
    return {
        "x_param": first,
        "y_param": second,
        "x_labels": [f"{x_edges[0]:g}–{x_edges[1]:g}",
                     f"{x_edges[1]:g}–{x_edges[2]:g}",
                     f"≥ {x_edges[2]:g}"],
        "y_labels": [f"{y_edges[0]:g}–{y_edges[1]:g}",
                     f"{y_edges[1]:g}–{y_edges[2]:g}",
                     f"≥ {y_edges[2]:g}"],
        "means": means,
        "counts": counts,
    }


def _budget_bands(trials: list[dict], source: str) -> list[dict]:
    ok = [trial for trial in trials
          if trial["ok"] and trial["score"] is not None]
    if not ok:
        return []
    bands = []
    if source == "asha":
        rungs = sorted({trial["rung"] for trial in ok
                        if trial["rung"] is not None})
        for rung in rungs:
            scores = [trial["score"] for trial in ok if trial["rung"] == rung]
            bands.append({
                "label": f"rung {rung}",
                "minutes": rung,
                "n": len(scores),
                "mean": _mean(scores),
                "best": max(scores) if scores else None,
            })
        return bands
    for kind, label in (("trial", "trials"), ("final", "validação final")):
        scores = [trial["score"] for trial in ok if trial["kind"] == kind]
        if scores:
            bands.append({
                "label": label,
                "n": len(scores),
                "mean": _mean(scores),
                "best": max(scores),
            })
    return bands


def _baseline_params(manifest: dict, warnings: list[str]) -> dict:
    base = manifest.get("base_config")
    if not isinstance(base, str):
        return {}
    try:
        resolved = resolve_under_root(base)
        data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except (AnalysisError, OSError, yaml.YAMLError):
        warnings.append(
            f"configuração base {base} não encontrada; comparação com o "
            "default fica indisponível")
        return {}
    if not isinstance(data, dict):
        return {}
    params = ((data.get("training") or {}).get("agent_params")
              or data.get("agent_params") or {})
    return params if isinstance(params, dict) else {}


# ---------------------------------------------------------------------------
# Campanha


def _load_algorithm(alg_dir: Path, algorithm: str,
                    manifest: dict) -> dict:
    warnings: list[str] = []
    trials_path = alg_dir / "trials.jsonl"
    records = _read_jsonl(trials_path, warnings)
    source = _detect_source(records, alg_dir)
    if not records:
        results_path = alg_dir / "results.csv"
        if results_path.is_file():
            records = _read_results_csv(results_path, warnings)
    best = _load_json(alg_dir / "best.json") or {}
    if not isinstance(best, dict):
        best = {}
    trials = [_normalize_trial(record, algorithm, source, alg_dir)
              for record in records]
    if not trials:
        warnings.append(
            f"{relative_to_root(alg_dir)}: nenhum trial encontrado "
            "(trials.jsonl e results.csv ausentes ou vazios)")
    kinds = _space_kinds(manifest, algorithm, warnings)
    selection = [trial for trial in trials if trial["kind"] == "trial"
                 and trial["ok"] and trial["score"] is not None]
    # A validação final roda o melhor trial com outro orçamento; ela entra na
    # convergência e na faixa de orçamento, mas não nas estatísticas de
    # parâmetro (seria uma duplicata do melhor com score mais alto).
    params = _parameter_stats(selection, kinds, warnings)
    series = _best_so_far(trials)
    ok_scores = [trial["score"] for trial in trials
                 if trial["ok"] and trial["score"] is not None]
    selection_scores = [trial["score"] for trial in selection]
    best_trial = best.get("best_trial")
    if best_trial is None:
        if selection:
            best_trial = max(selection,
                             key=lambda trial: trial["score"])["index"]
    for trial in trials:
        trial["is_best"] = (trial["index"] == best_trial
                            and trial["kind"] == "trial")
    best_score = _finite(best.get("best_score"))
    if best_score is None:
        best_score = _finite((best.get("best_low_fidelity") or {}).get("score"))
    if best_score is None and selection_scores:
        best_score = max(selection_scores)
    median = (float(np.median(selection_scores))
              if selection_scores else None)
    first = selection_scores[0] if selection_scores else None

    failed = [trial for trial in trials if not trial["ok"]]
    if failed:
        warnings.append(
            f"{len(failed)} trial(s) falho(s) listados na tabela e "
            "excluídos das estatísticas")
    if len(selection_scores) < 10:
        warnings.append(
            f"apenas {len(selection_scores)} trial(s) válido(s): os efeitos "
            "abaixo são exploratórios; não declare causalidade")
    if not best:
        warnings.append("best.json ausente; o melhor trial foi inferido "
                        "pelo maior score")

    baseline = _baseline_params(manifest, warnings)
    best_params = (best.get("best_params")
                   if isinstance(best.get("best_params"), dict)
                   else next((trial["params"] for trial in trials
                              if trial["is_best"]), {}))
    changed = {name: {"default": baseline.get(name),
                      "best": best_params.get(name)}
               for name in sorted(set(baseline) | set(best_params))
               if baseline.get(name) != best_params.get(name)}
    final = best.get("final") if isinstance(best.get("final"), dict) else None
    improvement = {
        "best_score": best_score,
        "median_score": median,
        "first_score": first,
        "delta_vs_median": ((best_score - median)
                            if best_score is not None and median is not None
                            else None),
        "delta_vs_first": ((best_score - first)
                           if best_score is not None and first is not None
                           else None),
        "baseline_params": baseline,
        "best_params": best_params,
        "changed_params": changed,
        "final": final,
    }

    # Outliers de score (não escondidos: apenas contados).
    outliers = 0
    if len(ok_scores) >= 4:
        arr = np.asarray(ok_scores, dtype=float)
        q1, q3 = np.quantile(arr, [0.25, 0.75])
        iqr = q3 - q1
        if iqr > 0:
            outliers = int(np.sum((arr < q1 - 1.5 * iqr)
                                  | (arr > q3 + 1.5 * iqr)))
    if outliers:
        warnings.append(
            f"{outliers} score(s) fora de 1,5×IQR; eles continuam nos "
            "gráficos e na tabela")

    return _clean({
        "algorithm": algorithm,
        "source": source,
        "directory": relative_to_root(alg_dir),
        "trials": trials,
        "best_so_far": series,
        "best": best,
        "best_score": best_score,
        "n_trials": len(trials),
        "n_ok": len(ok_scores),
        "n_failed": len(failed),
        "params": params,
        "interaction": _interaction(selection, params),
        "improvement": improvement,
        "budget": _budget_bands(trials, source),
        "warnings": warnings,
    })


def _discover_algorithm_dirs(root: Path) -> list[tuple[str, Path]]:
    files = _find_trial_files(root)
    directories: dict[Path, None] = {}
    for file in files:
        directories[file.parent] = None
    return sorted(((directory.name, directory)
                   for directory in directories), key=lambda item: item[0])


def load_bo(path: str | Path) -> dict:
    """Carrega uma campanha BO/ASHA de `path` (arquivo ou diretório)."""
    try:
        resolved = resolve_under_root(path)
    except AnalysisError as exc:
        raise BoAnalysisError(str(exc)) from exc
    if not resolved.exists():
        raise BoAnalysisError(
            f"campanha não encontrada: {relative_to_root(resolved)}")
    warnings: list[str] = []
    roots = _candidate_roots(resolved)
    campaigns = []
    manifest = {}
    root = None
    for candidate in roots:
        if not candidate.is_dir():
            continue
        candidate_manifest = _load_json(candidate / "campaign.json")
        algorithm_dirs = _discover_algorithm_dirs(candidate)
        if not algorithm_dirs:
            continue
        root = candidate
        if isinstance(candidate_manifest, dict):
            manifest = candidate_manifest
            break
    if root is None:
        raise BoAnalysisError(
            "nenhum trials.jsonl/results.csv encontrado em "
            f"{relative_to_root(resolved)}")
    for algorithm, directory in _discover_algorithm_dirs(root):
        campaigns.append(_load_algorithm(directory, algorithm, manifest))
    if not campaigns:
        raise BoAnalysisError(
            f"nenhuma campanha carregável em {relative_to_root(root)}")
    campaigns.sort(key=lambda campaign: campaign["algorithm"].lower())
    if manifest.get("algorithms"):
        listed = set(str(name) for name in manifest["algorithms"])
        loaded = {campaign["algorithm"] for campaign in campaigns}
        missing = sorted(listed - loaded)
        if missing:
            warnings.append(
                "algoritmos no manifesto sem trials: " + ", ".join(missing))
    return _clean({
        "root": relative_to_root(root),
        "name": _campaign_name(root),
        "manifest": manifest,
        "campaigns": campaigns,
        "warnings": warnings,
    })


def load_bo_for_job(job: dict, experiment_data: dict | None = None) -> dict:
    """Descobre a campanha de um job da fila.

    Tenta, nesta ordem: artefatos do job, artefatos do YAML do experimento,
    `--out`/`--experiment-dir` no comando e o diretório de trabalho.
    """
    candidates: list[str] = []
    for item in job.get("artifacts") or []:
        if isinstance(item, str):
            candidates.append(item)
    artifacts = (experiment_data or {}).get("artifacts") or {}
    if isinstance(artifacts, dict):
        for value in artifacts.values():
            if isinstance(value, str):
                candidates.append(value)
    command = [str(part) for part in job.get("command") or []]
    for flag in ("--out", "--experiment-dir", "--output"):
        if flag in command:
            position = command.index(flag)
            if position + 1 < len(command):
                candidates.append(command[position + 1])
    if job.get("cwd"):
        candidates.append(str(job["cwd"]))
    errors = []
    for candidate in candidates:
        try:
            return load_bo(candidate)
        except BoAnalysisError as exc:
            errors.append(str(exc))
    detail = errors[-1] if errors else "sem artefatos de campanha"
    raise BoAnalysisError(
        f"nenhuma campanha de BO/ASHA encontrada para o job "
        f"{job.get('id')}: {detail}")
