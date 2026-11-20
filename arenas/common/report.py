"""Relatório estatístico de experimentos.

Lê resultados por seed/execução de um arquivo CSV/JSONL (ou de um diretório)
e produz:

- estatísticas descritivas por algoritmo (média, mediana, desvio, IQR e IC);
- comparações par a par com teste de hipótese e tamanho de efeito;
- correção para múltiplas comparações (Holm-Bonferroni ou Benjamini-Hochberg);
- um relatório Markdown, JSON ou HTML interativo e reproduzível.

Formato canônico de entrada (CSV):

    algorithm,seed,reward_mean,coins_mean
    dqn,7,12.3,10.1
    dqn,42,11.8,9.7
    ppo,7,10.1,8.2

Também aceita JSONL com campos aninhados. Exemplo:

    {"tag": "dqn", "episode": 7, "metrics": {"overall": {"reward_mean": 12.3}}}

Uso:

    python -m arenas.common.report --input results.csv --metric reward_mean
    python -m arenas.common.report --input experiments/EXP-001 \\
        --value-field metrics.overall.reward_mean --algorithm-field tag \\
        --seed-field episode --paired --baseline dqn
    python -m arenas.common.report --input results.csv --metric reward_mean \
        --seed-field train_seed --aggregate --paired --baseline dqn

Com `--aggregate`, várias linhas da mesma seed/unidade viram uma média antes
da comparação; use isso quando o CSV tiver vários episódios por seed de
treino.

O relatório é escrito em `report.md` por padrão. Use `--out report.html`
para gerar a interface interativa (gráficos, filtros, tabelas e download).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np

from .report_html import render_html as _render_html

# ---------------------------------------------------------------------------
# Estruturas


@dataclass
class Observation:
    """Uma medida de um algoritmo (idealmente uma por seed/execução)."""

    algorithm: str
    value: float
    seed: str | None = None
    setup: str | None = None
    scenario: str | None = None
    level: str | None = None
    source: str = ""


@dataclass
class GroupSummary:
    algorithm: str
    n: int
    mean: float
    median: float
    std: float
    q1: float
    q3: float
    iqr: float
    ci_low: float
    ci_high: float


@dataclass
class Comparison:
    a: str
    b: str
    n: int
    n_a: int
    n_b: int
    diff: float
    diff_ci_low: float
    diff_ci_high: float
    statistic: float
    p_value: float
    p_adjusted: float
    effect: float
    effect_name: str
    method: str
    paired: bool
    significant: bool


@dataclass
class Report:
    metric: str
    input_label: str
    paired: bool
    correction: str
    alpha: float
    confidence: float
    summaries: list[GroupSummary] = field(default_factory=list)
    comparisons: list[Comparison] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    setups: list[str] = field(default_factory=list)
    scenarios: list[str] = field(default_factory=list)
    levels: list[str] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Estatística descritiva


def _as_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def bootstrap_ci(values, *, statistic=np.mean, n_boot=10000,
                 confidence=0.95, rng=None) -> tuple[float, float]:
    """IC percentil por bootstrap para uma estatística (padrão: média)."""
    data = _as_array(values)
    if data.size == 0:
        return (float("nan"), float("nan"))
    if data.size == 1:
        value = float(statistic(data))
        return (value, value)
    generator = np.random.default_rng(rng)
    indices = generator.integers(0, data.size, size=(int(n_boot), data.size))
    samples = statistic(data[indices], axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    low, high = np.percentile(samples, [100 * alpha, 100 * (1 - alpha)])
    return (float(low), float(high))


def bootstrap_diff_ci(a, b, *, paired=False, n_boot=10000,
                      confidence=0.95, rng=None) -> tuple[float, float]:
    """IC bootstrap para `mean(a) - mean(b)` (pareado ou independente)."""
    a = _as_array(a)
    b = _as_array(b)
    if paired:
        if a.shape != b.shape:
            raise ValueError("amostras pareadas precisam ter o mesmo tamanho")
        return bootstrap_ci(a - b, n_boot=n_boot, confidence=confidence,
                            rng=rng)
    generator = np.random.default_rng(rng)
    means_a = a[generator.integers(0, a.size, size=(int(n_boot), a.size))].mean(axis=1)
    means_b = b[generator.integers(0, b.size, size=(int(n_boot), b.size))].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    low, high = np.percentile(means_a - means_b,
                              [100 * alpha, 100 * (1 - alpha)])
    return (float(low), float(high))


def summarize(values, *, confidence=0.95, n_boot=10000,
              rng=None) -> tuple[float, ...]:
    """Média, mediana, desvio, quartis, IQR e IC da média."""
    data = _as_array(values)
    if data.size == 0:
        nan = float("nan")
        return (nan, nan, nan, nan, nan, nan, nan, nan)
    mean = float(np.mean(data))
    median = float(np.median(data))
    std = float(np.std(data, ddof=1)) if data.size > 1 else 0.0
    q1, q3 = np.percentile(data, [25, 75])
    ci_low, ci_high = bootstrap_ci(data, n_boot=n_boot,
                                   confidence=confidence, rng=rng)
    return (mean, median, std, float(q1), float(q3), float(q3 - q1),
            ci_low, ci_high)


# ---------------------------------------------------------------------------
# Testes e tamanhos de efeito


def _rankdata(values) -> np.ndarray:
    """Ranks 1..n com média em empates."""
    data = np.asarray(values, dtype=float)
    order = np.argsort(data, kind="mergesort")
    ranks = np.empty(data.size, dtype=float)
    sorted_data = data[order]
    i = 0
    while i < data.size:
        j = i
        while j + 1 < data.size and sorted_data[j + 1] == sorted_data[i]:
            j += 1
        rank = (i + j + 2) / 2.0  # média dos ranks 1-based
        ranks[order[i:j + 1]] = rank
        i = j + 1
    return ranks


def _normal_sf(z: float) -> float:
    return 0.5 * math.erfc(float(z) / math.sqrt(2.0))


def wilcoxon_signed_rank(a, b) -> tuple[float, float]:
    """Wilcoxon signed-rank pareado. Retorna (estatística, p-valor bicaudal).

    Usa distribuição exata por programação dinâmica para n <= 20 (inclusive
    com empates) e aproximação normal com correção de empates para n > 20.
    """
    a = _as_array(a)
    b = _as_array(b)
    if a.shape != b.shape:
        raise ValueError("Wilcoxon pareado exige amostras do mesmo tamanho")
    diff = a - b
    diff = diff[diff != 0]
    n = diff.size
    if n == 0:
        return (0.0, 1.0)

    ranks = _rankdata(np.abs(diff))
    w_plus = float(ranks[diff > 0].sum())
    w_minus = float(ranks[diff < 0].sum())
    statistic = min(w_plus, w_minus)

    if n <= 20:
        # Ranks*2 viram inteiros; DP sobre a soma de W+ assume sinais 1/2.
        scaled = np.rint(ranks * 2).astype(int).tolist()
        observed = int(round(w_plus * 2))
        dp: dict[int, float] = {0: 1.0}
        for rank in scaled:
            new: dict[int, float] = defaultdict(float)
            for total, probability in dp.items():
                new[total] += probability * 0.5
                new[total + rank] += probability * 0.5
            dp = dict(new)
        p_le = sum(p for total, p in dp.items() if total <= observed)
        p_ge = sum(p for total, p in dp.items() if total >= observed)
        p_value = 2.0 * min(p_le, p_ge)
    else:
        mean = n * (n + 1) / 4.0
        variance = float(np.sum(ranks ** 2)) / 4.0
        if variance <= 0:
            p_value = 1.0
        else:
            z = (w_plus - mean) / math.sqrt(variance)
            p_value = 2.0 * _normal_sf(abs(z))
    return (float(statistic), float(min(1.0, p_value)))


def mann_whitney_u(a, b) -> tuple[float, float]:
    """Mann-Whitney U independente. Retorna (U, p-valor bicaudal).

    Aproximação normal com correção de continuidade e de empates. Para
    amostras pequenas, o p-valor é aproximado; reporte isso no relatório.
    """
    a = _as_array(a)
    b = _as_array(b)
    n1, n2 = a.size, b.size
    if n1 == 0 or n2 == 0:
        return (float("nan"), 1.0)
    combined = np.concatenate([a, b])
    ranks = _rankdata(combined)
    r1 = float(ranks[:n1].sum())
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    total = n1 + n2
    _, counts = np.unique(combined, return_counts=True)
    tie_sum = float(np.sum(counts ** 3 - counts))
    variance = (n1 * n2 / 12.0) * (
        (total + 1.0) - tie_sum / (total * (total - 1.0)))
    if variance <= 0:
        return (float(u), 1.0)
    mean = n1 * n2 / 2.0
    # U é min(u1, u2), então fica abaixo da média; a correção aproxima de mean.
    z = (u - mean + 0.5) / math.sqrt(variance)
    p_value = 2.0 * _normal_sf(abs(z))
    return (float(u), float(min(1.0, p_value)))


def cliffs_delta(a, b) -> float:
    """Tamanho de efeito não paramétrico: P(a > b) - P(a < b)."""
    a = _as_array(a)
    b = _as_array(b)
    if a.size == 0 or b.size == 0:
        return float("nan")
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def cohens_d(a, b, *, paired=False) -> float:
    """Tamanho de efeito padronizado (pareado ou independente)."""
    a = _as_array(a)
    b = _as_array(b)
    if paired:
        if a.shape != b.shape:
            raise ValueError("Cohen's d pareado exige amostras do mesmo tamanho")
        diff = a - b
        sd = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
        mean = float(np.mean(diff))
    else:
        if a.size < 2 or b.size < 2:
            return float("nan")
        pooled = math.sqrt(
            ((a.size - 1) * np.var(a, ddof=1)
             + (b.size - 1) * np.var(b, ddof=1))
            / (a.size + b.size - 2))
        sd = float(pooled)
        mean = float(np.mean(a) - np.mean(b))
    if sd == 0:
        return 0.0 if mean == 0 else math.copysign(float("inf"), mean)
    return float(mean / sd)


# ---------------------------------------------------------------------------
# Múltiplas comparações


def holm_bonferroni(p_values) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p, kind="mergesort")
    adjusted = np.empty(m, dtype=float)
    previous = 0.0
    for rank, index in enumerate(order):
        value = max(previous, (m - rank) * p[index])
        adjusted[index] = min(1.0, value)
        previous = adjusted[index]
    return adjusted


def benjamini_hochberg(p_values) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p, kind="mergesort")
    adjusted = np.empty(m, dtype=float)
    previous = 1.0
    for rank in range(m - 1, -1, -1):
        index = order[rank]
        value = min(previous, p[index] * m / (rank + 1))
        adjusted[index] = min(1.0, value)
        previous = adjusted[index]
    return adjusted


def adjust_p_values(p_values, method: str) -> np.ndarray:
    method = method.lower()
    if method in ("none", "nenhuma"):
        return np.asarray(list(p_values), dtype=float)
    if method in ("holm", "holm-bonferroni", "bonferroni"):
        return holm_bonferroni(p_values)
    if method in ("bh", "benjamini-hochberg", "fdr"):
        return benjamini_hochberg(p_values)
    raise ValueError(f"correção desconhecida: {method!r}")


# ---------------------------------------------------------------------------
# Leitura de resultados


def _get_path(record, field_name: str):
    if field_name in record:
        return record[field_name]
    current = record
    for part in field_name.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _pick(record, field_name: str | None, candidates: Iterable[str]):
    names = [field_name] if field_name else []
    names.extend(candidates)
    for name in names:
        if not name:
            continue
        value = _get_path(record, name)
        if value is not None:
            return value
    return None


def _iter_records(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            yield from csv.DictReader(handle)
    elif suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            yield from payload
        elif isinstance(payload, dict):
            for key in ("records", "runs", "results", "data"):
                if isinstance(payload.get(key), list):
                    yield from payload[key]
                    break
            else:
                yield payload
        else:
            raise ValueError(f"JSON sem lista/dicionário: {path}")


def _iter_files(path: Path):
    if path.is_file():
        yield path
        return
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in (".csv", ".jsonl", ".json"):
            continue
        if any(part.startswith(".") or part == "__pycache__"
               for part in candidate.parts):
            continue
        yield candidate


def _infer_algorithm_from_path(path: Path) -> str | None:
    generic = {"logs", "runs", "checkpoints", "videos", "mixed", "t15"}
    for parent in path.parents:
        name = parent.name
        if not name or name in (".", ".."):
            continue
        if name.lower() in generic:
            continue
        if name.startswith("EXP-"):
            return name
        if name not in ("experiments",):
            return name
    return path.stem or None


def load_observations(
    input_path: str | Path,
    *,
    metric: str = "reward_mean",
    value_field: str | None = None,
    algorithm_field: str | None = None,
    seed_field: str | None = None,
    setup_field: str | None = None,
    scenario_field: str | None = None,
    level_field: str | None = None,
    filters: dict[str, str] | None = None,
) -> tuple[list[Observation], list[str]]:
    """Lê observações de um arquivo ou diretório.

    Devolve `(observações, avisos)`. Registros sem algoritmo ou sem valor
    numérico são ignorados com aviso.
    """
    path = Path(input_path)
    if not path.exists():
        raise SystemExit(f"entrada não encontrada: {path}")
    filters = filters or {}
    algorithm_candidates = ("algorithm", "alg", "policy", "tag", "run",
                            "run_name")
    seed_candidates = ("seed", "episode", "run_id", "trial")
    setup_candidates = ("setup", "config", "configuration")
    scenario_candidates = ("scenario", "cenario")
    level_candidates = ("level", "nível", "nivel")
    value_candidates = (
        "value", "metric", metric, f"metrics.{metric}",
        f"metrics.overall.{metric}", "reward_mean", "metrics.reward_mean",
        "metrics.overall.reward_mean", "reward",
    )

    observations: list[Observation] = []
    warnings: list[str] = []
    for file_path in _iter_files(path):
        try:
            records = list(_iter_records(file_path))
        except Exception as exc:  # noqa: BLE001 - aviso amigável por arquivo
            warnings.append(f"{file_path}: ignorado ({exc})")
            continue
        for row_number, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                warnings.append(
                    f"{file_path}:{row_number}: registro não é objeto; ignorado")
                continue
            if filters:
                mismatch = False
                for key, expected in filters.items():
                    actual = _get_path(record, key)
                    if actual is None or str(actual) != str(expected):
                        mismatch = True
                        break
                if mismatch:
                    continue
            algorithm = _pick(record, algorithm_field, algorithm_candidates)
            if algorithm is None:
                algorithm = _infer_algorithm_from_path(file_path)
            if algorithm is None or not str(algorithm).strip():
                warnings.append(
                    f"{file_path}:{row_number}: sem algoritmo; ignorado")
                continue
            raw_value = _pick(record, value_field, value_candidates)
            if raw_value is None:
                warnings.append(
                    f"{file_path}:{row_number}: sem valor para {metric!r}; ignorado")
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                warnings.append(
                    f"{file_path}:{row_number}: valor não numérico "
                    f"({raw_value!r}); ignorado")
                continue
            if not math.isfinite(value):
                warnings.append(
                    f"{file_path}:{row_number}: valor não finito; ignorado")
                continue
            seed = _pick(record, seed_field, seed_candidates)
            setup = _pick(record, setup_field, setup_candidates)
            scenario = _pick(record, scenario_field, scenario_candidates)
            level = _pick(record, level_field, level_candidates)
            observations.append(Observation(
                algorithm=str(algorithm),
                value=value,
                seed=None if seed is None else str(seed),
                setup=None if setup is None else str(setup),
                scenario=None if scenario is None else str(scenario),
                level=None if level is None else str(level),
                source=f"{file_path}:{row_number}",
            ))
    if not observations:
        raise SystemExit(
            f"nenhuma observação numérica encontrada em {path}; "
            "confira --value-field/--algorithm-field")
    return observations, warnings


def aggregate_by_unit(observations: list[Observation], *,
                      warnings: list[str] | None = None) -> list[Observation]:
    """Agrega observações repetidas da mesma unidade independente.

    A unidade é `(algoritmo, seed, setup, cenário, nível)`. Vários episódios
    de avaliação da mesma seed de treino viram uma única média, evitando
    tratá-los como runs independentes. Observações sem seed passam intactas.
    """
    warnings = warnings if warnings is not None else []
    grouped: dict[tuple, list[Observation]] = {}
    passthrough: list[Observation] = []
    for observation in observations:
        if observation.seed is None:
            passthrough.append(observation)
            continue
        key = (observation.algorithm, observation.seed, observation.setup,
               observation.scenario, observation.level)
        grouped.setdefault(key, []).append(observation)
    if not grouped:
        return list(observations)

    aggregated: list[Observation] = []
    for (algorithm, seed, setup, scenario, level), items in grouped.items():
        if len(items) > 1:
            warnings.append(
                f"{algorithm}: unidade {seed!r} tinha {len(items)} linhas; "
                "agregada pela média")
        aggregated.append(Observation(
            algorithm=algorithm,
            value=float(np.mean([item.value for item in items])),
            seed=seed,
            setup=setup,
            scenario=scenario,
            level=level,
            source="; ".join(item.source for item in items if item.source),
        ))
    aggregated.extend(passthrough)
    return aggregated


# ---------------------------------------------------------------------------
# Relatório


def _paired_arrays(obs_a: list[Observation], obs_b: list[Observation],
                   warnings: list[str], label_a: str, label_b: str):
    def keyed(observations, label):
        buckets: dict[str, list[float]] = defaultdict(list)
        for observation in observations:
            if observation.seed is None:
                raise ValueError(
                    "--paired exige seed em todas as observações; "
                    f"{label} tem registros sem seed")
            buckets[observation.seed].append(observation.value)
        result = {}
        for seed, values in buckets.items():
            if len(values) > 1:
                warnings.append(
                    f"{label}: seed {seed!r} repetida; usando a média "
                    f"de {len(values)} valores")
            result[seed] = float(np.mean(values))
        return result

    data_a = keyed(obs_a, label_a)
    data_b = keyed(obs_b, label_b)
    common = sorted(set(data_a) & set(data_b), key=str)
    if not common:
        raise ValueError(
            f"não há seeds em comum entre {label_a!r} e {label_b!r}")
    if len(common) < 2:
        warnings.append(
            f"{label_a} × {label_b}: apenas {len(common)} seed em comum; "
            "teste pareado tem baixo poder")
    a = np.array([data_a[seed] for seed in common], dtype=float)
    b = np.array([data_b[seed] for seed in common], dtype=float)
    return a, b, common


def build_report(
    observations: list[Observation],
    *,
    metric: str,
    input_label: str = "",
    paired: bool = False,
    baseline: str | None = None,
    correction: str = "holm",
    alpha: float = 0.05,
    confidence: float = 0.95,
    n_boot: int = 10000,
    rng: int | None = 7,
) -> Report:
    """Monta o relatório a partir das observações."""
    by_algorithm: dict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        by_algorithm[observation.algorithm].append(observation)
    algorithms = sorted(by_algorithm, key=str.lower)

    report = Report(
        metric=metric,
        input_label=input_label,
        paired=paired,
        correction=correction,
        alpha=alpha,
        confidence=confidence,
        setups=sorted({o.setup for o in observations if o.setup}),
        scenarios=sorted({o.scenario for o in observations if o.scenario}),
        levels=sorted({o.level for o in observations if o.level}),
        observations=list(observations),
    )
    if len(algorithms) < 2:
        report.warnings.append(
            "há menos de dois algoritmos; o relatório traz só descritivas")

    if not paired:
        for algorithm in algorithms:
            seeds = [o.seed for o in by_algorithm[algorithm]
                     if o.seed is not None]
            if len(seeds) > len(set(seeds)):
                report.warnings.append(
                    f"{algorithm}: seed repetida em análise independente; "
                    "agregue por unidade com --aggregate para evitar "
                    "pseudo-replicação")

    for algorithm in algorithms:
        values = [o.value for o in by_algorithm[algorithm]]
        mean, median, std, q1, q3, iqr, ci_low, ci_high = summarize(
            values, confidence=confidence, n_boot=n_boot, rng=rng)
        report.summaries.append(GroupSummary(
            algorithm=algorithm, n=len(values), mean=mean, median=median,
            std=std, q1=q1, q3=q3, iqr=iqr, ci_low=ci_low, ci_high=ci_high,
        ))
        if len(values) < 3:
            report.warnings.append(
                f"{algorithm}: apenas {len(values)} observações; "
                "use 3+ seeds para uma comparação mais confiável")

    if baseline is not None:
        if baseline not in by_algorithm:
            raise SystemExit(f"baseline {baseline!r} não encontrado")
        pairs = [(baseline, other) for other in algorithms
                 if other != baseline]
    else:
        pairs = list(combinations(algorithms, 2))

    raw_comparisons: list[Comparison] = []
    for label_a, label_b in pairs:
        obs_a, obs_b = by_algorithm[label_a], by_algorithm[label_b]
        if paired:
            values_a, values_b, seeds = _paired_arrays(
                obs_a, obs_b, report.warnings, label_a, label_b)
            n = len(seeds)
            method = "Wilcoxon signed-rank"
            statistic, p_value = wilcoxon_signed_rank(values_a, values_b)
            effect = cohens_d(values_a, values_b, paired=True)
            effect_name = "Cohen's d (pareado)"
            if not math.isfinite(effect):
                # Diferenças constantes deixam d infinito; Cliff's delta
                # continua finito e informa a direção/consistência.
                effect = cliffs_delta(values_a, values_b)
                effect_name = "Cliff's delta (d infinito)"
        else:
            values_a = np.array([o.value for o in obs_a], dtype=float)
            values_b = np.array([o.value for o in obs_b], dtype=float)
            n = values_a.size + values_b.size
            method = "Mann-Whitney U"
            statistic, p_value = mann_whitney_u(values_a, values_b)
            effect = cliffs_delta(values_a, values_b)
            effect_name = "Cliff's delta"
        diff = float(np.mean(values_a) - np.mean(values_b))
        diff_low, diff_high = bootstrap_diff_ci(
            values_a, values_b, paired=paired, n_boot=n_boot,
            confidence=confidence, rng=rng)
        raw_comparisons.append(Comparison(
            a=label_a, b=label_b, n=n,
            n_a=len(obs_a), n_b=len(obs_b),
            diff=diff, diff_ci_low=diff_low, diff_ci_high=diff_high,
            statistic=statistic, p_value=p_value, p_adjusted=p_value,
            effect=effect, effect_name=effect_name, method=method,
            paired=paired, significant=False,
        ))

    adjusted = adjust_p_values([c.p_value for c in raw_comparisons],
                               correction)
    for comparison, p_adjusted in zip(raw_comparisons, adjusted):
        comparison.p_adjusted = float(p_adjusted)
        comparison.significant = bool(p_adjusted < alpha)
        report.comparisons.append(comparison)
    return report


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/d"
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{digits}f}"


def render_markdown(report: Report) -> str:
    """Renderiza o relatório em Markdown."""
    lines: list[str] = []
    lines.append(f"# Relatório estatístico — {report.metric}")
    lines.append("")
    lines.append(f"- **Entrada:** `{report.input_label}`")
    lines.append(f"- **Comparação:** "
                 f"{'pareada por seed' if report.paired else 'independente'}")
    lines.append(f"- **Correção:** {report.correction}")
    lines.append(f"- **Alpha:** {report.alpha:g}")
    if report.setups:
        lines.append(f"- **Setups:** {', '.join(report.setups)}")
    if report.scenarios:
        lines.append(f"- **Cenários:** {', '.join(report.scenarios)}")
    if report.levels:
        lines.append(f"- **Níveis:** {', '.join(report.levels)}")
    lines.append("")
    lines.append("## Descritivas")
    lines.append("")
    lines.append("| Algoritmo | n | Média | Mediana | Desvio | IQR | "
                 f"IC {int(report.confidence * 100)}% da média |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in report.summaries:
        lines.append(
            f"| {summary.algorithm} | {summary.n} | {_fmt(summary.mean)} | "
            f"{_fmt(summary.median)} | {_fmt(summary.std)} | "
            f"{_fmt(summary.iqr)} | "
            f"[{_fmt(summary.ci_low)}, {_fmt(summary.ci_high)}] |")
    lines.append("")
    lines.append("## Comparações")
    lines.append("")
    if not report.comparisons:
        lines.append("Não há comparações (menos de dois algoritmos).")
    else:
        lines.append(
            "Diferença = `média(A) - média(B)`. Valores positivos favorecem A "
            "quando maior é melhor.")
        lines.append("")
        lines.append("| A | B | n | Diferença [IC] | Teste | p | p corrigido | "
                     "Efeito | Significativo |")
        lines.append("| --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |")
        for comparison in report.comparisons:
            n_label = (f"{comparison.n} pares" if comparison.paired
                       else f"{comparison.n_a}/{comparison.n_b}")
            lines.append(
                f"| {comparison.a} | {comparison.b} | {n_label} | "
                f"{_fmt(comparison.diff)} "
                f"[{_fmt(comparison.diff_ci_low)}, "
                f"{_fmt(comparison.diff_ci_high)}] | "
                f"{comparison.method} | {_fmt(comparison.p_value, 4)} | "
                f"{_fmt(comparison.p_adjusted, 4)} | "
                f"{comparison.effect_name}={_fmt(comparison.effect)} | "
                f"{'sim' if comparison.significant else 'não'} |")
    lines.append("")
    lines.append("## Conclusão")
    lines.append("")
    significant = [c for c in report.comparisons if c.significant]
    if not report.comparisons:
        lines.append("Sem comparações para concluir.")
    elif significant:
        lines.append(
            f"{len(significant)} de {len(report.comparisons)} comparações "
            f"permanecem significativas em alpha={report.alpha:g} após "
            f"correção {report.correction}:")
        for comparison in significant:
            direction = "A > B" if comparison.diff > 0 else "A < B"
            lines.append(
                f"- **{comparison.a} × {comparison.b}**: {direction} "
                f"(diferença {_fmt(comparison.diff)}, "
                f"p corrigido {_fmt(comparison.p_adjusted, 4)}, "
                f"{comparison.effect_name} {_fmt(comparison.effect)}).")
    else:
        lines.append(
            f"Nenhuma comparação ficou significativa em alpha={report.alpha:g} "
            f"após correção {report.correction}. Não há evidência estatística "
            "suficiente para declarar um vencedor.")
    if report.summaries:
        best = max(report.summaries, key=lambda s: s.mean)
        lines.append("")
        lines.append(
            f"Maior média observada: **{best.algorithm}** "
            f"({_fmt(best.mean)}; n={best.n}). Isso é descritivo; "
            "significância está na tabela acima.")
    lines.append("")
    if report.warnings:
        lines.append("## Avisos")
        lines.append("")
        for warning in report.warnings:
            lines.append(f"- {warning}")
        lines.append("")
    lines.append("## Como reproduzir")
    lines.append("")
    lines.append("```bash")
    lines.append("python -m arenas.common.report --input <caminho> "
                 f"--metric {report.metric}")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def report_to_dict(report: Report) -> dict:
    """Versão serializável (JSON) do relatório."""
    return asdict(report)


def _json_safe(value):
    """Converte NaN/Infinity em `null` para gerar JSON/HTML válidos."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def report_to_json(report: Report) -> dict:
    """Versão serializável e seguro para JSON (NaN/Infinity → null)."""
    return _json_safe(report_to_dict(report))


def render_html(report: Report) -> str:
    """Gera a interface HTML interativa com os dados embutidos."""
    payload = report_to_json(report)
    subtitle = (f"Análise visual de {report.metric} · "
                f"{'pareado' if report.paired else 'independente'} · "
                f"correção {report.correction}")
    return _render_html(payload,
                        title=f"Relatório — {report.metric}",
                        subtitle=subtitle)


# ---------------------------------------------------------------------------
# CLI


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True,
                        help="arquivo CSV/JSONL/JSON ou diretório de resultados")
    parser.add_argument("--metric", default="reward_mean",
                        help="nome da métrica (usado no título e como fallback)")
    parser.add_argument("--value-field", default=None,
                        help="campo do valor (aceita caminho com pontos)")
    parser.add_argument("--algorithm-field", default=None,
                        help="campo do algoritmo (aceita caminho com pontos)")
    parser.add_argument("--seed-field", default=None,
                        help="campo da seed/execução (necessário com --paired)")
    parser.add_argument("--setup-field", default=None)
    parser.add_argument("--scenario-field", default=None)
    parser.add_argument("--level-field", default=None)
    parser.add_argument("--filter", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="filtra registros por campo (pode repetir)")
    parser.add_argument("--paired", action="store_true",
                        help="compara por seed (exige --seed-field ou coluna seed)")
    parser.add_argument("--aggregate", action="store_true",
                        help="agrega linhas repetidas da mesma seed/unidade "
                             "pela média antes de comparar")
    parser.add_argument("--baseline", default=None,
                        help="compara todos contra este algoritmo")
    parser.add_argument("--correction", default="holm",
                        choices=["holm", "holm-bonferroni", "bh",
                                 "benjamini-hochberg", "none"],
                        help="correção para múltiplas comparações")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--rng", type=int, default=7,
                        help="seed do bootstrap")
    parser.add_argument("--out", default="report.md",
                        help="saída .md (Markdown), .json ou .html (interativo)")
    return parser.parse_args(argv)


def _parse_filters(items: list[str]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--filter inválido: {item!r} (use KEY=VALUE)")
        key, value = item.split("=", 1)
        filters[key.strip()] = value.strip()
    return filters


def main(argv=None) -> None:
    args = parse_args(argv)
    if not 0 < args.alpha < 1:
        raise SystemExit("--alpha deve estar entre 0 e 1")
    if not 0 < args.confidence < 1:
        raise SystemExit("--confidence deve estar entre 0 e 1")
    if args.bootstrap < 1:
        raise SystemExit("--bootstrap deve ser >= 1")
    observations, warnings = load_observations(
        args.input,
        metric=args.metric,
        value_field=args.value_field,
        algorithm_field=args.algorithm_field,
        seed_field=args.seed_field,
        setup_field=args.setup_field,
        scenario_field=args.scenario_field,
        level_field=args.level_field,
        filters=_parse_filters(args.filter),
    )
    if args.aggregate:
        observations = aggregate_by_unit(observations, warnings=warnings)
    try:
        report = build_report(
            observations,
            metric=args.metric,
            input_label=str(args.input),
            paired=args.paired,
            baseline=args.baseline,
            correction=args.correction,
            alpha=args.alpha,
            confidence=args.confidence,
            n_boot=args.bootstrap,
            rng=args.rng,
        )
    except ValueError as exc:
        raise SystemExit(f"não foi possível gerar o relatório: {exc}") from exc
    report.warnings = warnings + report.warnings
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".json":
        out.write_text(
            json.dumps(_json_safe(report_to_dict(report)),
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
    elif out.suffix.lower() in (".html", ".htm"):
        out.write_text(render_html(report), encoding="utf-8")
    else:
        out.write_text(render_markdown(report), encoding="utf-8")
    print(f"relatório salvo em {out}")
    if report.comparisons:
        significant = sum(1 for c in report.comparisons if c.significant)
        print(f"{significant}/{len(report.comparisons)} comparações "
              f"significativas em alpha={args.alpha:g}")


if __name__ == "__main__":
    main()
