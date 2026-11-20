"""Análise estatística de resultados a partir da Fila de Experimentos.

Este módulo não reimplementa estatística: descobre os arquivos de resultado
de um experimento, infere a unidade independente e delega descritivas, IC,
testes, tamanho de efeito e correção a [`report.py`](report.py). A interface
web usa o HTML interativo de [`report_html.py`](report_html.py).

Convenções de unidade:

- **primária**: uma linha por seed de treino/execução (`train_seed`, `seed`,
  `run_id`, `trial`);
- **secundária**: uma linha por episódio de avaliação (`episode`,
  `episode_seed`, `eval_seed`); várias linhas da mesma run não são runs
  independentes, então a análise avisa sobre pseudo-replicação.

Todos os caminhos são relativos à raiz do repositório e são recusados se
escaparem dela.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .report import (
    aggregate_by_unit,
    build_report,
    load_observations,
    render_html,
    report_to_json,
)

ROOT = Path(__file__).resolve().parents[2]

SEED_FIELDS = ("train_seed", "seed", "eval_seed", "episode_seed", "episode",
               "run_id", "trial")
PRIMARY_SEED_FIELDS = {"train_seed", "seed", "run_id", "trial"}
SECONDARY_SEED_FIELDS = {"episode", "episode_seed", "eval_seed"}
ALGORITHM_FIELDS = ("algorithm", "alg", "policy", "tag", "run", "run_name")
NON_METRIC_FIELDS = set(SEED_FIELDS) | {"episode", "step", "index",
                                        "timestamp", "time"}
RESULT_SUFFIXES = {".csv", ".jsonl", ".ndjson"}


class AnalysisError(ValueError):
    """Entrada inválida para a análise (caminho, coluna ou dados)."""


# ---------------------------------------------------------------------------
# Caminhos


def resolve_under_root(value: str | Path) -> Path:
    """Resolve um caminho e exige que ele fique dentro do repositório."""
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise AnalysisError(
            f"caminho fora do repositório: {path}") from exc
    return path


def relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Leitura de resultados


def _iter_records(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                yield row
        return
    if suffix in {".jsonl", ".ndjson"}:
        for line in path.read_text(encoding="utf-8",
                                   errors="replace").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                yield item
        return
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8",
                                            errors="replace"))
        if isinstance(payload, dict):
            payload = payload.get("records") or payload.get("results") or []
        if not isinstance(payload, list):
            raise AnalysisError(f"{path}: JSON deve ser uma lista de registros")
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    raise AnalysisError(f"formato não suportado: {suffix or path.name}")


def _is_number(value) -> bool:
    if value is None or (isinstance(value, str) and not value.strip()):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def inspect_results(path: str | Path) -> dict:
    """Descreve um arquivo de resultados: colunas, métricas e algoritmos."""
    resolved = resolve_under_root(path)
    if not resolved.is_file():
        raise AnalysisError(f"arquivo de resultados não encontrado: {resolved}")
    try:
        records = list(_iter_records(resolved))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"{resolved}: não foi possível ler ({exc})") from exc
    if not records:
        raise AnalysisError(f"{resolved}: nenhum registro encontrado")

    columns: list[str] = []
    for record in records:
        for key in record:
            if key not in columns:
                columns.append(str(key))

    metrics: list[str] = []
    for column in columns:
        if column in NON_METRIC_FIELDS:
            continue
        values = [record.get(column) for record in records
                  if record.get(column) not in (None, "")]
        if values and sum(1 for value in values if _is_number(value)) >= max(
                1, len(values) // 2):
            metrics.append(column)

    algorithm_field = next(
        (field for field in ALGORITHM_FIELDS if field in columns), None)
    algorithms = []
    if algorithm_field:
        algorithms = sorted(
            {str(record.get(algorithm_field)) for record in records
             if record.get(algorithm_field) not in (None, "")},
            key=str.lower)

    seed_fields = [field for field in SEED_FIELDS if field in columns]
    suggested_metric = next(
        (candidate for candidate in ("reward_mean", "reward", "coins_mean",
                                     "coins", "raroc", "score")
         if candidate in metrics),
        metrics[0] if metrics else None)

    return {
        "path": relative_to_root(resolved),
        "name": resolved.name,
        "rows": len(records),
        "columns": columns,
        "metrics": metrics,
        "seed_fields": seed_fields,
        "algorithm_field": algorithm_field,
        "algorithms": algorithms,
        "suggested_metric": suggested_metric,
        "suggested_seed_field": seed_fields[0] if seed_fields else None,
    }


def _collect_paths(value, target: list[Path]) -> None:
    if isinstance(value, list):
        for item in value:
            _collect_paths(item, target)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_paths(item, target)
    elif isinstance(value, str) and Path(value).suffix.lower() in RESULT_SUFFIXES:
        target.append(Path(value))


def discover_sources(experiment_data: dict | None,
                     extra_paths: list[str] | None = None) -> list[dict]:
    """Lista os arquivos de resultado de um experimento.

    Considera os artefatos declarados no YAML (`results_csv`, `stats`,
    `experiment_dir`) e caminhos extras (artefatos do job), sem sair do
    repositório. Arquivos ilegíveis são ignorados.
    """
    data = experiment_data or {}
    artifacts = data.get("artifacts") or {}
    candidates: list[Path] = []
    _collect_paths(artifacts, candidates)
    for raw in extra_paths or []:
        if Path(raw).suffix.lower() in RESULT_SUFFIXES:
            candidates.append(Path(raw))

    experiment_dir = artifacts.get("experiment_dir")
    if isinstance(experiment_dir, str):
        try:
            base = resolve_under_root(experiment_dir)
        except AnalysisError:
            base = None
        if base is not None and base.is_dir():
            for pattern in ("results*.csv", "results*.jsonl", "*.jsonl"):
                for path in sorted(base.rglob(pattern))[:50]:
                    candidates.append(path)

    sources: list[dict] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = resolve_under_root(candidate)
        except AnalysisError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        try:
            sources.append(inspect_results(resolved))
        except AnalysisError:
            continue
    return sources


# ---------------------------------------------------------------------------
# Análise


def analysis_unit(seed_field: str | None) -> tuple[str | None, str, str]:
    """Classifica a unidade independente em primária/secundária."""
    if not seed_field:
        return None, "unspecified", (
            "unidade independente não identificada: o arquivo não tem coluna "
            "de seed; a análise descritiva é possível, mas não há pareamento")
    if seed_field in PRIMARY_SEED_FIELDS:
        return seed_field, "primary", (
            f"análise primária: unidade independente = {seed_field} "
            "(seed de treino/execução)")
    if seed_field in SECONDARY_SEED_FIELDS:
        return seed_field, "secondary", (
            f"análise secundária: unidade independente = {seed_field} "
            "(episódio de avaliação); várias linhas da mesma run não são "
            "runs independentes — há risco de pseudo-replicação")
    return seed_field, "secondary", (
        f"unidade independente = {seed_field} (não reconhecida como seed de "
        "treino); trate como análise secundária")


def analyze_results(
    input_path: str | Path,
    *,
    metric: str | None = None,
    value_field: str | None = None,
    algorithm_field: str | None = None,
    seed_field: str | None = None,
    setup_field: str | None = None,
    filters: dict[str, str] | None = None,
    aggregate: bool = False,
    paired: bool = False,
    baseline: str | None = None,
    correction: str = "holm",
    alpha: float = 0.05,
    confidence: float = 0.95,
    n_boot: int = 10000,
    rng: int = 7,
) -> dict:
    """Gera o relatório estatístico de um arquivo de resultados.

    Devolve descritivas, comparações, avisos e o HTML interativo. Levanta
    `AnalysisError` para entradas inválidas.
    """
    path = resolve_under_root(input_path)
    if not path.is_file():
        raise AnalysisError(f"arquivo de resultados não encontrado: {path}")
    inspection = inspect_results(path)

    metric = metric or inspection["suggested_metric"]
    if not metric:
        raise AnalysisError(
            f"{path}: nenhuma coluna numérica de métrica encontrada")
    seed_field = seed_field or inspection["suggested_seed_field"]
    if paired and not seed_field:
        raise AnalysisError(
            "comparação pareada exige uma coluna de seed/episódio; escolha "
            "a unidade independente")
    if baseline and inspection["algorithms"] and baseline not in inspection[
            "algorithms"]:
        raise AnalysisError(
            f"baseline {baseline!r} não está entre os algoritmos do arquivo: "
            + ", ".join(inspection["algorithms"]))
    if not 0 < alpha < 1:
        raise AnalysisError("alpha deve estar entre 0 e 1")
    if not 0 < confidence < 1:
        raise AnalysisError("confidence deve estar entre 0 e 1")
    if n_boot < 1:
        raise AnalysisError("bootstrap deve ser >= 1")

    try:
        observations, warnings = load_observations(
            path, metric=metric, value_field=value_field,
            algorithm_field=algorithm_field, seed_field=seed_field,
            setup_field=setup_field, filters=filters or {})
    except SystemExit as exc:
        raise AnalysisError(str(exc)) from exc
    if not observations:
        raise AnalysisError(f"{path}: nenhuma observação numérica utilizável")

    if aggregate:
        observations = aggregate_by_unit(observations, warnings=warnings)

    try:
        report = build_report(
            observations, metric=metric, input_label=relative_to_root(path),
            paired=paired, baseline=baseline, correction=correction,
            alpha=alpha, confidence=confidence, n_boot=n_boot, rng=rng)
    except SystemExit as exc:
        raise AnalysisError(str(exc)) from exc
    except ValueError as exc:
        raise AnalysisError(str(exc)) from exc

    unit, quality, note = analysis_unit(seed_field)
    report.warnings = list(warnings) + list(report.warnings)
    if quality in {"secondary", "unspecified"}:
        report.warnings.insert(0, note)

    payload = report_to_json(report)
    return {
        "input": relative_to_root(path),
        "metric": metric,
        "analysis_unit": unit,
        "analysis_quality": quality,
        "analysis_note": note,
        "aggregate": bool(aggregate),
        "paired": report.paired,
        "summaries": payload["summaries"],
        "comparisons": payload["comparisons"],
        "warnings": report.warnings,
        "report": payload,
        "html": render_html(report),
        "columns": inspection["columns"],
        "algorithms": inspection["algorithms"],
    }
