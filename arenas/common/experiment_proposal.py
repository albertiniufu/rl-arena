"""Geração e validação de propostas de experimento (Markdown + YAML).

A validação é a fonte única usada pela skill `experimenter-helper`
(`.agents/skills/experimenter-helper/scripts/validate_yaml.py`) e pela aba
**Proposta de experimentos** da Fila de Experimentos.

O formulário da interface envia campos *planos* (ex.: `seeds_train: "7,42"`,
`command: "python train.py ..."`); `build_proposal` normaliza tudo para o
YAML estruturado esperado pela fila e pelos scripts de BO/ASHA. Nenhum
comando é executado aqui: só se valida que ele começa por Python e que
caminhos apontam para dentro do repositório.

Salvar nunca sobrescreve um arquivo existente, a menos que `overwrite=True`
seja pedido explicitamente.
"""
from __future__ import annotations

import os
import re
import shlex
import unicodedata
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

STATUSES = {"planned", "running", "done", "aborted"}
ARENAS = {"seek_coin", "risk", "economy"}
TUNING_METHODS = {"none", "bo", "asha", "grid", "random"}
CHECKPOINTS = {"latest", "best_validation", "fixed_step"}

REQUIRED = {
    "id": str,
    "status": str,
    "question": str,
    "hypothesis": str,
    "falsification": str,
    "setup": dict,
    "algorithms": dict,
    "metrics": dict,
    "budget": dict,
    "tuning": dict,
    "evaluation": dict,
    "statistics": dict,
    "decision_criteria": dict,
    "execution": dict,
    "artifacts": dict,
    "registration": dict,
}


class ProposalError(ValueError):
    """Proposta inválida; `errors` traz todas as mensagens."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


class ProposalConflict(FileExistsError):
    """Já existe um arquivo da proposta e `overwrite=False`."""

    def __init__(self, paths):
        self.paths = [str(path) for path in paths]
        super().__init__("proposta já existe: " + ", ".join(self.paths))


# ---------------------------------------------------------------------------
# Normalização de campos do formulário


def _text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _number(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip().replace(",", ".")
    try:
        number = float(text)
    except ValueError:
        return value.strip()
    return int(number) if number.is_integer() else number


def _int_or_none(value):
    number = _number(value)
    return number if isinstance(number, int) else None


def _bool(value, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "sim", "yes", "on"}


def _list(value) -> list:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if _text(item)]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _lines(value) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if _text(item)]
    return [line.strip() for line in str(value).splitlines() if line.strip()]


def _int_list(value) -> list[int]:
    seeds = []
    for item in _list(value):
        number = _int_or_none(item)
        if number is not None:
            seeds.append(number)
    return seeds


def _command(value) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(part) for part in value if str(part)]
    return shlex.split(str(value))


def slugify(value) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return re.sub(r"-+", "-", text)


# ---------------------------------------------------------------------------
# Construção do YAML estruturado


def build_proposal(fields: dict, *, now: datetime | None = None) -> dict:
    """Normaliza os campos planos do formulário no payload estruturado."""
    fields = dict(fields or {})
    now = now or datetime.now().astimezone()
    title = _text(fields.get("title")) or "Experimento"
    slug = _text(fields.get("slug")) or slugify(title) or "experimento"
    exp_id = _text(fields.get("id")) or f"EXP-{now:%Y%m%d}-{slug}"

    arena = _text(fields.get("arena")) or "seek_coin"
    try:
        command = _command(fields.get("command"))
    except ValueError:
        # Aspas desbalanceadas: a validação recusa o comando vazio resultante.
        command = []
    cwd = _text(fields.get("cwd")) or "."
    max_runtime = _number(fields.get("max_runtime"))
    if max_runtime is None:
        max_runtime = 21600

    seeds_train = _int_list(fields.get("seeds_train"))
    n_runs = _int_or_none(fields.get("n_runs"))
    if n_runs is None and seeds_train:
        n_runs = len(seeds_train)
    wall_hours = _number(fields.get("wall_hours_per_run"))

    env_steps = fields.get("env_steps_per_run")
    if isinstance(env_steps, str) and env_steps.strip():
        env_steps = env_steps.strip()

    method = _text(fields.get("tuning_method")) or "none"
    space_file = _text(fields.get("space_file")) or None

    payload = {
        "id": exp_id,
        "status": _text(fields.get("status")) or "planned",
        "created_at": now.isoformat(timespec="seconds"),
        "author": _text(fields.get("author")),
        "title": title,
        "question": _text(fields.get("question")),
        "hypothesis": _text(fields.get("hypothesis")),
        "falsification": _text(fields.get("falsification")),
        "claim_allowed": _text(fields.get("claim_allowed")),
        "evidence": {
            "prior_experiments": _list(fields.get("prior_experiments")),
            "existing_setups": _list(fields.get("existing_setups")),
            "gap": _text(fields.get("gap")),
            "analysis_only": _bool(fields.get("analysis_only")),
        },
        "setup": {
            "arena": arena,
            "level": _int_or_none(fields.get("level")),
            "scenario": _text(fields.get("scenario")) or None,
            "setup_id": _text(fields.get("setup_id")) or None,
            "config_files": _list(fields.get("config_files")),
            "frame_stack": _int_or_none(fields.get("frame_stack")),
            "frame_skip": _int_or_none(fields.get("frame_skip")),
            "episode_seconds": _number(fields.get("episode_seconds")),
            "reward": _text(fields.get("reward")),
        },
        "algorithms": {
            "candidates": _list(fields.get("candidates")),
            "baseline": _list(fields.get("baseline")),
            "ablations": _list(fields.get("ablations")),
        },
        "metrics": {
            "primary": {
                "name": _text(fields.get("metric_name")),
                "direction": _text(fields.get("metric_direction")) or "maximize",
                "definition": _text(fields.get("metric_definition")),
                "target": _number(fields.get("metric_target")),
            },
            "secondary": _list(fields.get("secondary_metrics")),
        },
        "budget": {
            "seeds_train": seeds_train,
            "seeds_eval": _int_list(fields.get("seeds_eval")),
            "eval_episodes": _int_or_none(fields.get("eval_episodes")),
            "env_steps_per_run": env_steps or None,
            "wall_hours_per_run": wall_hours,
            "n_runs": n_runs,
            "hardware": _text(fields.get("hardware")),
        },
        "tuning": {
            "method": method,
            "space_file": space_file,
            "budget_hours": _number(fields.get("tuning_budget_hours")),
            "trials": _int_or_none(fields.get("tuning_trials")),
            "validation_seeds": _int_list(fields.get("validation_seeds")),
            "selection_rule": _text(fields.get("selection_rule")),
        },
        "evaluation": {
            "checkpoint": _text(fields.get("checkpoint")) or "latest",
            "greedy": _bool(fields.get("greedy"), True),
            "truncation": _text(fields.get("truncation")),
            "same_protocol": _bool(fields.get("same_protocol"), True),
        },
        "statistics": {
            "independent_unit": _text(fields.get("independent_unit"))
            or "train_seed",
            "paired": _bool(fields.get("paired"), True),
            "test": _text(fields.get("test")) or "wilcoxon",
            "confidence": _number(fields.get("confidence")) or 0.95,
            "effect_size": _text(fields.get("effect_size")) or "cohens_d",
            "correction": _text(fields.get("correction")) or "holm",
            "primary_analysis": _text(fields.get("primary_analysis")),
            "secondary_analysis": _text(fields.get("secondary_analysis")),
        },
        "risks": _lines(fields.get("risks")),
        "decision_criteria": {
            "support": _text(fields.get("support")),
            "inconclusive": _text(fields.get("inconclusive")),
            "abandon": _text(fields.get("abandon")),
        },
        "execution": {
            "command": command,
            "cwd": cwd,
            "max_runtime": max_runtime,
            "priority": _int_or_none(fields.get("priority")) or 0,
            "depends_on": _list(fields.get("depends_on")),
        },
        "artifacts": {
            "experiment_dir": _text(fields.get("experiment_dir"))
            or f"experiments/{exp_id}",
            "checkpoints": _list(fields.get("checkpoints")),
            "results_csv": _text(fields.get("results_csv")) or None,
            "reports": _list(fields.get("reports")),
            "videos": [],
            "stats": [],
            "documents": [
                f"docs/experimentos/{exp_id}.md",
                f"docs/experimentos/{exp_id}.yaml",
            ],
            "commands": [],
        },
        "registration": {
            "registry_entry": _text(fields.get("registry_entry"))
            or "docs/experimentos/registro.md",
            "setup_entry": _text(fields.get("setup_entry")) or None,
            "backlog_item": _text(fields.get("backlog_item")) or None,
        },
        "references": _lines(fields.get("references")),
    }
    return payload


# ---------------------------------------------------------------------------
# Validação


def _non_empty(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def _validate_execution(execution: dict) -> list[str]:
    """Exige Python e caminhos dentro do repositório (sem shell)."""
    errors: list[str] = []
    try:
        command = _command(execution.get("command"))
    except ValueError:
        return ["execution.command inválido"]
    if not command:
        return []
    first = Path(command[0]).name
    if "python" not in first:
        errors.append(
            "execution.command deve começar por Python (sem shell)")
        return errors
    cwd = _text(execution.get("cwd")) or "."
    cwd_path = Path(cwd)
    if not cwd_path.is_absolute():
        cwd_path = ROOT / cwd_path
    try:
        cwd_path = cwd_path.resolve()
        cwd_path.relative_to(ROOT)
    except (OSError, ValueError):
        errors.append("execution.cwd deve estar dentro do repositório")
        return errors
    if len(command) >= 2 and command[1] not in ("-c", "-m"):
        target = (cwd_path / command[1]).resolve()
        try:
            target.relative_to(ROOT)
        except ValueError:
            errors.append(
                "execution.command aponta para um script fora do repositório")
    return errors


def validate(payload: dict) -> list[str]:
    """Valida o YAML estruturado; devolve a lista de erros."""
    errors = []
    for key, expected in REQUIRED.items():
        if key not in payload:
            errors.append(f"campo obrigatório ausente: {key}")
            continue
        if not isinstance(payload[key], expected):
            errors.append(f"{key}: esperado {expected.__name__}")
        elif not _non_empty(payload[key]):
            errors.append(f"{key}: não pode ser vazio")

    if payload.get("status") not in STATUSES:
        errors.append("status deve ser planned/running/done/aborted")

    exp_id = str(payload.get("id", ""))
    if not re.fullmatch(r"EXP-\d{8}-[a-z0-9-]+", exp_id):
        errors.append("id deve seguir EXP-AAAAMMDD-slug")

    setup = payload.get("setup") or {}
    if setup.get("arena") not in ARENAS:
        errors.append("setup.arena deve ser seek_coin/risk/economy")

    algorithms = payload.get("algorithms") or {}
    if not _non_empty(algorithms.get("candidates")):
        errors.append("algorithms.candidates não pode ser vazio")
    if not _non_empty(algorithms.get("baseline")):
        errors.append("algorithms.baseline não pode ser vazio")

    metrics = payload.get("metrics") or {}
    primary = metrics.get("primary") or {}
    if not _non_empty(primary.get("name")):
        errors.append("metrics.primary.name não pode ser vazio")
    if primary.get("direction") not in {"maximize", "minimize"}:
        errors.append("metrics.primary.direction deve ser maximize/minimize")
    if not _non_empty(primary.get("definition")):
        errors.append("metrics.primary.definition não pode ser vazio")

    budget = payload.get("budget") or {}
    if not _non_empty(budget.get("seeds_train")):
        errors.append("budget.seeds_train não pode ser vazio")
    if len(budget.get("seeds_train") or []) < 3:
        errors.append("budget.seeds_train deve ter 3+ seeds")
    if not _non_empty(budget.get("eval_episodes")):
        errors.append("budget.eval_episodes não pode ser vazio")
    if not _non_empty(budget.get("env_steps_per_run")):
        errors.append("budget.env_steps_per_run não pode ser vazio")
    if not _non_empty(budget.get("wall_hours_per_run")):
        errors.append("budget.wall_hours_per_run não pode ser vazio")
    if not _non_empty(budget.get("n_runs")):
        errors.append("budget.n_runs não pode ser vazio")

    tuning = payload.get("tuning") or {}
    if tuning.get("method") not in TUNING_METHODS:
        errors.append("tuning.method inválido")
    if tuning.get("method") in {"bo", "asha"}:
        if not _non_empty(tuning.get("space_file")):
            errors.append("tuning.space_file obrigatório para bo/asha")
        if not _non_empty(tuning.get("budget_hours")):
            errors.append("tuning.budget_hours obrigatório para bo/asha")

    evaluation = payload.get("evaluation") or {}
    if evaluation.get("checkpoint") not in CHECKPOINTS:
        errors.append("evaluation.checkpoint inválido")
    if evaluation.get("greedy") is not True:
        errors.append("evaluation.greedy deve ser true")
    if evaluation.get("same_protocol") is not True:
        errors.append("evaluation.same_protocol deve ser true")

    statistics = payload.get("statistics") or {}
    if not _non_empty(statistics.get("independent_unit")):
        errors.append("statistics.independent_unit não pode ser vazio")
    confidence = statistics.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 < confidence < 1:
        errors.append("statistics.confidence deve estar em (0, 1)")
    if not _non_empty(statistics.get("test")):
        errors.append("statistics.test não pode ser vazio")
    if not _non_empty(statistics.get("effect_size")):
        errors.append("statistics.effect_size não pode ser vazio")
    if not _non_empty(statistics.get("correction")):
        errors.append("statistics.correction não pode ser vazio")

    execution = payload.get("execution") or {}
    if not _non_empty(execution.get("command")):
        errors.append("execution.command não pode ser vazio")
    if not _non_empty(execution.get("cwd")):
        errors.append("execution.cwd não pode ser vazio")
    if not _non_empty(execution.get("max_runtime")):
        errors.append("execution.max_runtime não pode ser vazio")
    errors.extend(_validate_execution(execution))

    decisions = payload.get("decision_criteria") or {}
    for key in ("support", "inconclusive", "abandon"):
        if not _non_empty(decisions.get(key)):
            errors.append(f"decision_criteria.{key} não pode ser vazio")

    return errors


def validate_yaml_text(text: str) -> dict:
    """Valida um YAML em texto; devolve `{errors, proposal}`."""
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return {"errors": [f"YAML inválido: {exc}"], "proposal": None}
    if not isinstance(payload, dict):
        return {"errors": ["o YAML raiz deve ser um mapa"], "proposal": None}
    return {"errors": validate(payload), "proposal": payload}


# ---------------------------------------------------------------------------
# Renderização


def render_yaml(payload: dict) -> str:
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True,
                          default_flow_style=False)


def _bullet(value, empty: str = "—") -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) if value else empty
    if value is None or value == "":
        return empty
    return str(value)


def render_markdown(payload: dict) -> str:
    """Renderiza a proposta no formato narrativo de `TEMPLATE.md`."""
    setup = payload.get("setup") or {}
    algorithms = payload.get("algorithms") or {}
    metrics = payload.get("metrics") or {}
    primary = metrics.get("primary") or {}
    budget = payload.get("budget") or {}
    tuning = payload.get("tuning") or {}
    evaluation = payload.get("evaluation") or {}
    statistics = payload.get("statistics") or {}
    decisions = payload.get("decision_criteria") or {}
    execution = payload.get("execution") or {}
    artifacts = payload.get("artifacts") or {}
    evidence = payload.get("evidence") or {}
    registration = payload.get("registration") or {}
    risks = payload.get("risks") or []
    command = execution.get("command") or []
    command_text = " ".join(shlex.quote(str(part)) for part in command)

    hours = budget.get("wall_hours_per_run")
    runs = budget.get("n_runs")
    try:
        training_hours = float(hours) * float(runs)
        training_text = f"{training_hours:g} h"
    except (TypeError, ValueError):
        training_text = "—"
    tuning_hours = tuning.get("budget_hours")
    try:
        total_text = f"{float(tuning_hours) + training_hours:g} h"
    except (TypeError, ValueError):
        total_text = training_text

    lines: list[str] = []
    lines.append(f"# Proposta de experimento — {payload.get('id')}")
    lines.append("")
    lines.append("> Gerado pela aba **Proposta de experimentos** da Fila de")
    lines.append("> Experimentos. O YAML ao lado é a fonte de verdade para")
    lines.append("> automação; os dois arquivos precisam concordar.")
    lines.append("")
    lines.append("| Campo | Valor |")
    lines.append("| --- | --- |")
    lines.append(f"| **ID** | {_bullet(payload.get('id'))} |")
    lines.append(f"| **Status** | {_bullet(payload.get('status'))} |")
    lines.append(f"| **Data** | {_bullet(payload.get('created_at'))} |")
    lines.append(f"| **Autor** | {_bullet(payload.get('author'))} |")
    lines.append(f"| **Arena** | {_bullet(setup.get('arena'))} |")
    lines.append(f"| **Setup** | {_bullet(setup.get('setup_id'))} |")
    lines.append(f"| **Algoritmos** | {_bullet(algorithms.get('candidates'))} |")
    lines.append(f"| **Baseline** | {_bullet(algorithms.get('baseline'))} |")
    lines.append(f"| **Métrica primária** | "
                 f"{_bullet(primary.get('name'))} "
                 f"({_bullet(primary.get('direction'))}) |")
    lines.append(f"| **Unidade independente** | "
                 f"{_bullet(statistics.get('independent_unit'))} |")
    lines.append("")

    lines.append("## 1. Pergunta")
    lines.append("")
    lines.append(_bullet(payload.get("question")))
    lines.append("")
    lines.append("## 2. Hipótese")
    lines.append("")
    lines.append(_bullet(payload.get("hypothesis")))
    lines.append("")
    lines.append("## 3. Falsificação")
    lines.append("")
    lines.append(_bullet(payload.get("falsification")))
    lines.append("")
    lines.append("## 4. Evidência existente e lacuna")
    lines.append("")
    lines.append(f"- Experimentos relacionados: "
                 f"{_bullet(evidence.get('prior_experiments'))}")
    lines.append(f"- Setups existentes: "
                 f"{_bullet(evidence.get('existing_setups'))}")
    lines.append(f"- Lacuna: {_bullet(evidence.get('gap'))}")
    lines.append(f"- Somente análise (sem treino novo): "
                 f"{'sim' if evidence.get('analysis_only') else 'não'}")
    lines.append("")
    lines.append("**Claim permitido:** " + _bullet(payload.get("claim_allowed")))
    lines.append("")
    lines.append("## 5. Setup e algoritmos")
    lines.append("")
    lines.append(f"- Arena: {_bullet(setup.get('arena'))}")
    lines.append(f"- Nível / cenário: {_bullet(setup.get('level'))} / "
                 f"{_bullet(setup.get('scenario'))}")
    lines.append(f"- Setup: {_bullet(setup.get('setup_id'))}")
    lines.append(f"- Configs: {_bullet(setup.get('config_files'))}")
    lines.append(f"- `frame_stack` / `frame_skip`: "
                 f"{_bullet(setup.get('frame_stack'))} / "
                 f"{_bullet(setup.get('frame_skip'))}")
    lines.append(f"- Duração do episódio: {_bullet(setup.get('episode_seconds'))} s")
    lines.append(f"- Recompensa: {_bullet(setup.get('reward'))}")
    lines.append(f"- Candidatos: {_bullet(algorithms.get('candidates'))}")
    lines.append(f"- Baseline: {_bullet(algorithms.get('baseline'))}")
    lines.append(f"- Ablações: {_bullet(algorithms.get('ablations'))}")
    lines.append("")
    lines.append("## 6. Métricas")
    lines.append("")
    lines.append(f"- **Primária:** {_bullet(primary.get('name'))} "
                 f"(direção: {_bullet(primary.get('direction'))})")
    lines.append(f"- Definição: {_bullet(primary.get('definition'))}")
    lines.append(f"- Alvo: {_bullet(primary.get('target'))}")
    lines.append(f"- Secundárias: {_bullet(metrics.get('secondary'))}")
    lines.append("")
    lines.append("## 7. Seeds e orçamento")
    lines.append("")
    lines.append(f"- Seeds de treino: {_bullet(budget.get('seeds_train'))}")
    lines.append(f"- Seeds de avaliação: {_bullet(budget.get('seeds_eval'))}")
    lines.append(f"- Episódios de avaliação: "
                 f"{_bullet(budget.get('eval_episodes'))}")
    lines.append(f"- Passos de ambiente por run: "
                 f"{_bullet(budget.get('env_steps_per_run'))}")
    lines.append(f"- Horas por run: {_bullet(budget.get('wall_hours_per_run'))}")
    lines.append(f"- Número de runs: {_bullet(budget.get('n_runs'))}")
    lines.append(f"- Hardware: {_bullet(budget.get('hardware'))}")
    lines.append("")
    lines.append("## 8. Tuning")
    lines.append("")
    lines.append(f"- Método: {_bullet(tuning.get('method'))}")
    lines.append(f"- Espaço: {_bullet(tuning.get('space_file'))}")
    lines.append(f"- Orçamento: {_bullet(tuning.get('budget_hours'))} h; "
                 f"trials: {_bullet(tuning.get('trials'))}")
    lines.append(f"- Seeds de validação: "
                 f"{_bullet(tuning.get('validation_seeds'))}")
    lines.append(f"- Regra de seleção: {_bullet(tuning.get('selection_rule'))}")
    lines.append("")
    lines.append("## 9. Avaliação")
    lines.append("")
    lines.append(f"- Checkpoint: {_bullet(evaluation.get('checkpoint'))}")
    lines.append(f"- Política gulosa: "
                 f"{'sim' if evaluation.get('greedy') else 'não'}")
    lines.append(f"- Truncamento: {_bullet(evaluation.get('truncation'))}")
    lines.append(f"- Mesmo protocolo para todos os braços: "
                 f"{'sim' if evaluation.get('same_protocol') else 'não'}")
    lines.append("")
    lines.append("## 10. Análise estatística planejada")
    lines.append("")
    lines.append(f"- Unidade independente: "
                 f"{_bullet(statistics.get('independent_unit'))}")
    lines.append(f"- Pareado: {'sim' if statistics.get('paired') else 'não'}")
    lines.append(f"- Teste: {_bullet(statistics.get('test'))}")
    lines.append(f"- Confiança: {_bullet(statistics.get('confidence'))}")
    lines.append(f"- Tamanho de efeito: "
                 f"{_bullet(statistics.get('effect_size'))}")
    lines.append(f"- Correção: {_bullet(statistics.get('correction'))}")
    lines.append(f"- Primária: {_bullet(statistics.get('primary_analysis'))}")
    lines.append(f"- Secundária: "
                 f"{_bullet(statistics.get('secondary_analysis'))}")
    lines.append("")
    lines.append("## 11. Custo estimado")
    lines.append("")
    lines.append("| Item | Estimativa |")
    lines.append("| --- | ---: |")
    lines.append(f"| Runs de treino | {_bullet(budget.get('n_runs'))} |")
    lines.append(f"| Horas de treino | {training_text} |")
    lines.append(f"| Horas de tuning | {_bullet(tuning_hours)} |")
    lines.append(f"| Total | {total_text} |")
    lines.append("")
    lines.append("## 12. Riscos e confounds")
    lines.append("")
    if risks:
        lines.extend(f"- {risk}" for risk in risks)
    else:
        lines.append("- —")
    lines.append("")
    lines.append("## 13. Critério de decisão")
    lines.append("")
    lines.append(f"- **Suporta:** {_bullet(decisions.get('support'))}")
    lines.append(f"- **Inconclusivo:** {_bullet(decisions.get('inconclusive'))}")
    lines.append(f"- **Abandona:** {_bullet(decisions.get('abandon'))}")
    lines.append("")
    lines.append("## 14. Comandos previstos")
    lines.append("")
    lines.append("```bash")
    lines.append(command_text or "# (defina execution.command)")
    lines.append("```")
    lines.append("")
    lines.append("Relatório estatístico:")
    lines.append("")
    lines.append("```bash")
    results_csv = artifacts.get("results_csv") or (
        f"{artifacts.get('experiment_dir')}/results.csv")
    lines.append(
        f"python -m arenas.common.report --input {results_csv} "
        f"--metric {primary.get('name') or 'reward_mean'} "
        f"--paired --out {artifacts.get('experiment_dir')}/report.html")
    lines.append("```")
    lines.append("")
    lines.append("## 15. Execução na Fila de Experimentos")
    lines.append("")
    lines.append("```yaml")
    lines.append(yaml.safe_dump(execution, sort_keys=False,
                                allow_unicode=True).rstrip())
    lines.append("```")
    lines.append("")
    lines.append("## 16. Artefatos")
    lines.append("")
    lines.append(f"- Pasta do experimento: "
                 f"{_bullet(artifacts.get('experiment_dir'))}")
    lines.append(f"- CSV de resultados: "
                 f"{_bullet(artifacts.get('results_csv'))}")
    lines.append(f"- Relatórios: {_bullet(artifacts.get('reports'))}")
    lines.append(f"- Documentos: {_bullet(artifacts.get('documents'))}")
    lines.append(f"- Registro: {_bullet(registration.get('registry_entry'))}")
    lines.append("")
    lines.append("## 17. Checklist final")
    lines.append("")
    lines.append("- [ ] Pergunta, hipótese e falsificação.")
    lines.append("- [ ] Baseline forte e tuning comparável.")
    lines.append("- [ ] Mesmo orçamento em passos de ambiente.")
    lines.append("- [ ] Tuning separado da avaliação.")
    lines.append("- [ ] 3+ seeds de treino.")
    lines.append("- [ ] Métrica primária definida antes.")
    lines.append("- [ ] IC, teste, efeito e correção.")
    lines.append("- [ ] Custo declarado.")
    lines.append("- [ ] Registro em `docs/experimentos/`.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prévia, validação e gravação


def preview_proposal(fields: dict) -> dict:
    """Monta a proposta, valida e devolve YAML + Markdown (sem gravar)."""
    payload = build_proposal(fields)
    errors = validate(payload)
    return {
        "proposal": payload,
        "yaml": render_yaml(payload),
        "markdown": render_markdown(payload),
        "errors": errors,
    }


def _resolve_directory(directory: str | Path) -> Path:
    path = Path(directory)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ProposalError(
            [f"diretório fora do repositório: {path}"]) from exc
    return path


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_proposal(fields: dict, *, directory: str = "docs/experimentos",
                  overwrite: bool = False) -> dict:
    """Grava `EXP-....md` + `EXP-....yaml` sem sobrescrever por padrão."""
    preview = preview_proposal(fields)
    if preview["errors"]:
        raise ProposalError(preview["errors"])
    payload = preview["proposal"]
    base = _resolve_directory(directory)
    base.mkdir(parents=True, exist_ok=True)
    yaml_path = base / f"{payload['id']}.yaml"
    markdown_path = base / f"{payload['id']}.md"
    existing = [path for path in (yaml_path, markdown_path) if path.exists()]
    if existing and not overwrite:
        raise ProposalConflict(existing)
    _write_atomic(yaml_path, preview["yaml"])
    _write_atomic(markdown_path, preview["markdown"])
    return {
        "id": payload["id"],
        "yaml_path": str(yaml_path.relative_to(ROOT)),
        "markdown_path": str(markdown_path.relative_to(ROOT)),
        "proposal": payload,
        "yaml": preview["yaml"],
        "markdown": preview["markdown"],
        "errors": [],
    }
