#!/usr/bin/env bash
# Retoma os experimentos pausados (ver docs/experimentos/retomada-20261120.md).
#
# Uso:
#   scripts/retomar_experimentos.sh            # checa memória, sobe a Fila,
#                                              # despausa e dá retry no nível 3
#   scripts/retomar_experimentos.sh --status   # só mostra o estado atual
#   scripts/retomar_experimentos.sh --force    # ignora o aviso de memória
#
# O script é idempotente: se a Fila já estiver no ar ou o job já estiver
# rodando/fila, ele não duplica nada.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/albertini@win.ufu.br/venvs/main/bin/python}"
STATE="${QUEUE_STATE:-experiments/queue/queue.json}"
LOGS_DIR="${QUEUE_LOGS_DIR:-experiments/queue/logs}"
URL="${QUEUE_URL:-http://127.0.0.1:8765}"
JOB_NIVEL3="${JOB_NIVEL3:-job-20261120-183301-9e4e70}"
JOB_NIVEL4="${JOB_NIVEL4:-job-20261120-223455-65d70a}"
MEM_MIN_GIB="${MEM_MIN_GIB:-3}"

MODE="run"
for arg in "$@"; do
    case "$arg" in
        --status) MODE="status" ;;
        --force)  MODE="force" ;;
        -h|--help)
            sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "argumento desconhecido: $arg (use --status, --force ou --help)" >&2
            exit 2 ;;
    esac
done

info()  { printf '\033[1m== %s\033[0m\n' "$*"; }
aviso() { printf 'AVISO: %s\n' "$*" >&2; }

[ -x "$PYTHON" ] || { aviso "interpretador não encontrado: $PYTHON (defina PYTHON=...)"; exit 1; }
[ -f "$STATE" ]   || { aviso "estado da fila não encontrado: $STATE"; exit 1; }

# ---------------------------------------------------------------------------
# 1. Memória: o vazamento do Samba (rpcd_lsad) foi quem derrubou os treinos.
# ---------------------------------------------------------------------------
mem_avail_gib="$(awk '/^MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo)"
leak_count="$(pgrep -fc rpcd_lsad 2>/dev/null || true)"
leak_count="${leak_count:-0}"
swap_used="$(awk '/^SwapTotal|^SwapFree/ {print $2}' /proc/meminfo | paste -sd' ' | awk '{printf "%.0f%%", ($1-$2)*100/$1}')"

info "memória: disponível ${mem_avail_gib} GiB · swap em uso ${swap_used} · processos rpcd_lsad: ${leak_count}"

mem_ok=1
if [ "$leak_count" -gt 0 ]; then
    mem_ok=0
    aviso "há ${leak_count} processos rpcd_lsad (vazamento do Samba). Rode, como root:"
    printf '        sudo pkill -f rpcd_lsad\n'
fi
if awk -v a="$mem_avail_gib" -v m="$MEM_MIN_GIB" 'BEGIN {exit !(a < m)}'; then
    mem_ok=0
    aviso "apenas ${mem_avail_gib} GiB disponíveis (mínimo sugerido: ${MEM_MIN_GIB} GiB);"
    printf '        o systemd-oomd pode matar o treino de novo.\n'
fi
if [ "$MODE" != "status" ] && [ "$mem_ok" -eq 0 ] && [ "$MODE" != "force" ]; then
    aviso "abortando para não perder outra rodada; use --force para ignorar."
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Fila: sobe o servidor se preciso e mostra/retoma o estado.
# ---------------------------------------------------------------------------
info "Fila de Experimentos em ${URL} (estado: ${STATE})"
"$PYTHON" - "$STATE" "$LOGS_DIR" "$URL" "$JOB_NIVEL3" "$JOB_NIVEL4" "$MODE" <<'PY'
import sys
from pathlib import Path

from arenas.common.job_queue_client import _request, ensure_server, get_job, health

state, logs_dir, url, job3, job4, mode = sys.argv[1:7]
if not health(url):
    print("  servidor fora do ar; iniciando...")
    ensure_server(url=url, state=state, logs_dir=logs_dir)
    if not health(url):
        raise SystemExit("  não foi possível subir o servidor da Fila")
    print("  servidor no ar")

payload = _request(f"{url}/api/state")
print(f"  paused: {payload.get('paused')} | max_parallel: {payload.get('max_parallel')}")
for job in payload.get("jobs", []):
    if job["status"] in {"queued", "running", "paused"}:
        print(f"  {job['status']:<8} {job['id']}  {job['name'][:44]}"
              f"  deps={job['depends_on']}")

if mode == "status":
    raise SystemExit(0)

if payload.get("paused"):
    _request(f"{url}/api/queue/resume", "POST", {})
    print("  fila despausada")

job = get_job(url, job3)
if job is None:
    raise SystemExit(f"  job do nível 3 não encontrado: {job3}")
print(f"  nível 3: {job['status']}")
if job["status"] in {"failed", "canceled"}:
    retried = _request(f"{url}/api/jobs/{job3}/retry", "POST", {})
    print(f"  retry aplicado -> {retried['status']} (retoma dos marcadores DONE)")
elif job["status"] in {"queued", "running"}:
    print("  já está na fila/rodando; nada a fazer")
else:
    print("  já concluído; confira o nível 4 abaixo")

job = get_job(url, job4)
if job is not None:
    print(f"  nível 4: {job['status']} (deps={job['depends_on']}) — entra após o nível 3")

_log = Path(logs_dir) / f"{job3}.log"
print(f"  acompanhe: tail -f {_log}")
PY

info "pronto"
printf '  acompanhar o job:   tail -f %s/%s.log\n' "$LOGS_DIR" "$JOB_NIVEL3"
printf '  acompanhar o treino: tail -f experiments/EXP-20261120-ppo-vs-eppo-level3-10seeds/seed-123/eppo/train.log\n'
printf '  painel web:         %s\n' "$URL"
