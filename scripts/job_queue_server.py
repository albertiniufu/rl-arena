#!/usr/bin/env python3
"""Servidor local da fila de experimentos (API JSON + interface web).

Uso:
    python scripts/job_queue_server.py \
        --state experiments/queue/queue.json \
        --max-parallel 1 --port 8765

    python scripts/job_queue_server.py --load configs/queue-example.yaml

Abra http://127.0.0.1:8765/ no navegador.

Por padrão o servidor escuta apenas em 127.0.0.1 e só aceita comandos Python
do próprio repositório. Use `--allow-arbitrary` apenas se souber o que está
fazendo.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arenas.common.algorithm_catalog import algorithm_catalog  # noqa: E402
from arenas.common.bo_analysis import (  # noqa: E402
    BoAnalysisError,
    load_bo,
    load_bo_for_job,
)
from arenas.common.experiment_analysis import (  # noqa: E402
    AnalysisError,
    analyze_results,
    discover_sources,
)
from arenas.common.experiment_proposal import (  # noqa: E402
    ProposalConflict,
    ProposalError,
    preview_proposal,
    save_proposal,
    validate_yaml_text,
)
from arenas.common.job_queue import JobQueue  # noqa: E402
from arenas.common.job_queue_ui import HTML  # noqa: E402

DOC_PAGE = """<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ · RL Arena</title>
<style>
 body { margin:0; background:#f5f7fb; color:#0f172a;
        font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
 .wrap { max-width:940px; margin:0 auto; padding:26px 22px 70px; }
 .back { color:#2563eb; text-decoration:none; font-weight:600; font-size:.88rem; }
 article { background:#fff; border:1px solid #e2e8f0; border-radius:16px;
           padding:28px 32px; margin-top:14px;
           box-shadow:0 10px 30px rgba(15,23,42,.05); }
 article h1 { margin-top:0; letter-spacing:-.03em; }
 article h2 { margin-top:28px; }
 article table { width:100%; border-collapse:collapse; font-size:.9rem; }
 article th, article td { border-bottom:1px solid #e2e8f0; padding:8px;
                           text-align:left; vertical-align:top; }
 article th { color:#64748b; font-size:.78rem; text-transform:uppercase;
              letter-spacing:.05em; }
 article code { background:#f1f5f9; padding:1px 5px; border-radius:6px;
                font-size:.86em; }
 article pre { background:#0f172a; color:#dbeafe; padding:14px;
               border-radius:12px; overflow:auto; }
 article pre code { background:none; padding:0; }
 article blockquote { margin:0; padding:8px 14px; border-left:4px solid #93c5fd;
                       color:#475569; background:#f8fafc; }
 article img { max-width:100%; }
 article a { color:#2563eb; }
</style></head>
<body><div class="wrap">
<a class="back" href="/">← Fila de Experimentos</a>
<article>__BODY__</article>
</div></body></html>"""


def _filter_dict(value) -> dict[str, str]:
    if not value:
        return {}
    if isinstance(value, dict):
        return {str(key): str(item) for key, item in value.items()}
    filters: dict[str, str] = {}
    for item in str(value).split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"filtro inválido: {item!r} (use campo=valor)")
        key, expected = item.split("=", 1)
        filters[key.strip()] = expected.strip()
    return filters


class Handler(BaseHTTPRequestHandler):
    server_version = "RLArenaQueue/1.0"

    @property
    def queue(self) -> JobQueue:
        return self.server.queue  # type: ignore[attr-defined]

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode(),
                   "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _serve_file(self, relative: str) -> None:
        path = Path(relative)
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        try:
            path.relative_to(ROOT)
        except ValueError:
            self._error(403, "arquivo fora do repositório")
            return
        if not path.exists() or not path.is_file():
            self._error(404, "arquivo não encontrado")
            return
        data = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or (
            "application/octet-stream")
        range_header = self.headers.get("Range", "")
        if range_header.startswith("bytes="):
            try:
                start_s, end_s = range_header[6:].split("-", 1)
                start = int(start_s or 0)
                end = int(end_s) if end_s else len(data) - 1
                end = min(end, len(data) - 1)
                if start < 0 or start > end:
                    raise ValueError
                chunk = data[start:end + 1]
                self.send_response(206)
                self.send_header("Content-Type", content_type)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range",
                                 f"bytes {start}-{end}/{len(data)}")
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
                return
            except ValueError:
                pass
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_doc(self, relative: str) -> None:
        """Serve um documento do repositório; Markdown é renderizado."""
        relative = unquote(relative)
        path = (ROOT / relative).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError:
            self._error(403, "documento fora do repositório")
            return
        if not path.is_file():
            self._error(404, "documento não encontrado")
            return
        if path.suffix.lower() not in {".md", ".markdown"}:
            self._serve_file(str(path))
            return
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            import markdown
            body = markdown.markdown(
                text, extensions=["tables", "fenced_code", "toc"])
        except ImportError:
            body = f"<pre>{escape(text)}</pre>"
        page = (DOC_PAGE.replace("__TITLE__", escape(path.stem))
                .replace("__BODY__", body))
        self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("corpo JSON deve ser um objeto")
        return payload

    def log_message(self, fmt, *args):  # noqa: D401 - silencia o log padrão
        print(f"[queue] {self.address_string()} {fmt % args}", flush=True)

    @staticmethod
    def _analyze_payload(payload: dict) -> dict:
        if not payload.get("input"):
            raise ValueError("informe `input` com o arquivo de resultados")
        return analyze_results(
            payload["input"],
            metric=payload.get("metric") or None,
            value_field=payload.get("value_field") or None,
            algorithm_field=payload.get("algorithm_field") or None,
            seed_field=payload.get("seed_field") or None,
            setup_field=payload.get("setup_field") or None,
            filters=_filter_dict(payload.get("filter")),
            aggregate=bool(payload.get("aggregate")),
            paired=bool(payload.get("paired")),
            baseline=payload.get("baseline") or None,
            correction=payload.get("correction") or "holm",
            alpha=float(payload.get("alpha") or 0.05),
            confidence=float(payload.get("confidence") or 0.95),
            n_boot=int(payload.get("bootstrap") or 10000),
        )

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path in ("/", "/index.html"):
                self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/health":
                self._json(200, {"ok": True})
                return
            if parsed.path.startswith("/docs/"):
                self._serve_doc(parsed.path[len("/docs/"):])
                return
            if parsed.path == "/api/algorithms":
                self._json(200, algorithm_catalog())
                return
            if parsed.path == "/api/state":
                self._json(200, self.queue.snapshot())
                return
            if parsed.path == "/api/files":
                relative = parse_qs(parsed.query).get("path", [""])[0]
                self._serve_file(relative)
                return
            if parsed.path == "/api/bo":
                path = parse_qs(parsed.query).get("path", [""])[0]
                if not path:
                    raise ValueError("informe `path` da campanha de BO/ASHA")
                self._json(200, load_bo(path))
                return
            if (parsed.path.startswith("/api/jobs/")
                    and parsed.path.endswith("/bo")):
                job_id = parsed.path[len("/api/jobs/"):-len("/bo")].strip("/")
                job = self.queue.get(job_id)
                if job is None:
                    raise KeyError(job_id)
                try:
                    data = self.queue.experiment_data(job_id)
                except (KeyError, OSError, ValueError):
                    data = None
                self._json(200, {"id": job_id,
                                 **load_bo_for_job(job.to_dict(), data)})
                return
            if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/experiment"):
                job_id = parsed.path[len("/api/jobs/"):-len("/experiment")].strip("/")
                text = self.queue.experiment_yaml_text(job_id)
                self._json(200, {"id": job_id, "yaml": text})
                return
            if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/artifacts"):
                job_id = parsed.path[len("/api/jobs/"):-len("/artifacts")].strip("/")
                self._json(200, {"id": job_id,
                                 "artifacts": self.queue.experiment_artifacts(job_id)})
                return
            if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/results"):
                job_id = parsed.path[len("/api/jobs/"):-len("/results")].strip("/")
                data = self.queue.experiment_data(job_id)
                artifacts = self.queue.experiment_artifacts(job_id)
                extra = [item["path"] for item in artifacts
                         if item.get("exists")]
                self._json(200, {"id": job_id,
                                 "sources": discover_sources(data, extra),
                                 "artifacts": artifacts})
                return
            if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/log"):
                job_id = parsed.path[len("/api/jobs/"):-len("/log")].strip("/")
                lines = int(parse_qs(parsed.query).get("lines", ["200"])[0])
                self._json(200, {"id": job_id,
                                 "log": self.queue.tail(job_id, lines)})
                return
            self._error(404, "rota não encontrada")
        except KeyError as exc:
            self._error(404, f"job não encontrado: {exc}")
        except (AnalysisError, BoAnalysisError, ValueError) as exc:
            self._error(400, str(exc))
        except Exception as exc:  # noqa: BLE001
            self._error(500, str(exc))

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/experiments":
                payload = self._body()
                job = self.queue.add_experiment(
                    payload.get("path"),
                    name=payload.get("name"),
                    cwd=payload.get("cwd"),
                    priority=payload.get("priority"),
                    max_runtime=payload.get("max_runtime"),
                    tags=payload.get("tags"),
                )
                self._json(201, job.to_dict())
                return
            if parsed.path.startswith("/api/jobs/") and parsed.path.endswith(
                    "/analysis"):
                job_id = parsed.path[
                    len("/api/jobs/"):-len("/analysis")].strip("/")
                result = self._analyze_payload(self._body())
                self._json(200, {"id": job_id, **result})
                return
            if parsed.path == "/api/analysis":
                self._json(200, self._analyze_payload(self._body()))
                return
            if parsed.path == "/api/proposals/preview":
                payload = self._body()
                fields = payload.get("fields")
                if not isinstance(fields, dict):
                    raise ValueError("informe `fields` como objeto")
                self._json(200, preview_proposal(fields))
                return
            if parsed.path == "/api/proposals/validate":
                payload = self._body()
                if payload.get("yaml") is not None:
                    self._json(200, validate_yaml_text(str(payload["yaml"])))
                    return
                fields = payload.get("fields")
                if not isinstance(fields, dict):
                    raise ValueError("informe `yaml` ou `fields` como objeto")
                preview = preview_proposal(fields)
                self._json(200, {"errors": preview["errors"],
                                 "proposal": preview["proposal"]})
                return
            if parsed.path == "/api/proposals/save":
                payload = self._body()
                fields = payload.get("fields")
                if not isinstance(fields, dict):
                    raise ValueError("informe `fields` como objeto")
                saved = save_proposal(
                    fields,
                    directory=str(payload.get("directory")
                                  or "docs/experimentos"),
                    overwrite=bool(payload.get("overwrite")))
                self._json(201, saved)
                return
            if parsed.path == "/api/proposals/enqueue":
                payload = self._body()
                fields = payload.get("fields")
                if not isinstance(fields, dict):
                    raise ValueError("informe `fields` como objeto")
                saved = save_proposal(
                    fields,
                    directory=str(payload.get("directory")
                                  or "docs/experimentos"),
                    overwrite=bool(payload.get("overwrite")))
                job = self.queue.add_experiment(saved["yaml_path"],
                                                name=saved["id"])
                self._json(201, {"proposal": saved, "job": job.to_dict()})
                return
            if (parsed.path.startswith("/api/jobs/")
                    and parsed.path.endswith("/experiment/status")):
                job_id = parsed.path[
                    len("/api/jobs/"):-len("/experiment/status")].strip("/")
                result = self.queue.sync_experiment_status(job_id)
                self._json(200, result)
                return
            if parsed.path == "/api/jobs/local":
                payload = self._body()
                job = self.queue.add_local(
                    name=str(payload.get("name") or "local-job"),
                    command=payload.get("command"),
                    cwd=str(payload.get("cwd") or "."),
                    pid=int(payload.get("pid")),
                    priority=int(payload.get("priority") or 0),
                    depends_on=payload.get("depends_on") or [],
                    max_runtime=payload.get("max_runtime"),
                    artifacts=payload.get("artifacts") or [],
                    tags=payload.get("tags") or {},
                    experiment_yaml=payload.get("experiment_yaml"),
                )
                self._json(201, job.to_dict())
                return
            if parsed.path == "/api/jobs":
                payload = self._body()
                job = self.queue.add(
                    name=str(payload.get("name") or "job"),
                    command=payload.get("command"),
                    cwd=str(payload.get("cwd") or "."),
                    priority=int(payload.get("priority") or 0),
                    depends_on=payload.get("depends_on") or [],
                    max_runtime=payload.get("max_runtime"),
                    artifacts=payload.get("artifacts") or [],
                    tags=payload.get("tags") or {},
                    experiment_yaml=payload.get("experiment_yaml"),
                )
                self._json(201, job.to_dict())
                return
            if parsed.path == "/api/load":
                payload = self._body()
                path = Path(str(payload.get("path") or "")).resolve()
                try:
                    path.relative_to(ROOT)
                except ValueError as exc:
                    raise ValueError("YAML deve estar dentro do repositório") from exc
                jobs = self.queue.load_yaml(path)
                self._json(201, {"loaded": len(jobs),
                                 "ids": [job.id for job in jobs]})
                return
            if parsed.path == "/api/queue/pause":
                self.queue.pause()
                self._json(200, {"paused": True})
                return
            if parsed.path == "/api/queue/resume":
                self.queue.resume()
                self._json(200, {"paused": False})
                return
            if parsed.path.startswith("/api/jobs/"):
                parts = [part for part in parsed.path.split("/") if part]
                # /api/jobs/<id>/<action>
                if len(parts) != 4:
                    self._error(404, "rota não encontrada")
                    return
                job_id, action = parts[2], parts[3]
                if action == "cancel":
                    self._json(200, self.queue.cancel(job_id).to_dict())
                    return
                if action == "retry":
                    self._json(200, self.queue.retry(job_id).to_dict())
                    return
                if action == "finish":
                    payload = self._body()
                    self._json(200, self.queue.finish(
                        job_id,
                        exit_code=int(payload.get("exit_code", 0)),
                        error=str(payload.get("error") or "")).to_dict())
                    return
                if action == "move":
                    payload = self._body()
                    delta = int(payload.get("delta", 0))
                    self._json(200, self.queue.move(job_id, delta).to_dict())
                    return
            self._error(404, "rota não encontrada")
        except KeyError as exc:
            self._error(404, f"job não encontrado: {exc}")
        except ProposalConflict as exc:
            self._error(409, str(exc))
        except ProposalError as exc:
            self._error(400, "; ".join(exc.errors))
        except (AnalysisError, ValueError, TypeError,
                json.JSONDecodeError) as exc:
            self._error(400, str(exc))
        except Exception as exc:  # noqa: BLE001
            self._error(500, str(exc))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", default="experiments/queue/queue.json",
                        help="arquivo JSON de estado da fila")
    parser.add_argument("--logs-dir", default=None,
                        help="diretório dos logs (padrão: <state>/logs)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--load", default=None,
                        help="YAML de jobs para carregar ao iniciar")
    parser.add_argument("--allow-arbitrary", action="store_true",
                        help="permite qualquer comando (perigoso; padrão: só Python do repo)")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("AVISO: o servidor não tem autenticação; expor fora de "
              "localhost permite execução de jobs por terceiros.", flush=True)
    queue = JobQueue(
        args.state, logs_dir=args.logs_dir,
        max_parallel=args.max_parallel,
        poll_interval=args.poll_interval,
        allow_arbitrary=args.allow_arbitrary,
    )
    if args.load:
        jobs = queue.load_yaml(args.load)
        print(f"{len(jobs)} job(s) carregado(s) de {args.load}", flush=True)
    queue.start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.queue = queue  # type: ignore[attr-defined]
    print(f"Fila em http://{args.host}:{args.port}/", flush=True)
    print(f"estado: {queue.state_path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando servidor (jobs em execução continuam).", flush=True)
    finally:
        server.server_close()
        queue.stop(terminate=False)


if __name__ == "__main__":
    main()
