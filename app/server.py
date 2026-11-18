"""SmartNotify: simple local HTML frontend and Ollama OpenAI-compatible API.

Start: python app/server.py  (Python 3.10+, pip install openai).
Ollama native endpoints: /api/tags for discovery and /api/create for num_ctx aliases.
ALL LLM generations, including warmup, use /v1/chat/completions.
"""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import secrets
from threading import Lock
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from smartnotify_core import Config, SmartNotifyError, classify, generate_reply, make_client, complete

ROOT = Path(__file__).resolve().parent
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
BASE_MODEL = "qwen3.5:2b-q4_K_M"  # preferred default if installed, not a restriction
CONTEXTS = (1024, 1536, 2048, 3072, 4096, 8192)
ALIAS_PREFIX = "smartnotify-local-"
_lock = Lock()
# A tuple becomes connected only AFTER its OpenAI-compatible warmup succeeds.
_connections: dict[tuple[str, int], str] = {}


class AppError(Exception):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def ollama_api(path: str, payload: dict | None = None, timeout: int = 40) -> dict:
    url = OLLAMA_URL + path
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST" if payload is not None else "GET")
    try:
        with urlopen(req, timeout=timeout) as resp:
            value = json.loads(resp.read().decode("utf-8"))
            if not isinstance(value, dict):
                raise AppError("Ollama повернула некоректний JSON.")
            return value
    except HTTPError as exc:
        detail = exc.read(400).decode("utf-8", "replace")
        raise AppError(f"Ollama повернула HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise AppError("Ollama недоступна. Перевір, чи запущено Ollama (ollama serve).") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AppError("Ollama повернула некоректний JSON.") from exc


def installed_models() -> list[str]:
    """Return only real, locally available Ollama model tags (not app aliases)."""
    raw = ollama_api("/api/tags", timeout=8)
    models = raw.get("models")
    if not isinstance(models, list):
        raise AppError("Ollama повернула некоректний перелік моделей.")
    names = set()
    for entry in models:
        if isinstance(entry, dict):
            name = entry.get("name")
            if isinstance(name, str) and name and not name.startswith("smartnotify-"):
                names.add(name)
    return sorted(names, key=str.casefold)


def parse_selection(data: dict) -> tuple[str, int]:
    model = data.get("model")
    if not isinstance(model, str) or not model.strip() or len(model) > 200:
        raise AppError("Обери встановлену модель.", 400)
    try:
        ctx = int(data.get("num_ctx", 4096))
    except (TypeError, ValueError, OverflowError):
        raise AppError("Некоректна довжина контексту.", 400)
    if ctx not in CONTEXTS:
        raise AppError("Недопустимий розмір контексту.", 400)
    return model, ctx


def alias_for(model: str, ctx: int) -> str:
    """Unique and deterministic per (original model name, context size)."""
    stem = re.sub(r"[^a-z0-9]+", "-", model.casefold()).strip("-")[:24] or "model"
    digest = hashlib.sha256(model.encode("utf-8")).hexdigest()[:12]
    return f"{ALIAS_PREFIX}{stem}-{digest}:ctx{ctx}"


def context_model(model: str, ctx: int) -> str:
    """Configure real num_ctx using an alias with unchanged local model weights."""
    if ctx not in CONTEXTS:
        raise AppError("Недопустимий розмір контексту.", 400)
    with _lock:
        raw = ollama_api("/api/tags", timeout=8)
        tags = raw.get("models", [])
        if not isinstance(tags, list):
            raise AppError("Не вдалося прочитати моделі Ollama.")
        names = {item.get("name") for item in tags if isinstance(item, dict)}
        if model.startswith("smartnotify-") or model not in names:
            raise AppError(f"Модель '{model}' не встановлена в Ollama. Онови список.", 422)
        alias = alias_for(model, ctx)
        if alias not in names:
            result = ollama_api("/api/create", {
                "model": alias, "from": model,
                "parameters": {"num_ctx": ctx}, "stream": False,
            }, timeout=180)
            if result.get("status") != "success":
                raise AppError(f"Не вдалося встановити контекст: {result}")
        return alias


def connect(data: dict) -> dict:
    model, ctx = parse_selection(data)
    alias = context_model(model, ctx)
    config = Config(model=alias, base_url=OLLAMA_URL + "/v1", timeout_seconds=180)
    t0 = perf_counter()
    try:
        with make_client(config) as client:
            response = complete(client,
                model=alias,
                messages=[{"role": "user", "content": "Reply with the single word READY."}],
                temperature=0, seed=secrets.randbelow(2**31),
                max_tokens=12, reasoning_effort="none", stream=False,
            )
        answer = response.choices[0].message.content if response.choices else None
        if not isinstance(answer, str) or not answer.strip():
            raise SmartNotifyError("Модель повернула порожню відповідь на прогрів.")
    except SmartNotifyError as exc:
        raise AppError(f"Не вдалося прогріти модель: {exc}") from exc
    elapsed = round(perf_counter() - t0, 3)
    with _lock:
        _connections[(model, ctx)] = alias
    usage = response.usage
    return {"connected": True, "model": model, "num_ctx": ctx, "alias": alias,
            "warmup": {"elapsed_s": elapsed,
                       "prompt_tokens": usage.prompt_tokens if usage else None,
                       "completion_tokens": usage.completion_tokens if usage else None}}


def parse_seed(value) -> int:
    # A blank field means a newly sampled seed for each generation.
    if value is None or value == "":
        return secrets.randbelow(2**31)
    if type(value) is not int or not 0 <= value < 2**31:
        raise AppError("Seed має бути цілим числом від 0 до 2147483647 або порожнім.", 400)
    return value


def read_settings(data: dict) -> tuple[Config, dict]:
    model, num_ctx = parse_selection(data)
    try:
        temp = float(data.get("temperature", 0))
        max_tokens = int(data.get("max_tokens", 220))
    except (ValueError, TypeError, OverflowError):
        raise AppError("Некоректні параметри моделі.", 400)
    if not math.isfinite(temp) or not (0 <= temp <= 1) or not (64 <= max_tokens <= 320):
        raise AppError("Параметри поза допустимими межами.", 400)
    with _lock:
        alias = _connections.get((model, num_ctx))
    if alias is None:
        raise AppError("Спочатку натисни «Підключити» для вибраної моделі та контексту.", 409)
    seed = parse_seed(data.get("seed"))
    config = Config(model=alias, base_url=OLLAMA_URL + "/v1", temperature=temp,
                    seed=seed, max_tokens=max_tokens, target_num_ctx=num_ctx,
                    timeout_seconds=180)
    return config, {"model": model, "alias": alias, "temperature": temp,
                    "num_ctx": num_ctx, "max_tokens": max_tokens, "seed": seed}


class Handler(BaseHTTPRequestHandler):
    server_version = "SmartNotify/1.2"
    protocol_version = "HTTP/1.1"

    def send_json(self, obj: dict, status: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/api/models", "/api/status"):
            try:
                models = installed_models()
                with _lock:
                    connected = [{"model": model, "num_ctx": ctx} for (model, ctx) in _connections
                                 if model in models]
                self.send_json({"models": models, "connected": connected,
                                "preferred_model": BASE_MODEL, "ollama_online": True})
            except AppError as exc:
                self.send_json({"error": str(exc), "models": [], "connected": [],
                                "ollama_online": False}, status=503)
            return
        paths = {"/": "index.html", "/index.html": "index.html",
                 "/style.css": "style.css", "/app.js": "app.js"}
        name = paths.get(self.path)
        if not name:
            self.send_error(404)
            return
        path = ROOT / name
        try:
            content = path.read_bytes()
        except OSError:
            self.send_error(404)
            return
        typ = "text/html" if name.endswith(".html") else "text/css" if name.endswith(".css") else "application/javascript"
        self.send_response(200)
        self.send_header("Content-Type", typ + "; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        if self.path not in ("/api/connect", "/api/classify", "/api/reply"):
            self.send_json({"error": "Невідомий маршрут."}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= 20000:
                raise AppError("Некоректний розмір запиту.", 400)
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(data, dict):
                raise AppError("Очікується JSON-об'єкт.", 400)
            if self.path == "/api/connect":
                self.send_json(connect(data))
                return
            sender = data.get("sender", "")
            message = data.get("message", "")
            if not isinstance(sender, str) or not isinstance(message, str) or len(sender) > 120 or not 0 < len(message.strip()) <= 4000:
                raise AppError("Вкажи відправника та повідомлення (до 4000 символів).", 400)
            config, settings = read_settings(data)
            with make_client(config) as client:
                if self.path == "/api/classify":
                    result, metrics = classify(client, config, sender, message)
                else:
                    level = data.get("urgency_level")
                    if not isinstance(level, str):
                        raise AppError("Спочатку виконай класифікацію.", 400)
                    result, metrics = generate_reply(client, config, sender, message, level)
            self.send_json({"result": result, "metrics": metrics, "settings": settings})
        except AppError as exc:
            self.send_json({"error": str(exc)}, exc.status)
        except SmartNotifyError as exc:
            self.send_json({"error": str(exc)}, 502)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.send_json({"error": f"Помилка JSON: {exc}"}, 400)
        except Exception as exc:
            self.send_json({"error": f"Помилка сервера: {exc}"}, 500)


def main():
    parser = argparse.ArgumentParser(description="SmartNotify local HTML app")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    httpd.daemon_threads = True
    print(f"SmartNotify: http://127.0.0.1:{args.port} (Ctrl+C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nЗупинено.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
