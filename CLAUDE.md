# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`ll-router` is a small FastAPI service that fronts multiple llama.cpp `llama-server` backends and exposes a single OpenAI-compatible API (`/v1/models`, `/v1/chat/completions`, `/v1/completions`). It is the public API surface; the heavy lifting (model loading, slot management, hot-swapping GGUFs) happens in the upstream `llama-home-router` and direct `llama-server` instances it proxies to.

The repository is **not** a git repo — there is no commit history to consult.

## Code layout

Only three Python files matter; everything in the repo root prefixed with `bench_` is ad-hoc benchmarking scripts unrelated to the service.

- `ll_router/main.py` — FastAPI app + entrypoint (`run()`), uvicorn launch. `app = FastAPI(...)` is module-level and `config = load_config(...)` runs at import time, so importing the module hits the filesystem.
- `ll_router/config.py` — `Config`, `Backend` dataclasses, YAML loading, and the model-profiles override loader. Builds an alias→(backend, backend_model) map on construction.
- `ll_router/proxy.py` — `proxy_request` + streaming/non-streaming helpers; per-backend `asyncio.Semaphore` for `max_concurrent`; `_deep_merge` for applying profile overrides.

## Architecture: how a request flows

1. Client POSTs to `/v1/chat/completions` (or `/v1/completions`) with `model: <alias>`.
2. `Config.get_backend(alias)` returns `(Backend, backend_model_name)`. The `backend_model_name` is the GGUF basename — that's the key everything else uses.
3. Body's `model` field is rewritten to the GGUF basename (so the upstream llama-server / llama-home-router recognizes it).
4. `overrides_for(backend_model)` looks up request overrides keyed by GGUF basename (loaded from `/etc/llama-home/model-profiles.json`, path overridable via `LL_ROUTER_PROFILES_PATH`).
5. `proxy.py::_deep_merge` merges those overrides into the body, with **client values winning** — overrides only fill holes. This is intentional: profile sets defaults (e.g. `chat_template_kwargs.enable_thinking=false` for Gemma), client can override.
6. Request is forwarded to `backend.url` after acquiring the backend's semaphore (`max_concurrent`, typically 1 because the underlying llama-server has limited slots/VRAM).
7. Streaming responses pass through as `text/event-stream`; non-streaming forwards the JSON body and status code.

The semaphore is released in `finally` for both sync and stream paths; `_proxy_stream` keeps the httpx client open for the lifetime of the generator.

## The llama-home-router relationship

`config.yaml` points several backend entries (e.g. `gemma-26b-a4b`, `qwen-35b-a3b`) at `http://localhost:8001` — that's **llama-home-router**, not a llama-server directly. llama-home-router holds slot A and hot-swaps GGUFs on demand based on the request's model field. First request after a model switch incurs slot-reload latency.

Other backends (e.g. `gemma-31b-mtp` on `:8202`) bypass llama-home-router and hit a dedicated `llama-server` systemd unit directly because they hold a model statically. Comments in `config.yaml` document which is which — preserve those comments when editing.

`/etc/llama-home/model-profiles.json` is owned by the llama-home stack, not this repo; ll-router only reads the `request_overrides` field per model entry.

## Running and operating

Install / develop:
```
.venv/bin/pip install -e .         # editable install, project script `ll-router`
.venv/bin/python -m ll_router.main # run directly
```

Env vars (read in `main.run()`):
- `LL_ROUTER_HOST` (default `0.0.0.0`)
- `LL_ROUTER_PORT` (default `8000`)
- `LL_ROUTER_RELOAD` (`1`/`true` to enable uvicorn reload)
- `LL_ROUTER_PROFILES_PATH` (override path to model-profiles.json)

Production runs under systemd as `ll-router.service` (see `systemd/ll-router.service`). The service expects `WorkingDirectory=/home/pavel/ll-router` because `main.py` resolves `config.yaml` relative to the module's parent.

There is no test suite and no linter config. `bench.py` is a load test that hits a running router on `localhost:8000`.

## Operational scripts (`scripts/`)

These manipulate the upstream `llama-server-gemma-26b-a4b` systemd unit via `sudo`, not ll-router itself:

- `gemma-mode <preset>` — swap `/etc/default/llama-server-gemma-26b-a4b` to a preset from `systemd/presets/*.env`, then restart both the llama-server unit and ll-router. Has aliases (`gemma-max-vram`, `gemma-max-ctx`, etc.) — see the script.
- `gemma-slot-save <name> [slot_id]` / `gemma-slot-restore` — POST to `:8001/slots/<id>?action=save|restore` to persist a llama.cpp slot's KV cache to disk.
- `gemma-main-slot-cycle <save-name> <preset>` — save slot 0 then swap presets in one shot.

Editing `config.yaml` requires a router restart (`systemctl restart ll-router.service`) because the file is read once at import.

## Conventions worth knowing

- Some comments in `config.py` / `proxy.py` are in Russian — keep both languages consistent with what's already there if extending; don't translate existing comments unnecessarily.
- The "model alias → backend / backend_model" indirection in `config.yaml` is the only mapping layer; the same GGUF basename can appear under multiple aliases (see `gemma-31b` + `gemma-31b-mtp` both pointing at `gemma-4-31B-it-Q4_K_M.gguf`).
- `max_concurrent: 1` is the norm — increasing it will let multiple requests hit the same llama-server slot, which usually isn't what you want.
