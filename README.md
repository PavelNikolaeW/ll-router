# ll-router

A small **FastAPI** service that fronts multiple [`llama.cpp`](https://github.com/ggerganov/llama.cpp)
`llama-server` backends behind a **single OpenAI-compatible API**
(`/v1/models`, `/v1/chat/completions`, `/v1/completions`).

Point any OpenAI client at one endpoint; `ll-router` maps a model **alias** to the
right backend, rewrites the request for the underlying GGUF, applies per-model
default overrides, and guards each backend with a concurrency semaphore.

## Why

A self-hosted LLM rig runs several `llama-server` instances (plus a hot-swapping
router) across limited VRAM. Clients shouldn't care which box or GGUF serves a
request — they just ask for `gemma-31b` or `qwen-35b`. `ll-router` is that
indirection layer.

## How a request flows

1. Client POSTs to `/v1/chat/completions` with `model: <alias>`.
2. The alias resolves to `(backend, gguf_basename)`; the body's `model` field is
   rewritten to the GGUF basename the upstream server expects.
3. Per-model **request overrides** (e.g. disabling thinking for Gemma) are
   deep-merged into the body — **client values always win**, overrides only fill holes.
4. The request is forwarded once the backend's concurrency semaphore is acquired
   (backends typically allow one in-flight request given VRAM/slot limits).
5. Streaming responses pass through as `text/event-stream`; non-streaming forwards
   the JSON body and status verbatim.

## Layout

| File | Role |
|---|---|
| `ll_router/main.py` | FastAPI app + `run()` entrypoint (uvicorn) |
| `ll_router/config.py` | `Config` / `Backend` dataclasses, YAML loading, alias → backend map |
| `ll_router/proxy.py` | request proxying, streaming, per-backend semaphores, override deep-merge |
| `config.yaml` | alias → backend / GGUF mapping |
| `bench_*.py` | ad-hoc benchmarking scripts (VRAM, context length, MoE, throughput, quality) |

## Run

```bash
pip install -e .
python -m ll_router.main      # or the `ll-router` console script
```

Configuration via env vars: `LL_ROUTER_HOST` (default `0.0.0.0`),
`LL_ROUTER_PORT` (default `8000`), `LL_ROUTER_RELOAD`, `LL_ROUTER_PROFILES_PATH`.
Backends are defined in `config.yaml`, read once at startup — restart after edits.
