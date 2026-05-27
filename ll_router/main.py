from __future__ import annotations

import asyncio
import os
import time
import httpx
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from pathlib import Path

from .config import load_config
from .orchestrator import ensure_active
from . import proxy
from .proxy import proxy_request

config = load_config(Path(__file__).parent.parent / "config.yaml")
app = FastAPI(title="ll-router", version="0.2.1")


@app.on_event("shutdown")
async def _shutdown() -> None:
    await proxy.aclose()

_START_MONO = time.monotonic()
_LOAD_STATUS_PROBE_TIMEOUT_S = 1.0


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "uptime_seconds": round(time.monotonic() - _START_MONO, 1),
        "version": app.version,
    }


@app.get("/v1/models")
async def list_models():
    models = []
    for b in config.backends:
        for alias, backend_name in b.models.items():
            models.append({
                "id": alias,
                "object": "model",
                "created": 0,
                "owned_by": b.name,
                "kind": b.kind,
                "backend_type": b.type,
                "backend_model": backend_name,
                "request_overrides": config.overrides_for(backend_name),
            })
    return {"object": "list", "data": models}


async def _probe_backend(backend) -> dict:
    """Probe backend health with a short timeout. Returns shape:
    {"health_ok": bool, "status_code": int|None, "error": str|None, "probed_url": str}.

    Uses backend.health_url if set (deliberately distinct from backend.url for
    proxy-fronted backends like gemma-26b-a4b: proxy on :8001, actual
    llama-server on :8101). Otherwise falls back to `backend.url + /health`,
    which llama-server exposes on every instance.
    """
    probe_url = backend.health_url or f"{backend.url.rstrip('/')}/health"
    try:
        async with httpx.AsyncClient(timeout=_LOAD_STATUS_PROBE_TIMEOUT_S) as client:
            resp = await client.get(probe_url)
            return {
                "health_ok": resp.status_code < 400,
                "status_code": resp.status_code,
                "error": None,
                "probed_url": probe_url,
            }
    except Exception as exc:  # noqa: BLE001
        return {"health_ok": False, "status_code": None, "error": type(exc).__name__, "probed_url": probe_url}


@app.get("/v1/models/load_status")
async def models_load_status():
    """Per-alias liveness snapshot. Cheap (parallel HEAD-style probes with 1s
    timeout each), safe to call from clients before submitting expensive work
    so they can detect CPU-fallback / dead backend before a 60s LLM timeout.
    """
    probes = await asyncio.gather(*(_probe_backend(b) for b in config.backends))
    by_backend = {b.name: p for b, p in zip(config.backends, probes)}
    entries = []
    for b in config.backends:
        probe = by_backend[b.name]
        for alias, backend_name in b.models.items():
            entries.append({
                "id": alias,
                "backend": b.name,
                "backend_type": b.type,
                "kind": b.kind,
                "url": b.url,
                "backend_model": backend_name,
                "health_ok": probe["health_ok"],
                "health_status_code": probe["status_code"],
                "health_error": probe["error"],
                "probed_url": probe["probed_url"],
            })
    return {"object": "list", "data": entries}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "")

    result = config.get_backend(model)
    if not result:
        available = ", ".join(config.all_models())
        raise HTTPException(404, f"Model '{model}' not found. Available: {available}")

    backend, backend_model = result
    body["model"] = backend_model
    overrides = config.overrides_for(backend_model)
    client = request.client.host if request.client else "?"
    client_port = request.client.port if request.client else 0
    ua = request.headers.get("user-agent", "")[:60]
    print(f"REQ chat from={client}:{client_port} model={model!r} backend={backend.name} ua={ua!r}", flush=True)
    try:
        await ensure_active(config, backend)
    except RuntimeError as exc:
        raise HTTPException(503, f"backend not ready: {exc}")
    return await proxy_request(request, backend, "/v1/chat/completions", body, overrides)


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    model = body.get("model", "")

    result = config.get_backend(model)
    if not result:
        available = ", ".join(config.all_models())
        raise HTTPException(404, f"Model '{model}' not found. Available: {available}")

    backend, backend_model = result
    body["model"] = backend_model
    overrides = config.overrides_for(backend_model)
    try:
        await ensure_active(config, backend)
    except RuntimeError as exc:
        raise HTTPException(503, f"backend not ready: {exc}")
    return await proxy_request(request, backend, "/v1/completions", body, overrides)


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    model = body.get("model", "")

    result = config.get_embedding_backend(model)
    if not result:
        available = ", ".join(config.all_embedding_models()) or "(none configured)"
        raise HTTPException(404, f"Embedding model '{model}' not found. Available: {available}")

    backend, backend_model = result
    body["model"] = backend_model
    client = request.client.host if request.client else "?"
    client_port = request.client.port if request.client else 0
    ua = request.headers.get("user-agent", "")[:60]
    n_inputs = (
        len(body["input"]) if isinstance(body.get("input"), list)
        else (1 if "input" in body else 0)
    )
    print(f"REQ embed from={client}:{client_port} model={model!r} backend={backend.name} n={n_inputs} ua={ua!r}", flush=True)
    try:
        await ensure_active(config, backend)
    except RuntimeError as exc:
        raise HTTPException(503, f"backend not ready: {exc}")
    return await proxy_request(request, backend, "/v1/embeddings", body, overrides=None)


def run():
    host = os.getenv("LL_ROUTER_HOST", "0.0.0.0")
    port = int(os.getenv("LL_ROUTER_PORT", "8000"))
    reload = os.getenv("LL_ROUTER_RELOAD", "").lower() in {"1", "true", "yes", "on"}
    uvicorn.run("ll_router.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run()
