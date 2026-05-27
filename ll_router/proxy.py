from __future__ import annotations

import asyncio
import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

from .config import Backend


# backend_name -> semaphore
_semaphores: dict[str, asyncio.Semaphore] = {}


# Module-level client shared by all requests. Без переиспользования каждый
# POST поднимал свой TCP-handshake к loopback (~30ms на embed-вызовы,
# незаметно для chat). Лимиты: до 64 одновременных соединений суммарно,
# 32 keepalive-сокета держим открытыми, по 60s. Таймауты подобраны под самый
# долгий путь (cold prefill в llama-server может занять минуты).
_LIMITS = httpx.Limits(
    max_connections=64,
    max_keepalive_connections=32,
    keepalive_expiry=60.0,
)
_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)
_HTTPX = httpx.AsyncClient(limits=_LIMITS, timeout=_TIMEOUT)


async def aclose() -> None:
    """Close the shared client. Wire from FastAPI shutdown to avoid leaking
    sockets on `systemctl restart ll-router`."""
    await _HTTPX.aclose()


def _get_semaphore(backend: Backend) -> asyncio.Semaphore:
    if backend.name not in _semaphores:
        _semaphores[backend.name] = asyncio.Semaphore(backend.max_concurrent)
    return _semaphores[backend.name]


def _deep_merge(dst: dict, src: dict) -> dict:
    """Recursively merge src into dst. dst wins on existing scalar/list keys
    (клиент явно прислал → не перетираем); только заполняем дыры из src.

    Это то поведение что мы хотим для request_overrides из профиля: профиль
    задаёт дефолты, явный запрос клиента всегда сильнее.
    """
    for k, v in src.items():
        if k in dst:
            if isinstance(dst[k], dict) and isinstance(v, dict):
                _deep_merge(dst[k], v)
            # else: client value wins, do nothing
        else:
            dst[k] = v
    return dst


async def proxy_request(
    request: Request,
    backend: Backend,
    path: str,
    body: dict,
    overrides: dict | None = None,
) -> StreamingResponse | JSONResponse:
    sem = _get_semaphore(backend)
    url = f"{backend.url.rstrip('/')}/{path.lstrip('/')}"

    if overrides:
        _deep_merge(body, overrides)

    await sem.acquire()
    try:
        if body.get("stream"):
            return await _proxy_stream(url, body, sem)
        else:
            return await _proxy_sync(url, body, sem)
    except Exception:
        sem.release()
        raise


async def _proxy_sync(
    url: str,
    body: dict,
    sem: asyncio.Semaphore,
) -> JSONResponse:
    try:
        resp = await _HTTPX.post(url, json=body)
        return JSONResponse(
            content=resp.json(),
            status_code=resp.status_code,
        )
    finally:
        sem.release()


async def _proxy_stream(
    url: str,
    body: dict,
    sem: asyncio.Semaphore,
) -> StreamingResponse:
    async def generate():
        try:
            async with _HTTPX.stream("POST", url, json=body) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk
        finally:
            sem.release()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
