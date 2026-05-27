"""GPU mutex coordinator for ll-router.

Backends declare an optional ``service`` (systemd unit) and ``mutex_group``.
Before proxying, ``ensure_active`` makes sure the right service is up and any
competing service from the same group is stopped. Used so two llama.cpp
instances do not try to coexist on the same GPU.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Iterable

import httpx

from .config import Backend, Config


log = logging.getLogger(__name__)


# Single global lock — transitions are rare and must be serialized to avoid
# double-stop / double-start races between concurrent requests.
_TRANSITION_LOCK = asyncio.Lock()

# Cold-load of a 23 GiB Q4 GGUF from disk + cuBLAS init can take ~60–120s.
_HEALTH_TIMEOUT_S = 180.0
_HEALTH_INTERVAL_S = 1.0
_SYSTEMCTL_TIMEOUT_S = 30.0


async def _run(*argv: str, timeout: float = _SYSTEMCTL_TIMEOUT_S) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError(f"timeout running {' '.join(argv)}")
    return proc.returncode or 0, stdout.decode(errors="replace").strip()


async def _is_active(service: str) -> bool:
    rc, _ = await _run("systemctl", "is-active", "--quiet", service)
    return rc == 0


async def _stop(service: str) -> None:
    log.info("orchestrator: stopping %s", service)
    rc, out = await _run("sudo", "-n", "systemctl", "stop", service)
    if rc != 0:
        raise RuntimeError(f"systemctl stop {service} failed: {out}")


async def _start(service: str) -> None:
    log.info("orchestrator: starting %s", service)
    rc, out = await _run("sudo", "-n", "systemctl", "start", service)
    if rc != 0:
        raise RuntimeError(f"systemctl start {service} failed: {out}")


async def _wait_healthy(url: str, timeout: float = _HEALTH_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout
    last_error: str = ""
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            try:
                resp = await client.get(url)
                if resp.status_code < 400:
                    log.info("orchestrator: %s healthy (%d)", url, resp.status_code)
                    return
                last_error = f"status {resp.status_code}"
            except Exception as exc:  # noqa: BLE001
                last_error = type(exc).__name__
            await asyncio.sleep(_HEALTH_INTERVAL_S)
    raise RuntimeError(f"health check timeout for {url}: {last_error}")


def _peers_in_group(config: Config, target: Backend) -> Iterable[Backend]:
    if not target.mutex_group:
        return ()
    return [
        b
        for b in config.backends
        if b is not target
        and b.mutex_group == target.mutex_group
        and b.service
        and b.service != target.service
    ]


async def ensure_active(config: Config, target: Backend) -> None:
    """Make sure ``target.service`` is the only active service in its group.

    Serialized through a single lock so concurrent requests don't race on
    stop/start. Stays cheap on the steady-state path: when target is already
    active and no peer is running, the lock body is just two ``is-active``
    checks. The slow path (peer stop + target start + health wait) only fires
    on a real transition.
    """
    if not target.service:
        return

    async with _TRANSITION_LOCK:
        target_active = await _is_active(target.service)
        active_peers = [
            p for p in _peers_in_group(config, target)
            if await _is_active(p.service)
        ]

        if target_active and not active_peers:
            return

        for peer in active_peers:
            await _stop(peer.service)

        if not target_active:
            await _start(target.service)
            if target.health_url:
                await _wait_healthy(target.health_url)
