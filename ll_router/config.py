from __future__ import annotations

import json
import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path


PROFILES_PATH = Path(os.getenv(
    "LL_ROUTER_PROFILES_PATH",
    "/etc/llama-home/model-profiles.json",
))


@dataclass
class Backend:
    name: str
    url: str
    type: str  # "cpu" or "gpu"
    max_concurrent: int
    models: dict[str, str] = field(default_factory=dict)  # alias -> backend model name
    # systemd unit that must be active before proxying to this backend.
    # None = no orchestration needed (e.g. always-on services).
    service: str | None = None
    # Backends sharing a mutex_group cannot run simultaneously; orchestrator
    # stops the active service from the same group before starting `service`.
    mutex_group: str | None = None
    # HTTP endpoint polled after service start; non-2xx → keep waiting.
    # Also used by /v1/models/load_status for live probes.
    health_url: str | None = None
    # "chat" backends serve /v1/chat/completions + /v1/completions.
    # "embedding" backends serve /v1/embeddings. Routes are kept separate so
    # alias collisions across kinds are allowed (e.g. an embed model named
    # `bge-m3` cannot accidentally answer a chat request).
    kind: str = "chat"


@dataclass
class Config:
    backends: list[Backend]
    # gguf_basename -> request_overrides dict (from model-profiles.json).
    # Применяется в proxy: deep-merge в body запроса, чтобы напр. для gemma
    # автоматически летел chat_template_kwargs.enable_thinking=false.
    overrides_by_basename: dict[str, dict] = field(default_factory=dict)

    # alias -> (Backend, backend_model_name), split by kind so /v1/chat/completions
    # and /v1/embeddings each see only their own aliases.
    _chat_map: dict[str, tuple[Backend, str]] = field(default_factory=dict, repr=False)
    _embed_map: dict[str, tuple[Backend, str]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        for b in self.backends:
            target = self._embed_map if b.kind == "embedding" else self._chat_map
            for alias, backend_name in b.models.items():
                target[alias] = (b, backend_name)

    def get_backend(self, model: str) -> tuple[Backend, str] | None:
        return self._chat_map.get(model)

    def get_embedding_backend(self, model: str) -> tuple[Backend, str] | None:
        return self._embed_map.get(model)

    def all_models(self) -> list[str]:
        return list(self._chat_map.keys())

    def all_embedding_models(self) -> list[str]:
        return list(self._embed_map.keys())

    def overrides_for(self, backend_model: str) -> dict:
        """Look up request_overrides for a backend model (gguf basename)."""
        return self.overrides_by_basename.get(backend_model, {})


def _load_overrides(path: Path) -> dict[str, dict]:
    """Read /etc/llama-home/model-profiles.json и собрать basename → overrides.

    Структура profiles: {"models": {<rel_path>: {..., "request_overrides": {...}}}}.
    Нас интересует только request_overrides; ключ — basename gguf-файла,
    потому что router оперирует backend_model именно как basename из
    config.yaml.
    """
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    out: dict[str, dict] = {}
    for rel_path, entry in (raw.get("models") or {}).items():
        ov = entry.get("request_overrides") or {}
        if not ov:
            continue
        basename = rel_path.rsplit("/", 1)[-1]
        out[basename] = ov
    return out


def load_config(path: str | Path = "config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    backends = [Backend(**b) for b in raw["backends"]]
    return Config(
        backends=backends,
        overrides_by_basename=_load_overrides(PROFILES_PATH),
    )
