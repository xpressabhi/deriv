"""Configuration loading and environment overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

from .io_utils import read_json, sha256_json


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"

API_KEY_ENV_NAMES = ("OPENCODE_API_KEY", "OPENCODE_GO_API_KEY", "OPENCODE_ZEN_API_KEY")


def _dotenv_values() -> dict[str, str]:
    """Read a local, gitignored .env file as a convenience for development."""
    if not DOTENV_PATH.exists():
        return {}
    values: dict[str, str] = {}
    try:
        for raw_line in DOTENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        return {}
    return values


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    config = copy.deepcopy(read_json(config_path))
    generation = config.setdefault("generation", {})

    if os.environ.get("SUPPORT_MODEL"):
        generation["model"] = os.environ["SUPPORT_MODEL"].strip()
    if os.environ.get("OPENCODE_GO_BASE_URL"):
        generation["base_url"] = os.environ["OPENCODE_GO_BASE_URL"].strip()
    required = _env_bool("SUPPORT_LLM_REQUIRED")
    if required is not None:
        generation["llm_required"] = required

    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    retrieval = config.get("retrieval", {})
    if int(retrieval.get("top_k", 0)) < 1:
        raise ConfigError("retrieval.top_k must be at least 1")
    if float(retrieval.get("bm25_k1", 0)) <= 0:
        raise ConfigError("retrieval.bm25_k1 must be positive")
    if not 0 <= float(retrieval.get("bm25_b", -1)) <= 1:
        raise ConfigError("retrieval.bm25_b must be between 0 and 1")

    chunking = config.get("chunking", {})
    if int(chunking.get("max_chars", 0)) < 80:
        raise ConfigError("chunking.max_chars must be at least 80")
    if int(chunking.get("overlap_sentences", -1)) < 0:
        raise ConfigError("chunking.overlap_sentences must not be negative")

    gate = config.get("gate", {})
    if not gate.get("categories"):
        raise ConfigError("gate.categories must be configured")

    verification = config.get("verification", {})
    if not verification.get("prohibited_patterns"):
        raise ConfigError("verification.prohibited_patterns must be configured")
    if not 0 <= float(verification.get("min_sentence_support", -1)) <= 1:
        raise ConfigError("verification.min_sentence_support must be between 0 and 1")


def get_api_key() -> str | None:
    dotenv = _dotenv_values()
    for name in API_KEY_ENV_NAMES:
        value = os.environ.get(name) or dotenv.get(name)
        if value and value.strip():
            return value.strip()
    return None


def api_key_source() -> str | None:
    dotenv = _dotenv_values()
    for name in API_KEY_ENV_NAMES:
        value = os.environ.get(name) or dotenv.get(name)
        if value and value.strip():
            return name + ("" if os.environ.get(name) else " (.env)")
    return None


def api_key_source_kind() -> str | None:
    dotenv = _dotenv_values()
    for name in API_KEY_ENV_NAMES:
        if os.environ.get(name):
            return "environment"
        if dotenv.get(name):
            return "dotenv"
    return None


def retrieval_config_hash(config: dict[str, Any]) -> str:
    retrieval = config.get("retrieval", {})
    chunking = config.get("chunking", {})
    return sha256_json({"retrieval": retrieval, "chunking": chunking})
