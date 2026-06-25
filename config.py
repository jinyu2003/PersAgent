"""Configuration helpers for PersAgent.

The demo defaults to deterministic local knowledge-base tools. Model names and
API keys are still exposed so the agent classes can be upgraded to live LLM
calls without changing their public interfaces.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def _load_dotenv(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs without adding a runtime dependency."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    brain_model: str
    knowledge_model: str
    verifier_model: str
    api_key: str | None
    base_url: str | None
    max_tokens: int
    attribution_parallelism: int
    use_live_llm: bool


PROVIDER_DEFAULTS = {
    "deepseek": {
        "brain_model": "deepseek-v4-flash",
        "knowledge_model": "deepseek-v4-flash",
        "verifier_model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "openai": {
        "brain_model": "gpt-4o",
        "knowledge_model": "gpt-4o-mini",
        "verifier_model": "gpt-4o-mini",
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
    },
}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in TRUE_VALUES


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        return default


def _provider_api_key(provider: str) -> str | None:
    if os.getenv("LLM_API_KEY"):
        return os.getenv("LLM_API_KEY")
    default_env = PROVIDER_DEFAULTS.get(provider, {}).get("api_key_env")
    if default_env and os.getenv(default_env):
        return os.getenv(default_env)
    if provider == "deepseek":
        return os.getenv("DEEPSEEK_API_KEY")
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    return os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")


def _provider_base_url(provider: str) -> str | None:
    if os.getenv("LLM_BASE_URL"):
        return os.getenv("LLM_BASE_URL")
    if provider == "deepseek" and os.getenv("DEEPSEEK_BASE_URL"):
        return os.getenv("DEEPSEEK_BASE_URL")
    if provider == "openai" and os.getenv("OPENAI_BASE_URL"):
        return os.getenv("OPENAI_BASE_URL")
    return PROVIDER_DEFAULTS.get(provider, {}).get("base_url")


def get_model_config() -> ModelConfig:
    _load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
    defaults = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["deepseek"])
    model = os.getenv("LLM_MODEL")

    return ModelConfig(
        provider=provider,
        brain_model=os.getenv("BRAIN_MODEL") or model or defaults["brain_model"],
        knowledge_model=os.getenv("KNOWLEDGE_MODEL") or model or defaults["knowledge_model"],
        verifier_model=os.getenv("VERIFIER_MODEL") or model or defaults["verifier_model"],
        api_key=_provider_api_key(provider),
        base_url=_provider_base_url(provider),
        max_tokens=_env_int("LLM_MAX_TOKENS", 2048),
        attribution_parallelism=max(1, _env_int("LLM_ATTRIBUTION_PARALLELISM", 2)),
        use_live_llm=_env_flag("PERSAGENT_USE_LIVE_LLM"),
    )
