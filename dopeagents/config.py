"""Global configuration and settings."""

from __future__ import annotations

import threading

from pydantic import Field
from pydantic_settings import BaseSettings


class DopeAgentsConfig(BaseSettings):
    """Global configuration for DopeAgents via environment variables.

    Reads from DOPEAGENTS_* environment variables and optional TOML file.

    Usage:
        config = DopeAgentsConfig.from_env()
        DEFAULT_MODEL = config.default_model

    Or using the global singleton:
        config = get_config()
        set_config(custom_config)
    """

    # ── Provider model defaults ──────────────────────────────────
    # These live in config (not .env) — they're defaults, not secrets.
    # Override any of them via DOPEAGENTS_<FIELD> env var if needed.

    groq_default_model: str = "groq/llama-3.3-70b-versatile"
    """Default model for Groq provider."""

    openrouter_default_model: str = "openrouter/meta-llama/llama-3.3-70b-instruct:free"
    """Default model for OpenRouter provider."""

    together_default_model: str = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
    """Default model for Together AI provider."""

    # ── Provider API keys (read from bare env vars, not DOPEAGENTS_ prefixed) ──
    # Just uncomment the matching key in .env — config auto-detects provider.

    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    together_api_key: str | None = Field(default=None, alias="TOGETHER_API_KEY")

    # ── Active model & API ────────────────────────────────────────

    default_model: str | None = None
    """Explicit model override. If None, auto-detected from active API key."""

    api_key: str | None = None
    """LLM API key (usually read from provider-specific env var instead)."""

    api_base: str | None = None
    """Optional custom API base URL for provider routing."""

    # ── Cost management ───────────────────────────────────────────

    enable_cost_tracking: bool = True
    """Whether to track step-level costs."""

    max_cost_per_call: float | None = None
    """Max cost in USD per agent.run() call."""

    max_cost_per_step: float | None = None
    """Max cost in USD per individual step."""

    max_cost_global: float | None = None
    """Max total cost in USD across all agents in this session."""

    default_cost_exceeded_action: str = "error"
    """What to do when cost limit exceeded: 'error' or 'degrade'."""

    # ── Observability ─────────────────────────────────────────────

    enable_step_metrics: bool = True
    """Whether to collect per-step cost and latency metrics."""

    tracer_type: str = "console"
    """Which tracer to use: 'console', 'otel', or 'noop'."""

    otel_endpoint: str | None = None
    """OpenTelemetry collector endpoint (if tracer_type='otel')."""

    log_prompts: bool = False
    """Whether to log full prompts (verbose; impacts PII)."""

    log_color: bool = True
    """Whether to use colored output in logs (ANSI codes)."""

    log_level: str = "INFO"
    """Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL."""

    # ── Retry & Resilience ────────────────────────────────────────

    enable_retry: bool = True
    """Whether to retry transient errors."""

    max_retries: int = 3
    """Max retries per step for transient errors."""

    retry_base_delay_seconds: float = 1.0
    """Base delay for exponential backoff."""

    # ── Caching ───────────────────────────────────────────────────

    enable_cache: bool = False
    """Whether to cache step outputs for repeated inputs."""

    cache_backend: str = "memory"
    """Cache implementation: 'memory' or 'disk'."""

    cache_dir: str | None = None
    """Directory for disk cache (if cache_backend='disk')."""

    # ── Security ──────────────────────────────────────────────────

    enable_pii_redaction: bool = True
    """Whether to redact PII in observability output."""

    redaction_patterns: list[str] = []
    """Custom regex patterns for PII detection."""

    # ── Sandbox ───────────────────────────────────────────────────

    sandbox_enabled: bool = False
    """Whether to run agents in sandbox (if available)."""

    sandbox_timeout_seconds: float = 300.0
    """Sandbox execution timeout."""

    # ── Development & Debug ───────────────────────────────────────

    debug_mode: bool = False
    """Enable debug logging and detailed error messages."""

    strict_type_checking: bool = True
    """Enforce strict type checking on agent I/O."""

    class Config:
        """Pydantic settings config."""

        env_prefix = "DOPEAGENTS_"
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"
        populate_by_name = True

    @classmethod
    def from_env(cls) -> DopeAgentsConfig:
        """Load configuration from environment variables."""
        return cls()

    def resolve_model(self) -> str:
        """Return the active model, auto-detecting from whichever API key is set.

        Priority:
          1. DOPEAGENTS_DEFAULT_MODEL (explicit override in .env)
          2. GROQ_API_KEY present  → groq_default_model
          3. OPENROUTER_API_KEY present → openrouter_default_model
          4. TOGETHER_API_KEY present  → together_default_model
          5. groq_default_model (hardcoded fallback)
        """
        if self.default_model:
            return self.default_model
        if self.groq_api_key:
            return self.groq_default_model
        if self.openrouter_api_key:
            return self.openrouter_default_model
        if self.together_api_key:
            return self.together_default_model
        return self.groq_default_model


# ── Global singleton management ────────────────────────────────────

_config_lock = threading.Lock()
_default_config: DopeAgentsConfig | None = None


def get_config() -> DopeAgentsConfig:
    """Get the global DopeAgents configuration.

    Thread-safe. Returns cached singleton or creates new one.
    """
    global _default_config

    if _default_config is None:
        with _config_lock:
            if _default_config is None:
                _default_config = DopeAgentsConfig.from_env()

    return _default_config


def set_config(config: DopeAgentsConfig) -> None:
    """Set the global DopeAgents configuration.

    Thread-safe. All subsequent get_config() calls return this config.
    """
    global _default_config

    with _config_lock:
        _default_config = config


def reset_config() -> None:
    """Reset configuration to defaults.

    Used in testing to start fresh.
    """
    global _default_config

    with _config_lock:
        _default_config = None
