"""
List free models available on each configured provider using their REST APIs.

Usage:
    python scripts/check_free_models.py

Calls each provider's /models endpoint directly (no LiteLLM) and filters
for free-tier models. Only runs for providers whose API key is set in .env.

Provider endpoints:
  Groq        — GET https://api.groq.com/openai/v1/models       (all models are free-tier)
  OpenRouter  — GET https://openrouter.ai/api/v1/models          (no auth; filter price==0)
  Together AI — GET https://api.together.xyz/v1/models           (filter input_price==0)
"""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from dopeagents.config import get_config


def _get(url: str, api_key: str | None = None, timeout: int = 15) -> list[dict[str, Any]] | None:
    """GET url, return parsed JSON list or None on error."""
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = httpx.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # most providers wrap in {"data": [...]}
        return data.get("data", data) if isinstance(data, dict) else data  # type: ignore[no-any-return]
    except httpx.HTTPStatusError as e:
        print(f"      HTTP {e.response.status_code}: {e.response.text[:120]}")
        return None
    except Exception as e:
        print(f"      Error: {e}")
        return None


# ── Groq ─────────────────────────────────────────────────────────────────────
# All Groq models are free-tier; endpoint requires auth.
def list_groq(api_key: str) -> None:
    masked = api_key[:8] + "..." + api_key[-4:]
    print(f"\n  🔑  GROQ ({masked})")
    models = _get("https://api.groq.com/openai/v1/models", api_key)
    if models is None:
        return
    chat_models = [m for m in models if m.get("object") == "model"]
    print(f"      {len(chat_models)} models available (all free-tier on Groq):\n")
    for m in sorted(chat_models, key=lambda x: x.get("id", "")):
        print(f"        • {m['id']}")


# ── OpenRouter ────────────────────────────────────────────────────────────────
# Public endpoint — no auth needed. Free models have pricing.prompt == "0".
def list_openrouter(api_key: str | None) -> None:
    label = (api_key[:8] + "..." + api_key[-4:]) if api_key else "no key (public endpoint)"
    print(f"\n  🔑  OPENROUTER ({label})")
    models = _get("https://openrouter.ai/api/v1/models")  # no auth needed
    if models is None:
        return
    free = [m for m in models if str(m.get("pricing", {}).get("prompt", "1")) == "0"]
    print(f"      {len(free)} free models (prompt price = $0):\n")
    for m in sorted(free, key=lambda x: x.get("id", "")):
        ctx = m.get("context_length", "?")
        print(f"        • {m['id']:<60}  ctx={ctx}")


# ── Together AI ───────────────────────────────────────────────────────────────
# Endpoint requires auth. Free models have pricing.input == 0.
def list_together(api_key: str) -> None:
    masked = api_key[:8] + "..." + api_key[-4:]
    print(f"\n  🔑  TOGETHER AI ({masked})")
    models = _get("https://api.together.xyz/v1/models", api_key)
    if models is None:
        return
    free = [m for m in models if float(m.get("pricing", {}).get("input", 1) or 1) == 0]
    all_count = len(models)
    print(f"      {len(free)} free models out of {all_count} total:\n")
    for m in sorted(free, key=lambda x: x.get("id", "")):
        mtype = m.get("type", "")
        print(f"        • {m['id']:<60}  [{mtype}]")
    if not free:
        # fallback: show all and note pricing unknown
        print("      (no models with input_price=0 found — showing all chat models)")
        chat = [m for m in models if "chat" in m.get("type", "").lower()][:15]
        for m in chat:
            price = m.get("pricing", {}).get("input", "?")
            print(f"        • {m['id']:<60}  input=${price}/M")


def main() -> None:
    config = get_config()

    print("=" * 70)
    print("  DopeAgents — Free Model Listing (live REST API calls)")
    print("=" * 70)
    print(f"  Active model (resolve_model): {config.resolve_model()}")

    if config.groq_api_key:
        list_groq(config.groq_api_key)
    else:
        print("\n  ⚪  GROQ — GROQ_API_KEY not set, skipping")

    list_openrouter(config.openrouter_api_key)  # public endpoint, always runs

    if config.together_api_key:
        list_together(config.together_api_key)
    else:
        print("\n  ⚪  TOGETHER AI — TOGETHER_API_KEY not set, skipping")

    print("\n" + "=" * 70)
    print("  Tip: uncomment the key you want in .env to switch active provider.")
    print("=" * 70)


if __name__ == "__main__":
    main()
