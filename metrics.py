"""Per-profile structured logging for the autopilot loop.

Each profile run produces one JSONL line in `debug/session_log.jsonl`
with timing, token usage, decision, and message metadata. Append-only
and machine-parseable so chart-making downstream is trivial.
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import config


# Pricing per 1M tokens (USD). Resolved at import time based on the
# active backend and model so cost estimates are always accurate even
# when overridden via .env.
MODEL_PRICING: dict[str, dict[str, float]] = {
    # ---------- Anthropic ----------
    "claude-sonnet-4-6": {
        "input_tokens":     3.00,
        "output_tokens":   15.00,
    },
    "claude-haiku-4-5": {
        "input_tokens":     0.80,
        "output_tokens":    4.00,
    },
    "claude-opus-4-7": {
        "input_tokens":    15.00,
        "output_tokens":   75.00,
    },
    # ---------- Gemini ----------
    "gemini-3.5-flash": {
        "input_tokens":     1.50,
        "output_tokens":    9.00,
    },
    "gemini-3.6-flash": {
        "input_tokens":     1.50,
        "output_tokens":    7.50,
    },
    "gemini-3.1-flash-lite": {
        "input_tokens":     0.25,
        "output_tokens":    1.50,
    },
    "gemini-3-flash-preview": {
        "input_tokens":     0.30,
        "output_tokens":    1.00,
    },
    # ---------- Ollama ----------
    "qwen2.5-vl": {
        "input_tokens":     0.0,   # free / local
        "output_tokens":    0.0,
    },
    "llama3.2-vision": {
        "input_tokens":     0.0,
        "output_tokens":    0.0,
    },
}


def _resolve_model_name() -> str:
    """Return the active model name string based on backend config."""
    backend = getattr(config, "JUDGE_BACKEND", "anthropic").lower()
    if backend == "anthropic":
        return getattr(config, "MODEL", "claude-sonnet-4-6")
    if backend == "gemini":
        return getattr(config, "GEMINI_MODEL", "gemini-3.1-flash-lite")
    if backend == "ollama":
        return getattr(config, "OLLAMA_MODEL", "qwen2.5-vl")
    return "unknown"


_ACTIVE_MODEL = _resolve_model_name()
_PRICING = MODEL_PRICING.get(_ACTIVE_MODEL)
if _PRICING is None:
    print(f"[metrics] WARN: unknown model {_ACTIVE_MODEL!r}, cost will be $0 "
          f"— add pricing to metrics.py")
    _PRICING = {"input_tokens": 0.0, "output_tokens": 0.0}


def estimated_cost(usage: dict[str, int]) -> float:
    """Dollar estimate from a usage dict returned by judge()."""
    if not usage:
        return 0.0
    return sum(
        usage.get(k, 0) * price / 1_000_000
        for k, price in _PRICING.items()
    )


def log_profile(
    profile_idx: int,
    decision,
    timing: dict[str, float],
    log_path: Path | None = None,
) -> None:
    """Append a JSONL record for one profile."""
    if log_path is None:
        log_path = config.DEBUG_DIR / "session_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "profile_idx": profile_idx,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": config.MODE_NAME,
        "model": _ACTIVE_MODEL,
        "name": decision.name,
        "decision": decision.decision,
        "confidence": decision.confidence,
        "reasoning": decision.reasoning,
        "message": decision.message,
        "message_length": len(decision.message),
        "message_archetype": decision.message_archetype,
        "premade_id": decision.premade_id,
        "prompt_referenced": decision.prompt_referenced,
        "skip_reason": decision.skip_reason,
        "timing": timing,
        "tokens": decision.usage,
        "estimated_cost_usd": round(estimated_cost(decision.usage), 5),
    }

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_running_totals(
    profiles_seen: int,
    likes_sent: int,
    skips: int,
    total_cost: float,
    total_seconds: float,
) -> None:
    """One-line summary printed every loop iteration."""
    avg_cost = total_cost / profiles_seen if profiles_seen else 0
    avg_time = total_seconds / profiles_seen if profiles_seen else 0
    like_rate = likes_sent / profiles_seen if profiles_seen else 0
    model_tag = _ACTIVE_MODEL or "?"
    print(
        f"[totals] {profiles_seen} profiles | {likes_sent} likes "
        f"({like_rate:.0%}) | {skips} skips | "
        f"${total_cost:.3f} (~${avg_cost:.4f}/profile) | "
        f"avg {avg_time:.1f}s/profile | {model_tag}"
    )
