"""DeepSeek backend for HingeAuto judging.

Sends profile screenshots to DeepSeek's OpenAI-compatible chat completions
endpoint (https://api.deepseek.com) with a forced tool call to extract a
structured Decision. Same interface as `judge.py`.

Usage: set JUDGE_BACKEND = "deepseek" in config.py, add DEEPSEEK_API_KEY
to .env.

Two knobs (see config.py):
  DEEPSEEK_MODEL      — "deepseek-flash" (vision, default)
  DEEPSEEK_THINKING   — False (default) | True

Thinking mode and forced tool choice are mutually exclusive on this API:
`tool_choice` naming a function (or "required") returns a 400 while
thinking is on. So the two settings move together:

  DEEPSEEK_THINKING=False (default) — thinking disabled, `tool_choice`
    forces `submit_decision`, mirroring the Anthropic backend. Cheap,
    fast, and the model always answers in the schema.

  DEEPSEEK_THINKING=True — thinking enabled at DEEPSEEK_REASONING_EFFORT,
    `tool_choice="auto"`. Better reasoning on ambiguous profiles, but the
    model may answer in prose instead of calling the tool, so the
    content-JSON fallback below does more of the work.

Cost is well below the Anthropic and Gemini backends either way —
see MODEL_PRICING in metrics.py.
"""

import base64
import json
import os

import httpx
from dotenv import load_dotenv

import config
from judge_common import (
    DECIDE_INPUT_SCHEMA,
    Decision,
    build_system_prompt,
    decision_from_tool_args,
    enforce_premade_verbatim,
)


API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-flash"
# Non-thinking output is a small JSON blob; thinking tokens (reasoning_content)
# are billed as output too, so thinking mode needs a much larger budget.
MAX_TOKENS_PLAIN = 2000
MAX_TOKENS_THINKING = 8000
REQUEST_TIMEOUT_S = 180.0


def _tool_spec() -> dict:
    """OpenAI-compatible tool envelope wrapping the shared schema."""
    return {
        "type": "function",
        "function": {
            "name": "submit_decision",
            "description": "Submit a fit-score assessment for this Hinge profile.",
            "parameters": DECIDE_INPUT_SCHEMA,
        },
    }


def _image_block(png_bytes: bytes) -> dict:
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,"
                   + base64.standard_b64encode(png_bytes).decode("utf-8"),
        },
    }


def _request_body(model: str, frames: list[bytes], thinking: bool) -> dict:
    content = [_image_block(f) for f in frames]
    content.append({
        "type": "text",
        "text": (
            f"Above are {len(frames)} screenshots of one Hinge profile, in order "
            "from top to bottom. Score how well it fits, then call the "
            "submit_decision tool with the structured result."
        ),
    })

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": content},
        ],
        "tools": [_tool_spec()],
        "max_tokens": MAX_TOKENS_THINKING if thinking else MAX_TOKENS_PLAIN,
        "stream": False,
    }
    if thinking:
        body["thinking"] = {"type": "enabled"}
        body["reasoning_effort"] = getattr(config, "DEEPSEEK_REASONING_EFFORT", "low")
        # Can't force a tool in thinking mode (400) — fall back to auto.
        body["tool_choice"] = "auto"
    else:
        body["thinking"] = {"type": "disabled"}
        body["tool_choice"] = {
            "type": "function",
            "function": {"name": "submit_decision"},
        }
        body["temperature"] = 0.2
    return body


def _usage_from(payload: dict) -> dict:
    raw = payload.get("usage") or {}
    return {
        "input_tokens": raw.get("prompt_tokens", 0) or 0,
        "output_tokens": raw.get("completion_tokens", 0) or 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": raw.get("prompt_cache_hit_tokens", 0) or 0,
    }


def _json_args(raw) -> dict:
    """Tool arguments arrive as a JSON string (occasionally already a dict)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def judge(frames: list[bytes]) -> Decision:
    """Given an ordered list of PNG frames of one profile, return a Decision."""
    load_dotenv()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set. Add it to .env or export it.")

    model = getattr(config, "DEEPSEEK_MODEL", DEFAULT_MODEL)
    thinking = bool(getattr(config, "DEEPSEEK_THINKING", False))
    body = _request_body(model, frames, thinking)

    response = httpx.post(
        API_URL,
        json=body,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=REQUEST_TIMEOUT_S,
    )
    if response.status_code != 200:
        # 400s here are usually one of the thinking/tool_choice conflicts or a
        # context overrun — surface the API's own message rather than guessing.
        err = RuntimeError(
            f"DeepSeek API error {response.status_code}: {response.text[:500]}"
        )
        # Attach the status so the harness can tell a fatal error (bad key,
        # empty balance) from a retryable one without parsing the text.
        # See judge_common.is_fatal_judge_error.
        err.status_code = response.status_code
        raise err
    payload = response.json()
    usage = _usage_from(payload)

    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"DeepSeek ({model}) returned no choices: {payload}")
    choice = choices[0]
    message = choice.get("message") or {}

    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        if fn.get("name") != "submit_decision":
            continue
        decision = decision_from_tool_args(_json_args(fn.get("arguments")), usage)
        enforce_premade_verbatim(decision)
        return decision

    # Fallback: with thinking on (tool_choice="auto") the model sometimes
    # answers in prose. Pull the first JSON object out of the content.
    content = message.get("content") or ""
    start, end = content.find("{"), content.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            decision = decision_from_tool_args(data, usage)
            enforce_premade_verbatim(decision)
            return decision

    raise RuntimeError(
        f"DeepSeek ({model}) did not return a usable submit_decision call. "
        f"finish_reason={choice.get('finish_reason')!r}. If this repeats with "
        f"DEEPSEEK_THINKING=True, set it back to False (forced tool choice)."
    )
