"""Backend-agnostic pieces of the judging pipeline.

Both `judge.py` (Anthropic) and `judge_ollama.py` (Ollama) import from
here so the system prompt, decision shape, and tool schema stay in sync.

A backend module needs to expose `judge(frames: list[bytes]) -> Decision`.
"""

from dataclasses import dataclass, field
from typing import Any

import config


SYSTEM_PROMPT_TEMPLATE = """You are evaluating Hinge dating profiles on behalf of the user.

The user's preferences:
{preferences}
{age_clause}
You will be shown a sequence of screenshots representing a single profile, in
order from top to bottom. The profile may include photos, prompt responses
(short text), and basic info (age, height, location, job, education, etc.).

{fit_clause}
IMPORTANT: If any dialog, popup, overlay, or non-profile screen is present
that covers any part of the profile and prevents you from analyzing it fully —
including settings panels, subscription upsells, notification prompts, rate-the-app
nags, or any other interruption — set decision="NOT_A_PROFILE". Use reasoning
to describe what you're seeing (e.g. "a subscription upsell dialog is blocking
the profile"). Only set decision="profile" when you can see the full profile.

{message_voice}
{premades_section}
Submit your decision via the submit_decision tool."""


# Generic, voice-neutral fallback used when the active mode does not set
# MESSAGE_VOICE. Replace by writing a voice file under voice/<name>.py
# and pointing your mode at it.
DEFAULT_MESSAGE_VOICE = """## Message rubric (when the profile is a pass)

Write a short opener that goes out with the like — but ONLY when the
profile passes (fit_score >= FIT_SCORE_MIN). Empty string otherwise.

Aim for: one specific reference to something visible in the profile (a
prompt answer or a concrete photo detail), followed by a short question
about it. Keep it friendly and curious. Around 60-120 characters total.

Constraints:
- Plain ASCII only. No emoji, no smart quotes, no em-dashes.
- Avoid the characters \\, ", $, ` — they break the typing layer.
- Empty string when fit_score is below the threshold (won't be liked).
- If the profile passes but you cannot write a specific, non-generic
  opener, output the empty string for `message` and set
  `message_archetype` to "empty". A like with no message is acceptable.

This is the GENERIC fallback voice. Most users will want to override it
by setting `MESSAGE_VOICE` in their mode file (or pointing it at a
template under voice/). See voice/example_casual.py and
voice/example_polished.py for two contrasting starting points."""


# JSON schema for the submit_decision tool. Backends wrap this in their
# own tool-spec envelope (Anthropic uses `input_schema`, Ollama uses
# OpenAI-compatible `parameters`).
DECIDE_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": (
                "The profile's first name as shown at the top of the "
                "profile. Lowercase, ASCII letters only — strip "
                "spaces, punctuation, emoji. If not visible, use "
                "\"unknown\"."
            ),
        },
        "decision": {
            "type": "string",
            "enum": ["profile", "NOT_A_PROFILE"],
            "description": "'profile' when the screenshots show a real profile you can score. 'NOT_A_PROFILE' when a dialog, popup, overlay, or non-profile screen blocks full analysis (use reasoning to describe it). Do NOT choose like vs skip — the harness decides that from fit_score.",
        },
        "fit_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "How well this profile fits the user's preferences, 0-100. Use the full range: 90+ = exact match, 75-89 = strong fit, 60-74 = decent, 40-59 = neutral, 0-39 = not a fit. This is the authoritative score the run uses to decide like vs skip.",
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "How confident you are in the decision.",
        },
        "reasoning": {
            "type": "string",
            "description": (
                "One or two sentences explaining the decision. Reference "
                "specific details from the profile (e.g., a prompt answer, "
                "an activity in a photo, the bio/info line)."
            ),
        },
        "message": {
            "type": "string",
            "description": (
                "The opener to send with the like — write it only when the "
                "profile is a pass (fit_score >= FIT_SCORE_MIN); empty string "
                "otherwise. Max ~150 chars, plain ASCII, no emoji. See the "
                "message rubric in the system prompt."
            ),
        },
        "skip_reason": {
            "type": "string",
            "enum": ["none", "age", "preferences", "low_effort", "other"],
            "description": (
                "Categorical reason a profile scored poorly / was skipped, "
                "for downstream analytics. \"age\" when the AGE GATE fired, "
                "\"preferences\" when a specific PREFERENCES rule fired, "
                "\"low_effort\" when the profile was too thin to engage with "
                "(no readable prompts, single photo, etc.). \"none\" when not "
                "applicable (you don't know the harness's threshold), \"other\" "
                "only if nothing else fits."
            ),
        },
        "message_archetype": {
            "type": "string",
            "enum": [
                "empty",
                "observation_question",
                "prompt_callback",
                "photo_callback",
                "tease",
                "premade",
                "other",
            ],
            "description": (
                "Categorical label for the message style. "
                "\"observation_question\" = specific detail + a question. "
                "\"prompt_callback\" = references a prompt with no question. "
                "\"photo_callback\" = references a photo detail with no "
                "question. \"tease\" = mild playful disagreement. "
                "\"premade\" when the message is a verbatim copy of one of "
                "the mode's premade openers (set premade_id too). "
                "\"empty\" when the message is empty (skip OR a like with "
                "no opener). \"other\" only when nothing else fits."
            ),
        },
        "premade_id": {
            "type": "string",
            "description": (
                "Id of the premade opener used, when message_archetype == "
                "\"premade\". Must match one of the ids listed in the "
                "Premade openers section of the system prompt. Empty "
                "string when the message was written fresh or is empty."
            ),
        },
        "prompt_referenced": {
            "type": "string",
            "description": (
                "Short label (under 50 chars) of the prompt or photo "
                "detail the message references, e.g. \"travel prompt\", "
                "\"sunset photo 1\". Empty string when message is empty "
                "or when using a premade that doesn't reference a specific "
                "profile detail."
            ),
        },
    },
    "required": [
        "name", "decision", "fit_score", "confidence", "reasoning",
        "message", "skip_reason", "message_archetype", "premade_id",
        "prompt_referenced",
    ],
}


@dataclass
class Decision:
    name: str
    decision: str  # "like" | "skip" | "NOT_A_PROFILE"
    confidence: str  # "low" | "medium" | "high"
    reasoning: str
    message: str = ""
    fit_score: int = 0  # 0-100, authoritative for like/skip gating
    skip_reason: str = "none"
    message_archetype: str = "empty"
    premade_id: str = ""
    prompt_referenced: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


def resolve_voice(voice: str | None) -> str:
    """Resolve the active mode's MESSAGE_VOICE into a prompt string.

    Three shapes accepted:
      - None        -> use DEFAULT_MESSAGE_VOICE
      - "<name>"    -> single-token name; load voice/<name>.py and read
                       its MESSAGE_VOICE module attribute
      - "...\\n..." -> multi-line literal; passed through as-is
    """
    if voice is None:
        return DEFAULT_MESSAGE_VOICE
    stripped = voice.strip()
    is_name = (
        stripped
        and "\n" not in stripped
        and " " not in stripped
        and len(stripped) <= 64
        and all(c.isalnum() or c in "_-" for c in stripped)
    )
    if is_name:
        try:
            import importlib
            mod = importlib.import_module(f"voice.{stripped}")
        except ModuleNotFoundError as e:
            raise RuntimeError(
                f"MESSAGE_VOICE={stripped!r} but voice/{stripped}.py not found. "
                f"Create it (see voice/example_casual.py) or use a multi-line "
                f"string for MESSAGE_VOICE instead."
            ) from e
        msg = getattr(mod, "MESSAGE_VOICE", None)
        if not isinstance(msg, str) or not msg.strip():
            raise RuntimeError(
                f"voice/{stripped}.py must export a non-empty MESSAGE_VOICE string."
            )
        return msg
    return voice


def _premades_section(premades: list[dict]) -> str:
    if not premades:
        return ""
    lines = [
        "",
        "## Premade openers (verbatim, mode-specific)",
        "",
        "Instead of writing a fresh opener, you may select one of the following "
        "pre-written messages. Each has explicit guidance for when to use it. "
        "Premades bypass the voice rules above — they are sent EXACTLY as "
        "written, including capitalization and punctuation. If you select a "
        "premade:",
        '  - set `premade_id` to its id',
        '  - set `message` to the premade\'s text VERBATIM (copy character-for-character)',
        '  - set `message_archetype` to "premade"',
        "",
        "If no premade fits the profile, write a fresh opener per the voice "
        "rules above and leave `premade_id` as an empty string.",
        "",
        "Available premades:",
        "",
    ]
    for i, p in enumerate(premades, 1):
        lines.append(f'{i}. id: "{p["id"]}"')
        lines.append(f'   message (verbatim): {p["message"]!r}')
        lines.append(f'   use_when: {p["use_when"]}')
        lines.append("")
    return "\n".join(lines)


def _fit_clause(fit_score_min: int) -> str:
    return (
        f"\nFIT SCORE: assign a single integer fit_score (0-100) for how well "
        f"this profile matches the user's preferences. Use the full range — don't "
        f"cluster around the middle. 90+ = exact match, 75-89 = strong fit, "
        f"60-74 = decent, 40-59 = neutral, 0-39 = not a fit. Only set "
        f"decision=\"profile\" (or \"NOT_A_PROFILE\" for an overlay). You do NOT "
        f"choose like vs skip — the run's harness does, like iff fit_score >= "
        f"{fit_score_min}. Base the score on preferences, profile info, prompt "
        f"answers, and photos. When genuinely ambiguous, lean toward a lower "
        f"score.\n"
    )


def _age_clause(age_min: int | None, age_max: int | None) -> str:
    if age_min is None and age_max is None:
        return ""
    lo = age_min if age_min is not None else 18
    hi = age_max if age_max is not None else 99
    return (
        f"\nAGE GATE: only LIKE if the profile's stated age is between "
        f"{lo} and {hi} inclusive. Hinge shows age in basic-info "
        f"(\"NN\" next to height/location). If age is visible and out of "
        f"range, decision=skip, skip_reason=\"age\", message=\"\". If age "
        f"genuinely isn't visible across any frame, proceed with the normal "
        f"rubric.\n"
    )


def build_system_prompt() -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        preferences=config.PREFERENCES.strip(),
        age_clause=_age_clause(config.AGE_MIN, config.AGE_MAX),
        fit_clause=_fit_clause(getattr(config, "FIT_SCORE_MIN", 50)),
        message_voice=resolve_voice(config.MESSAGE_VOICE).strip(),
        premades_section=_premades_section(config.PREMADES),
    )


def enforce_premade_verbatim(decision: Decision) -> None:
    """If the model picked a premade by id, overwrite `message` with the
    canonical text so character drift can't leak. Unknown ids are
    cleared so the message goes out as-written."""
    if not decision.premade_id:
        return
    for p in config.PREMADES:
        if p["id"] == decision.premade_id:
            decision.message = p["message"]
            decision.message_archetype = "premade"
            return
    print(
        f"[judge] warning: unknown premade_id={decision.premade_id!r}, "
        f"clearing and treating message as fresh"
    )
    decision.premade_id = ""


def apply_fit_threshold(decision: Decision) -> Decision:
    """Gate the like/skip action entirely on fit_score.

    The model's `decision` field is advisory; this re-derives it from
    config.FIT_SCORE_MIN so pickiness is a configurable dial rather than
    prompt prose. NOT_A_PROFILE is preserved — dialog/popup recovery relies
    on it. Clamps fit_score to 0-100 and mutates + returns `decision`.
    """
    if decision.decision == "NOT_A_PROFILE":
        return decision
    try:
        score = max(0, min(100, int(decision.fit_score)))
    except (TypeError, ValueError):
        print(f"[judge] WARN: invalid fit_score={decision.fit_score!r}, defaulting to 0")
        score = 0
    decision.fit_score = score
    decision.decision = "like" if score >= config.FIT_SCORE_MIN else "skip"
    if decision.decision == "skip":
        # An opener must never go out on a skip (e.g. a high-scoring profile
        # the model misjudged, or a low-scoring one it still drafted for).
        decision.message = ""
        decision.message_archetype = "empty"
        decision.premade_id = ""
    return decision


def load_backend():
    """Resolve config.JUDGE_BACKEND to a module exposing judge(frames)."""
    backend = getattr(config, "JUDGE_BACKEND", "anthropic").lower()
    if backend == "anthropic":
        import judge
        return judge
    if backend == "ollama":
        import judge_ollama
        return judge_ollama
    if backend == "gemini":
        import judge_gemini
        return judge_gemini
    raise ValueError(
        f"Unknown JUDGE_BACKEND={backend!r}. Use 'anthropic', 'ollama', or 'gemini'."
    )
