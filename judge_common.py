"""Backend-agnostic pieces of the judging pipeline.

Every backend module (`judge.py`, `judge_deepseek.py`, `judge_gemini.py`,
`judge_ollama.py`) imports from here so the system prompt, decision shape,
and tool schema stay in sync.

A backend module needs to expose `judge(frames: list[bytes]) -> Decision`.
"""

import re
from dataclasses import dataclass, field
from typing import Any

import httpx

import config


SYSTEM_PROMPT_TEMPLATE = """You are evaluating Hinge dating profiles on behalf of the user.

The user's preferences:
{preferences}
{age_clause}
Each profile is photos, prompt responses, and basic info (age, height, job,
education).
{location_clause}
{fit_clause}
If a dialog, popup, overlay, or other non-profile screen blocks any part of
the profile — settings panel, upsell, notification prompt, rating nag — set
decision="NOT_A_PROFILE" and describe it in reasoning.

{message_voice}
{premades_section}
Submit your decision via the submit_decision tool."""


# Generic, voice-neutral fallback used when the active mode does not set
# MESSAGE_VOICE. Replace by writing a voice file under voice/<name>.py
# and pointing your mode at it.
DEFAULT_MESSAGE_VOICE = """## Message rubric

Always write a short opener for this profile. The harness decides whether
it's actually used — a high fit_score sends it with a like, a low one
discards it. You don't decide; just draft a best-effort opener.

Aim for: one specific reference to something visible in the profile (a
prompt answer or a concrete photo detail), followed by a short question
about it. Keep it friendly and curious. Around 60-120 characters total.

Constraints:
- Plain ASCII only. No emoji, no smart quotes, no em-dashes.
- Avoid the characters \\, ", $, ` — they break the typing layer.
- Never leave `message` empty just because you think it'll be skipped —
  always draft something usable as an opener.
- Only if you genuinely cannot write a specific, non-generic opener, use
  the empty string for `message` and set `message_archetype` to "empty".
  A like with no message is still acceptable.

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
            "description": "How confident you are in the fit_score.",
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
                "A short opener for this profile — always write one; the "
                "harness sends it only with a like. Max ~150 chars, plain "
                "ASCII, no emoji. See the message rubric in the system prompt."
            ),
        },
        "dominant_factor": {
            "type": "string",
            "enum": [
                "none",
                "other",
                "age",
                "reachability",
                "looks",
                "build",
                "photos",
                "height",
                "religion",
                "low_effort",
                "grooming",
                "tattoos",
                "ethnicity",
                "lifestyle",
                "interests",
                "humor",
                "frame",
                "style",
            ],
            "description": (
                "Which single factor from the PREFERENCES rubric most drove "
                "this fit_score, for downstream analytics. Report it in BOTH "
                "directions — a factor that pulled the score UP is as worth "
                "recording as one that pulled it down; fit_score says which "
                "way it cut. \"reachability\" when the reachability gate "
                "drove the score, \"age\" when the AGE GATE fired, "
                "\"low_effort\" when the profile was too thin to engage with "
                "(no readable prompts, single photo, etc.). Use \"none\" only "
                "when no single factor dominated and the score came from the "
                "overall read of the profile, and \"other\" only if nothing "
                "above fits. Name the factor even when it argues for a like."
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
        "message", "dominant_factor", "message_archetype", "premade_id",
        "prompt_referenced",
    ],
}


@dataclass
class Decision:
    name: str
    decision: str  # model: "profile" | "NOT_A_PROFILE"; after gate: "like" | "skip" | "NOT_A_PROFILE"
    confidence: str  # "low" | "medium" | "high"
    reasoning: str
    message: str = ""
    drafted_message: str = ""  # what the model wrote before the gate (kept for logs)
    fit_score: int = 0  # 0-100, authoritative for like/skip gating
    dominant_factor: str = "none"
    message_archetype: str = "empty"
    premade_id: str = ""
    prompt_referenced: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


def decision_from_tool_args(args: dict, usage: dict) -> Decision:
    """Build a Decision from a tool-call argument dict, tolerating mild
    schema drift (non-Anthropic backends miss keys more often than Claude).

    Used by the OpenAI-compatible backends (Ollama, DeepSeek). Missing
    fields fall back to safe defaults; enum-like fields are clamped to
    allowed values so a stray value can't break the loop.
    """
    defaults = {
        "name": "unknown",
        # Missing/odd verdicts are treated as a scoreable profile — the
        # fit-score gate downstream is what decides like vs skip.
        "decision": "profile",
        "fit_score": 0,
        "confidence": "low",
        "reasoning": "",
        "message": "",
        "dominant_factor": "other",
        "message_archetype": "empty",
        "premade_id": "",
        "prompt_referenced": "",
    }
    merged = {**defaults, **{k: v for k, v in args.items() if k in defaults}}
    # Clamp the model's profile/NOT_A_PROFILE verdict — NOT_A_PROFILE must
    # survive so main.py's dialog recovery still fires. Anything else
    # (including a stray like/skip) is treated as a scoreable profile;
    # like vs skip is decided downstream by apply_fit_threshold.
    if merged["decision"] != "NOT_A_PROFILE":
        merged["decision"] = "profile"
    if merged["confidence"] not in ("low", "medium", "high"):
        merged["confidence"] = "low"
    return Decision(**merged, usage=usage)


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


def _fit_clause() -> str:
    return (
        f"\nFIT SCORE: one integer fit_score (0-100) for how well this profile "
        f"matches the user's preferences. Use the full range — don't cluster "
        f"around the middle. 90+ = exact match, 75-89 = strong fit, 60-74 = "
        f"decent, 40-59 = neutral, 0-39 = not a fit. You do NOT choose like vs "
        f"skip — the harness decides that from fit_score. Always draft an "
        f"opener (see the message rubric). When genuinely ambiguous, lean "
        f"toward a lower score.\n"
    )


def _age_clause(age_min: int | None, age_max: int | None) -> str:
    if age_min is None and age_max is None:
        return ""
    lo = age_min if age_min is not None else 18
    hi = age_max if age_max is not None else 99
    return (
        f"\nAGE GATE: only score as a fit if the profile's stated age is between "
        f"{lo} and {hi} inclusive. Hinge shows age in basic-info "
        f"(\"NN\" next to height/location). If age is visible and out of "
        f"range, set fit_score=0 and dominant_factor=\"age\" so the harness skips "
        f"it. If age genuinely isn't visible across any frame, proceed with "
        f"the normal rubric.\n"
    )


def _location_clause() -> str:
    """Tell the model to disregard location.

    Dropping "location" from the profile-info list isn't enough on its own:
    Hinge renders a hometown and a distance right in the basic info, so the
    judge reads them off the screenshots whether or not we name the field.
    This clause is the half that actually does the work.
    """
    return (
        "\nLOCATION: ignore it. Hometown and distance "
        '("3 miles away") are decoration — never let them affect fit_score, '
        "and don't mention them in your reasoning or the opener you draft.\n"
    )


def build_system_prompt() -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        preferences=config.PREFERENCES.strip(),
        age_clause=_age_clause(config.AGE_MIN, config.AGE_MAX),
        location_clause=_location_clause(),
        fit_clause=_fit_clause(),
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
    """Decide like/skip entirely from fit_score and the FIT_SCORE_MIN dial.

    The model never chooses like vs skip — it only scores each profile and
    always drafts an opener. This is the single place the run decides: like
    iff fit_score >= config.FIT_SCORE_MIN. NOT_A_PROFILE is preserved so
    dialog/popup recovery keeps working. Clamps fit_score to 0-100 and
    mutates + returns `decision`.
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
    # Preserve the model's drafted opener before any discard so it can be
    # logged for review even when the profile is gated to a skip.
    decision.drafted_message = decision.message
    if decision.decision == "skip":
        # The model always drafts an opener; never ship it on a skip. The
        # harness decides, so an opener on a sub-threshold profile is dropped.
        decision.message = ""
        decision.message_archetype = "empty"
        decision.premade_id = ""
    return decision


# ── Fatal judge errors ────────────────────────────────────────────────
# Errors that will NOT clear on a retry. Retrying them burns Hinge swipes
# blind: a judge that fails all three attempts falls through to a
# force-skip, so a dead API key or an empty balance silently skips every
# profile in the feed until the daily quota is gone. (Seen once on
# Anthropic when the credit balance hit zero mid-run — 124 profiles were
# skipped before anyone noticed.)
#
# Classify by HTTP status where one exists; that's the only
# backend-independent signal. Providers word the same failure differently
# (Anthropic: "credit balance is too low", DeepSeek: "Insufficient
# Balance"), so message text is a fallback, not the primary key.
FATAL_STATUS_CODES = frozenset({
    400,  # malformed request / unsupported parameter — a config bug, not a blip
    401,  # bad or missing API key
    402,  # payment required — DeepSeek's "Insufficient Balance"
    403,  # key is valid but not permitted for this call
    404,  # model not found
    422,  # unprocessable request — also a request-shape bug
})

# Catches backends that only put the code in the text, e.g.
# judge_deepseek.py's RuntimeError("DeepSeek API error 402: {...}").
_STATUS_IN_TEXT = re.compile(r"(?:API|HTTP|status)\D{0,10}(\d{3})\b", re.IGNORECASE)

# Last resort, for errors carrying neither a status attribute nor a
# parseable one in the message.
_FATAL_PHRASES = (
    "credit balance is too low",
    "insufficient balance",
    "insufficient_quota",
    "authentication_error",
    "invalid_api_key",
    "permission_error",
    "invalid_request_error",
    "model not found",
    # Shared tail of the "<NAME>_API_KEY not set. Add it to .env or export
    # it." guard in judge_gemini.py and judge_deepseek.py. A missing key
    # never clears on retry, and without this the loop spends all three
    # attempts failing before blindly skipping every profile in the feed.
    "not set. add it to .env",
)


def _http_status(exc: BaseException) -> int | None:
    """Best-effort HTTP status for an exception from any backend."""
    for attr in ("status_code", "status", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    # requests/httpx style: HTTPStatusError carries .response.status_code
    val = getattr(getattr(exc, "response", None), "status_code", None)
    return val if isinstance(val, int) else None


def is_fatal_judge_error(exc: BaseException) -> bool:
    """True if `exc` won't clear on retry, so the run should halt.

    Walks the exception chain, because backends and vendor SDKs wrap the
    interesting error (`raise X from e`) and the status usually sits on an
    inner link rather than the one the caller catches.
    """
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))

        status = _http_status(cur)
        if status is not None and status in FATAL_STATUS_CODES:
            return True

        text = str(cur)
        for match in _STATUS_IN_TEXT.finditer(text):
            if int(match.group(1)) in FATAL_STATUS_CODES:
                return True

        lowered = text.lower()
        if any(phrase in lowered for phrase in _FATAL_PHRASES):
            return True

        cur = cur.__cause__ or cur.__context__
    return False


# httpx is the transport under every backend — the Anthropic and OpenAI SDKs
# both sit on it, google-genai does too, and judge_deepseek.py calls it
# directly — so a transport-level failure surfaces as the same family of
# exceptions whichever judge is configured.
#
# The names cover errors an SDK wrapper raises instead of passing the httpx
# one through: Anthropic's APIConnectionError subclasses APIError, not
# httpx.TransportError, so isinstance alone would miss it. gaierror is DNS
# failure, the usual shape of "the internet cut out" underneath httpx's
# own wrapping.
_NETWORK_ERROR_NAMES = frozenset({
    "ConnectError", "ConnectTimeout", "ReadError", "ReadTimeout",
    "WriteError", "WriteTimeout", "PoolTimeout", "RemoteProtocolError",
    "APIConnectionError", "APITimeoutError", "gaierror",
})


def is_network_error(exc: BaseException) -> bool:
    """True if `exc` means the request never reached the model.

    These are deliberately absent from FATAL_STATUS_CODES: a dropped packet
    is retryable in principle, so one shouldn't abort a run. But when they
    outlast every retry the network is down, and the caller has to end the
    run — force-skipping would spend real Hinge profiles on people the judge
    never scored. Walks the exception chain the way is_fatal_judge_error
    does, so wrapper layers don't hide the cause.
    """
    seen = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, httpx.TransportError):
            return True
        if _NETWORK_ERROR_NAMES & {c.__name__ for c in type(cur).__mro__}:
            return True
        cur = cur.__cause__ or cur.__context__
    return False


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
    if backend == "deepseek":
        import judge_deepseek
        return judge_deepseek
    raise ValueError(
        f"Unknown JUDGE_BACKEND={backend!r}. Use 'anthropic', 'deepseek', "
        f"'gemini', or 'ollama'."
    )
