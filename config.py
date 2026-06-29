"""Configuration for HingeAuto.

PREFERENCES, AGE_MIN/MAX, and MESSAGE_VOICE come from the active mode (see
`modes/`). Set ACTIVE_MODE here for the persistent default; override per-run
via `python main.py --mode <name>`.

The COORDS defaults below are calibrated for a Pixel 10 emulator
(1080x2424). If that's what you're running and Hinge hasn't shifted
its layout, they should work as-is. Otherwise run `python calibrate.py`
and update the values that don't match your device.
"""

from pathlib import Path

# ---------- Mode selection ----------
# Which `modes/<name>.py` to load. Overridden per-run by `python main.py --mode X`.
ACTIVE_MODE = "example_lenient"

# These get filled in by _apply_mode() at the bottom of this file. Declared
# here so static analyzers / IDEs see them. Do not edit by hand — edit the
# mode module instead.
PREFERENCES: str = ""
AGE_MIN: int | None = None
AGE_MAX: int | None = None
MESSAGE_VOICE: str | None = None
MODE_NAME: str = ""
PREMADES: list[dict] = []

# ---------- Run mode ----------
# DRY_RUN = False -> actually like / send messages (default)
# DRY_RUN = True  -> decide and log, but force-skip every profile
#                    instead of liking. Every "would-like" profile gets
#                    skipped (gone from your queue) but no likes are
#                    spent.
#
# When to flip this to True:
#   - Free Hinge (8 likes/day): YES, for your first run or two. Lets
#     you watch decisions without spending your daily cap on a rubric
#     you haven't tuned. Once decisions look right, flip back to False.
#   - Hinge+ (unlimited likes): NO. Just run small live batches
#     (MAX_LIKES_PER_SESSION = 5) and Ctrl-C if something looks off.
DRY_RUN = False

# Default = 8, which matches free-tier Hinge's daily like cap (resets at
# 4am local). One session per day exhausts the free allotment cleanly.
#
# If you have Hinge+ (no daily cap), bump this to ~25-50 per session and
# run multiple sessions throughout the day. Going much higher per session
# tends to trigger Hinge's soft-throttle (empty Discover after a burst);
# spacing batches across the day works better than one giant batch.
MAX_LIKES_PER_SESSION = 8
MAX_PROFILES_PER_SESSION = 100

# ---------- Emulator settings ----------
# Moto e20 real phone is 720x1600. Change if using a different device.
SCREEN_WIDTH = 720
SCREEN_HEIGHT = 1600

# Number of scroll-and-screenshot passes per profile.
# Longer profiles (6 photos + 3 prompts) need ~7 frames at the scroll step
# below. Duplicate end frames on shorter profiles are harmless.
FRAMES_PER_PROFILE = 7

# ---------- Coordinates ----------
# Calibrated for Moto e20 (720x1600) on 2026-06-28.
# Run `python calibrate.py` to verify/adjust after any Hinge UI update.
COORDS = {
    # Skip / like action targets (Discover screen, photo 1 at top)
    "skip_button":       (89, 1319),   # X icon
    "heart_photo_1":     (624, 1354),  # Heart icon

    # Compose box (anchors to the element whose heart was tapped; these
    # values are mostly fallbacks — vision.py re-finds them at tap-time
    # because the box shifts per profile).
    "send_like_button":  (463, 872),
    "comment_input":     (333, 753),
    "compose_close":     (650, 135),

    # Scroll gesture (swipe up = scroll down through profile).
    "scroll_from":       (360, 1125),
    "scroll_to":         (360, 465),
    "scroll_duration_ms": 350,

    # Bottom nav (5 evenly-spaced icons across the bottom strip).
    "nav_discover":      (72, 1498),
    "nav_standouts":     (216, 1498),
    "nav_likes_you":     (360, 1498),
    "nav_matches":       (504, 1498),
    "nav_self_pfp":      (648, 1498),

    # Self-profile flow (used by scan_self.py — "what does my profile
    # look like to others").
    "self_avatar":       (360, 330),
    "view_tab":          (540, 205),
    "back_arrow":        (43, 135),

    # Discover filter row (tap the chip to open its bottom sheet).
    "sliders_icon":      (64, 150),
    "age_chip":          (307, 150),
}

# ---------- Timing ----------
# Random delay between actions (seconds, min/max for jitter).
DELAYS = {
    "after_scroll":     (0.6, 1.0),
    "after_screenshot": (0.2, 0.4),
    "after_tap":        (0.8, 1.4),
    "after_like_sent":  (2.5, 4.0),
    "after_skip":       (1.5, 2.5),
}

# ---------- Judge backend ----------
# "anthropic" -> uses your ANTHROPIC_API_KEY; best quality, ~$0.02-0.05/profile.
# "ollama"    -> uses Ollama Cloud (free tier) or local Ollama; lower quality
#                but no per-token cost.
# "gemini"    -> uses Gemini via GEMINI_API_KEY; cheapest option.
JUDGE_BACKEND = "anthropic"

# ---------- Anthropic settings (when JUDGE_BACKEND == "anthropic") ----------
# Sonnet is the default — cheaper than Opus and plenty capable for this task.
# Switch to "claude-opus-4-7" if you want top-quality judgment, or
# "claude-haiku-4-5" for cheapest (may miss subtle cues).
MODEL = "claude-sonnet-4-6"
EFFORT = "medium"  # low | medium | high

# ---------- Ollama settings (when JUDGE_BACKEND == "ollama") ----------
# Vision-capable models that handle multiple images per turn:
#   "qwen2.5-vl"        — strong all-around vision model (recommended)
#   "qwen2.5-vl:7b"     — smaller, faster, weaker
#   "llama3.2-vision"   — alternative; tool-calling can be flakier
OLLAMA_MODEL = "qwen2.5-vl"

# OLLAMA_HOST: None or "" -> default http://localhost:11434
#              "https://ollama.com" -> Ollama Cloud (requires OLLAMA_API_KEY)
# Can also be set via the OLLAMA_HOST environment variable.
OLLAMA_HOST = None

# ---------- Gemini settings (when JUDGE_BACKEND == "gemini") ----------
# GEMINI_API_KEY must be set in .env or environment.
# Uses gemini-3.1-flash-lite by default (cheapest vision model). Override
# via GEMINI_MODEL env var or edit the default below.
GEMINI_MODEL = "gemini-3.1-flash-lite"

# ---------- Paths ----------
BASE_DIR = Path(__file__).parent
DEBUG_DIR = BASE_DIR / "debug"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
SAVE_DEBUG_FRAMES = True  # keep frames + decisions in debug/ for review


def _apply_mode() -> None:
    """Resolve ACTIVE_MODE and populate this module's PREFERENCES /
    AGE_MIN / AGE_MAX / MESSAGE_VOICE / MODE_NAME / cap overrides.

    Re-entrant — main.py calls this again after parsing --mode so a CLI
    override takes effect before the judge sees config.
    """
    import modes
    mode = modes.load(ACTIVE_MODE)
    g = globals()
    g["PREFERENCES"] = mode.PREFERENCES
    g["AGE_MIN"] = getattr(mode, "AGE_MIN", None)
    g["AGE_MAX"] = getattr(mode, "AGE_MAX", None)
    g["MESSAGE_VOICE"] = getattr(mode, "MESSAGE_VOICE", None)
    g["MODE_NAME"] = mode.NAME
    g["PREMADES"] = list(getattr(mode, "PREMADES", []))
    for k in ("MAX_LIKES_PER_SESSION", "MAX_PROFILES_PER_SESSION"):
        v = getattr(mode, k, None)
        if v is not None:
            g[k] = v


_apply_mode()
