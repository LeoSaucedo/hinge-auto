"""Cougar Mode — age band + young-buck premades.

A themed example mode for a younger user (early 20s) targeting an older
age band, with openers that name the dynamic playfully. This is the
mode behind the "AI bot dates an older crowd" framing some viewers
asked about.

Demonstrates capabilities the basic examples don't:
  - AGE_MIN / AGE_MAX (judge-side age gate; pair with --set-filters
    to also drive Hinge's in-app slider)
  - Inline MESSAGE_VOICE (a multi-line voice string instead of
    referencing voice/<name>.py)
  - Themed PREMADES that lean into a specific opener angle

What this mode is NOT: a demographic-filter mode. Bodily, ethnic, and
"culture signal" filtering rules from a previous iteration of this
mode were intentionally removed when going public — they were the
author's personal preferences, not part of the demo. If you want
those, write them yourself in a private mode file (and re-read the
guidance in template.py.example about not pushing taste rules into
shared examples).
"""

NAME = "cougar"
DESCRIPTION = (
    "Targets older age band (33-44) with young-buck-themed openers. "
    "Pair with `python main.py --mode cougar --set-filters` to also "
    "drive Hinge's in-app age slider."
)

# Hinge in-app filter can also enforce this — but the judge-side gate
# is a backstop in case the filter snaps off (Hinge+ tiering, etc).
AGE_MIN = 33
AGE_MAX = 44

MAX_LIKES_PER_SESSION = None  # inherit config default
MAX_PROFILES_PER_SESSION = None

# Verbatim openers the judge can pick from. The first one is the line
# the project got asked about; it's the default pick on most likes
# in this mode. The second is for profiles that read more
# explicitly flirty / forward — name the dynamic directly.
PREMADES = [
    {
        "id": "young_buck_default",
        "message": "Does the young buck get a shot?",
        "use_when": (
            "DEFAULT pick for likes in this mode. Frames the user as the "
            "younger person auditioning for the older woman — the signature "
            "line. Use unless the profile reads as explicitly flirty / "
            "forward (then prefer young_buck_cougar) or there's a very "
            "strong specific prompt callback available (then write fresh)."
        ),
    },
    {
        "id": "young_buck_cougar",
        "message": "does the young buck have a shot at the cougar role",
        "use_when": (
            "Use ONLY when the profile reads as explicitly flirty / forward "
            "— concrete signals like body-forward photos AS THE PRIMARY "
            "PHOTO SET (3+ such photos, not just one), prompt answers with "
            "obvious sexual subtext, or 'short-term' / 'figuring it out' "
            "Dating Intentions paired with that energy. A single beach pic "
            "or one flirty line is NOT enough — fall back to "
            "young_buck_default. Names the cougar dynamic directly; lands "
            "in the right context, reads creepy in the wrong one."
        ),
    },
]

# Inline voice — overrides judge_common.DEFAULT_MESSAGE_VOICE for this
# mode. Could equivalently live in voice/cougar_playful.py and be
# referenced as MESSAGE_VOICE = "cougar_playful".
MESSAGE_VOICE = """## Message rubric (when decision == "like")

Default: pick one of the PREMADES below — they're the whole reason
this mode exists. Write a fresh opener only when there's a genuinely
specific, strong prompt callback that beats the premade.

When writing fresh:

**Voice:**
- All lowercase. No exclamation marks.
- 50-90 characters total.
- Playful, slightly self-aware about the age gap. NOT thirsty, NOT
  earnest, NOT romantic.
- Comfortable naming the dynamic when the profile vibes that way.
- Sound like a 24-year-old who knows what he's doing.

**Avoid:**
- "Genuinely", "honestly", "truly" as intensifiers.
- "Love that", "obsessed", "iconic", "fellow [X]".
- Em-dashes. Plain hyphens only.
- Generic age-gap pickup lines ("age is just a number" etc — instant
  cringe).
- Pretending the age gap isn't there. The premise is the hook.

**What works:**
1. Specific prompt callback with a playful angle — e.g. if her prompt
   says she runs a business, "young buck applying for an internship?"
2. Tease a prompt claim, with the dynamic implied — e.g. "claims to
   like spontaneous, lets see how spontaneous"
3. Use a PREMADE. Most likes in this mode should be a premade.

**Hard constraints:**
- Plain ASCII only. No emoji, no smart quotes, no em-dashes.
- Avoid \\, ", $, ` — they break the typing layer.
- Empty string when decision == "skip"."""

PREFERENCES = """
This mode targets a specific dating dynamic — younger user, older age
band, openers that name it playfully. It is NOT a demographic-filter
mode. Decisions here are about VIBE COMPATIBILITY for that dynamic,
not about who deserves a like.

Default: LIKE. Lean strongly toward LIKE — the volume on this age
band is already self-limiting, and the premades do the work of
filtering out bad matches at the message-reply stage.

SKIP signals (use sparingly):

1. The profile signals heavy commitment / settling-down energy that
   is incompatible with the young-buck framing the openers carry.
   Concrete signals (need at least TWO):
   - Dating Intentions = "Life partner" or "Long-term relationship"
     PAIRED with a prompt about "my person" / "settling down" /
     "ready for the next chapter" / wanting kids soon.
   - Bio/prompt language about marriage, wedding, "building a
     family", or similar long-horizon framing.
   - A single Long-term-relationship Dating Intention by itself is
     NOT a skip — many people select it by default.
   Use dominant_factor="other".

2. Profile is bot-like / spam (single AI-looking photo, only an
   off-platform handle in the bio, photos contradict the stated info).
   Use dominant_factor="low_effort".

3. Profile is too thin to write an opener at all — no readable
   prompts, no specific photo details, just basic info. The premades
   work without a specific hook, so this rule should fire rarely.
   Use dominant_factor="low_effort".

Everything else is a LIKE. Career, hobbies, photo settings, vibe,
group dynamics — none are skip signals. Don't get clever.

When writing reasoning:
- Reference concrete details ("photo 3 shows...", "the [prompt]
  answer about X").
- Stay neutral and non-judgmental about the person. The rubric is
  about vibe-fit for THIS specific opener angle, not a judgment of
  her.
"""
