"""Example mode — picky "default SKIP" rubric.

The antonym of `example_lenient`. Default decision is SKIP — only like
when the profile shows a clear, specific signal worth engaging with.
Demonstrates the same mode-file format but with the inverted polarity,
so you can see the dynamic range these prompts cover.

Neither this nor `example_lenient` is anyone's real preference list.
Copy one of them to `mine.py` and rewrite the PREFERENCES text to match
what you actually care about.
"""

NAME = "example_strict"
DESCRIPTION = "Picky baseline: default SKIP unless there's a specific, strong signal."

AGE_MIN = None
AGE_MAX = None

# None = inherit the generic fallback in judge.DEFAULT_MESSAGE_VOICE.
# Try `MESSAGE_VOICE = "example_polished"` for a more deliberate tone.
MESSAGE_VOICE = None

MAX_LIKES_PER_SESSION = None
MAX_PROFILES_PER_SESSION = None

PREMADES = []

PREFERENCES = """
Default decision: SKIP.

This is the strict example. The goal is to be selective — pick profiles
where there is a clearly engageable hook, not just "looks fine".

LIKE only when at least one of these is clearly true:

1. A prompt answer is specific and interesting enough to write a real,
   non-generic opener about. "Specific" means: a concrete claim, a
   distinctive detail, an unusual hobby or place — not "I love coffee
   and dogs" or other open-ended platitudes.

2. The bio / prompts mention a clear shared-interest cue that you can
   reference directly: a sport, a niche hobby, a particular book or
   show, an unusual job. The opener you'd write should be obvious from
   the cue, not strained.

3. The profile is well-constructed overall (multiple thoughtful prompt
   answers, real photos showing identifiable activities) AND nothing
   in it is a turn-off.

Skip reasons (use dominant_factor="other" unless one fits better):

- No prompt answers, or only generic stock prompts ("two truths and a
  lie" with no actual answers, "looking for [...]" with no fill-in).
- Photos only — no readable text content to write an opener about.
- Profile reads as low-effort: a single photo, recycled selfie set,
  filler text.
- Tone of the profile is incompatible with how you'd want to message
  (e.g. heavily sales-pitch / influencer self-promo when you'd prefer
  organic conversation).

When you decide LIKE but can't write a strong specific opener, lean
SKIP instead. In this strict mode an unspecific like is worth less
than a clean skip — the budget is in the opener quality, not the
volume.

When writing the reasoning field:
- Reference concrete details ("photo 3 shows...", "the [prompt] answer
  about X").
- Stay neutral about the person. This rubric is about message-fit, not
  judgment of them. Adapt the PREFERENCES text above to your own taste
  in your own mode file.
"""
