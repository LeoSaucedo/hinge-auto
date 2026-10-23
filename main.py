"""HingeAuto orchestrator.

Loop: capture profile frames -> ask Claude -> tap like or skip -> repeat.
"""

import argparse
import fcntl
import hashlib
import random
import re
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

import io

from PIL import Image

import adb
import config
import metrics
import report
import vision
from judge_common import (apply_fit_threshold, call_with_deadline,
                          is_fatal_judge_error, is_judge_unavailable,
                          is_network_error, load_backend)

judge = load_backend().judge

# Set once an action has advanced the feed to a fresh profile rendered at the
# top of the screen — either a confirmed like (do_like) or a skip tap
# (do_skip). The next capture_profile() can then skip its defensive
# scroll-back, which would otherwise burn 3-8 swipes on a profile that is
# already at the top. Consumed on read.
_feed_at_top = False


def _profile_region_hash(png: bytes) -> str:
    """md5 over a cropped region of the frame, excluding status bar (clock
    ticks every minute) and bottom nav (badges flicker). Two screenshots
    of the same profile taken 60+ seconds apart should produce the same
    hash; two different profiles should always differ."""
    img = Image.open(io.BytesIO(png))
    w, h = img.size
    crop = img.crop((0, 120, w, h - 220))
    return hashlib.md5(crop.tobytes()).hexdigest()


def capture_profile() -> list[bytes]:
    """Scroll through the current profile, returning a list of PNG frames."""
    global _feed_at_top

    # Check for stuck loading screen BEFORE any scrolls or interactions.
    # Saves ~3-8 wasted scroll-up swipes + avoids burning API credits
    # judging a loading screen as if it were a profile.
    initial = adb.screenshot()
    if vision.is_app_loading(initial):
        raise RuntimeError("app stuck on loading screen")

    if _feed_at_top:
        # The last action (confirmed like or skip) advanced the feed to a
        # fresh profile already rendered at the top — the scroll-back below
        # would burn 3-8 swipes on nothing. Pause instead, so the new profile
        # still gets the beat to render that the swipes used to provide (an
        # unrendered feed reads as a stuck loading screen and force-restarts
        # the app).
        print("Feed already at top after last action — skipping scroll-back.")
        time.sleep(1.5)
        _feed_at_top = False
    else:
        # Defensive: new profiles load at the top, so this is just guarding
        # against the app being mid-scroll from a prior partial action. A
        # handful of swipes is enough — full 18-swipe sweep isn't needed
        # because we aren't recovering from a 7-frame scroll-down.
        scroll_back_to_top(swipes=random.randint(3, 8))

    frames = []
    frames.append(adb.screenshot())
    adb.jitter_sleep("after_screenshot")

    for _ in range(config.FRAMES_PER_PROFILE - 1):
        adb.scroll_down()
        adb.jitter_sleep("after_scroll")
        frames.append(adb.screenshot())
        adb.jitter_sleep("after_screenshot")

    return frames


def scroll_back_to_top(swipes: int | None = None) -> None:
    """Scroll back to the top of the profile.

    Hinge uses momentum scrolling; each swipe's actual travel is much less
    than the gesture's pixel distance. Default `swipes` is `FRAMES_PER_PROFILE
    * 2 + 4` (18 with FRAMES=7) — enough to recover from a full scroll-down
    through the profile. Pass a smaller `swipes` for defensive top-scrolls
    where the profile is already at or near the top.
    """
    if swipes is None:
        swipes = config.FRAMES_PER_PROFILE * 2 + 4
    print(f"Scrolling back to top ({swipes} swipes)...")
    for _ in range(swipes):
        adb.scroll_up()
        adb.jitter_sleep("after_scroll")
    print(f"Scrolling stopped — at top of profile.")


def do_skip() -> None:
    """Tap the X to advance to the next profile. Always taps (even in dry run);
    advancing is needed for the loop to see new profiles."""
    global _feed_at_top
    x, y = config.COORDS["skip_button"]
    adb.tap(x, y)
    adb.jitter_sleep("after_skip")
    # A skip advances the feed the same way a sent like does: the next profile
    # renders already at the top. Flag it only after the tap returns — if
    # adb.tap itself failed there's no new profile to be at the top of, and
    # leaving the flag clear keeps the defensive scroll-back for that case.
    _feed_at_top = True


class FeedAlreadyAdvanced(RuntimeError):
    """do_like aborted *after* the feed moved on — this profile is spent.

    Clearing a stale compose card means tapping skip, which is the same
    gesture as skipping a profile. main's do_like handler treats any failure
    as "recover by skipping this profile", so a plain RuntimeError here made
    it tap skip a second time — spending the *next* profile too, one the
    judge never saw. Raising a distinct type lets the handler tell "the feed
    already moved" from "the like failed and nothing has moved yet".
    """


# Consecutive Send Like failures (button not found, or the card never
# closed) mean the send path is broken, not that one profile was unlucky.
# Sitting on it burns profiles without sending likes, so the run gives up
# at this streak and reports every failure screenshot.
SEND_LIKE_STREAK_LIMIT = 3


class SendLikeFailure(RuntimeError):
    """do_like reached the Send Like step and that step failed.

    Carries the debug screenshot saved at the failure site so main() can
    attach it to the run's single abort post. main() aborts after
    SEND_LIKE_STREAK_LIMIT of these on consecutive like attempts; skips
    and failures before the Send Like step leave the streak untouched.
    """

    def __init__(self, message: str, screenshot_path: str | None = None):
        super().__init__(message)
        self.screenshot_path = screenshot_path


class RoseSheetStuck(SendLikeFailure):
    """The "Send a Rose instead?" sheet survived a tap and a Back press.

    Fatal for the run, not for the profile: the sheet covers the composer
    *and* the skip X (which floats above it but lands on the sheet's
    margin and does nothing). No tap the loop knows how to make advances
    the feed from here, and tapping repeatedly risks spending the rose
    twice. main() aborts immediately instead of skipping.
    """


class RosePurchaseStuck(SendLikeFailure):
    """The rose purchase page survived Back.

    That page is the answer to a rose tap on an empty balance, and every
    control on it spends money. If Back does not clear it, no tap the loop
    knows how to make is safe, so main() aborts immediately instead of
    skipping — skip lands on the page's margin and does nothing anyway.
    """


def _dismiss_compose_card_if_visible() -> None:
    """Check for a stale compose card, rose sheet, or purchase page.

    If one is still open, something went wrong upstream — resolve it and
    advance past this profile, logging the event.
    """
    ss = adb.screenshot()
    if _recover_from_rose_purchase("stale", ss):
        # The page is only ever the answer to a rose send, so the like
        # underneath it was interrupted. Recovery either sent that like or
        # revealed that the feed already moved — either way this profile
        # is consumed and must not be skipped a second time.
        raise FeedAlreadyAdvanced(
            "stale rose purchase page resolved; feed already advanced"
        )
    sheet = vision.find_rose_sheet(ss)
    if sheet is not None:
        # The sheet is a *response* to a send tap, so the like underneath
        # it never went out and the feed never advanced. Finish that like
        # the safe way — never with a rose here, this is recovery.
        print("⚠️  Stale rose sheet detected — resolving the interrupted like")
        save_error_screenshot("stale-rose-sheet", ss)
        if sheet.send_like_anyway is not None:
            adb.tap(*sheet.send_like_anyway)
            adb.jitter_sleep("after_like_sent")
            raise FeedAlreadyAdvanced(
                "stale rose sheet resolved; feed already advanced"
            )
        _fail_after_sheet("stale rose sheet has no 'Send Like anyway' target",
                          "stale", ss)
    if vision.find_send_like(ss) is not None:
        print("⚠️  Stale compose card detected — tapping skip")
        save_error_screenshot("stale-compose-card", ss)
        do_skip()
        raise FeedAlreadyAdvanced(
            "stale compose card dismissed; profile already skipped"
        )


def _fail_after_sheet(reason: str, context: str,
                      png: bytes | None = None) -> None:
    """Leave a rose sheet via Back, then report the screen it uncovered.

    Always raises, with the type chosen by what Back revealed:

      sheet still up -> RoseSheetStuck (main aborts; never tap skip on it)
      compose card   -> SendLikeFailure (main skips this profile)
      neither        -> FeedAlreadyAdvanced (the like may have landed)

    The error screenshot is taken first, on the sheet itself — that is the
    frame worth attaching to the abort post. `png` is the caller's frame
    showing that sheet; without it one is captured here.
    """
    screenshot = _screenshot_or_none(f"rose-sheet-{context}", png)
    print(f"Rose sheet failure: {reason} — pressing Back.")
    _recover_from_dialog()
    png = adb.screenshot()
    if vision.find_rose_sheet(png) is not None:
        raise RoseSheetStuck(
            f"{reason}; rose sheet survived Back — aborting instead of skipping",
            screenshot_path=screenshot,
        )
    if vision.find_send_like(png) is not None:
        raise SendLikeFailure(reason, screenshot_path=screenshot)
    raise FeedAlreadyAdvanced(reason)


def _recover_from_rose_purchase(context: str,
                                png: bytes | None = None) -> bool:
    """Leave the rose purchase page and finish as a normal like.

    The page appears when the rose pill — or "Send a Rose" on the sheet —
    is tapped with an empty balance, i.e. exactly when the 0-rose detection
    was supposed to stop us. Back returns to the composer with the draft
    intact, so one Send Like there sends the like without a rose. If Back
    does not clear the page, abort: every control on it buys something, so
    no tap the loop knows how to make is safe.

    `png` is a frame the caller just matched the page on; pass it so this
    does not capture and re-check the same screen. Without it, one
    screenshot is taken here.

    Returns False when there was no purchase page to handle (the caller
    can carry on with what it saw), True once the page is resolved and
    the send settled. A purchase page never spends a rose.
    """
    if png is None:
        png = adb.screenshot()
    if not vision.find_rose_purchase(png):
        return False

    screenshot = _screenshot_or_none(f"rose-purchase-{context}", png)
    print("Rose purchase page detected — pressing Back, then sending a "
          "normal like.")
    _recover_from_dialog()
    png = adb.screenshot()
    if vision.find_rose_purchase(png):
        raise RosePurchaseStuck(
            "rose purchase page survived Back — aborting instead of tapping "
            "through a page of buy buttons",
            screenshot_path=screenshot,
        )

    send_xy = vision.find_send_like(png)
    if send_xy is not None:
        print(f"Tapping Send Like at {send_xy} — sending like without a rose. 🎯")
        adb.tap(*send_xy)
        adb.jitter_sleep("after_like_sent")
        if adb.dismiss_keyboard_if_visible():
            print("Keyboard was still open after like — dismissed.")
        # No second purchase recovery: if the page comes back after this
        # ordinary send, something is badly wrong and the run must stop.
        _settle_after_send(False, f"{context}-after-purchase",
                           recover_purchase=False)
        return True

    sheet = vision.find_rose_sheet(png)
    if sheet is not None:
        # Back uncovered a sheet instead of the composer (rare, but it is
        # the one screen we know how to resolve safely).
        _resolve_rose_sheet(sheet, prefer_rose=False, context=context,
                            png=png)
        return True

    # Neither: Back landed on the feed, or the like had already gone out.
    print("  Rose purchase page cleared — nothing left to send.")
    return True


def _resolve_rose_sheet(sheet: vision.RoseSheet, prefer_rose: bool,
                        context: str, png: bytes | None = None) -> bool:
    """Tap exactly one button on a rose sheet; True if that button was a rose.

    A rose intent can only be honoured when the "Send a Rose" target was
    actually matched; a sheet without it falls back to "Send Like anyway"
    rather than tapping blind. One resolution tap per sighting: if the
    sheet is still up afterwards, stop tapping and back out — repeated
    taps either hit nothing or spend a second action.

    `png` is the caller's frame showing the sheet, used for error evidence;
    without it, the failure paths capture their own.
    """
    if prefer_rose and sheet.send_a_rose is not None:
        target, rose, label = sheet.send_a_rose, True, "Send a Rose"
    elif sheet.send_like_anyway is not None:
        target, rose, label = sheet.send_like_anyway, False, "Send Like anyway"
    else:
        _fail_after_sheet(
            "rose sheet is up but neither button was found", context, png
        )

    print(f"Rose sheet detected — tapping '{label}' at {target}"
          + (" 🌹" if rose else ""))
    adb.tap(*target)
    adb.jitter_sleep("after_like_sent")
    if adb.dismiss_keyboard_if_visible():
        print("Keyboard was still open after resolving the sheet — dismissed.")

    png = adb.screenshot()
    if vision.find_rose_purchase(png):
        # "Send a Rose" opened the purchase page, so the balance was empty
        # after all. Back out and send the like without a rose; nothing
        # was spent. Recovery either resolves the page or aborts.
        _recover_from_rose_purchase(context, png)
        return False
    if vision.find_rose_sheet(png) is not None:
        _fail_after_sheet("rose sheet still up after one tap", context, png)
    if vision.find_send_like(png) is not None:
        # Sheet gone, card back: the resolution tap did not send anything.
        raise SendLikeFailure(
            f"like not confirmed: compose card still open after '{label}' tap",
            screenshot_path=_screenshot_or_none(context, png),
        )
    return rose


def _purchase_during_settle(recover_purchase: bool, context: str,
                            png: bytes) -> bool:
    """Handle a purchase page seen while confirming a send; always False.

    False is "no rose spent", which is true either way: a recovery sends
    an ordinary like. With recover_purchase=False the page came back after
    that recovery, which no retry can improve on — abort instead. `png` is
    the frame the caller matched the page on, so neither path recaptures it.
    """
    if not recover_purchase:
        raise RosePurchaseStuck(
            "rose purchase page reappeared after backing out and sending a "
            "normal like — aborting",
            screenshot_path=_screenshot_or_none(
                f"rose-purchase-{context}", png),
        )
    _recover_from_rose_purchase(context, png)
    return False


def _settle_after_send(prefer_rose: bool, context: str,
                       pill_tapped: bool = False,
                       recover_purchase: bool = True,
                       png: bytes | None = None,
                       fallback_send_xy: tuple[int, int] | None = None) -> bool:
    """Confirm a send went out, resolving a rose sheet if one came up.

    Returns True if a rose was spent: the sheet was resolved with "Send a
    Rose", or the send was the pill tap itself (`pill_tapped`) and no
    sheet, card, or purchase page came back. A purchase page is not a
    spend — it means the balance was empty — and it is backed out of and
    finished as an ordinary like.

    A sent like dismisses the compose card and advances the feed; so a
    visible card, a visible sheet, a purchase page, or any combination all
    mean "not sent". The card gets one extra beat before that is called,
    because a mid-dismiss card can still register (same retry the
    pre-sheet code used).

    `png` is a frame the caller already captured of this same post-tap
    state; without it, the first check captures its own.
    `fallback_send_xy` makes a card that is still up on that first check
    mean "the tap before this did nothing" rather than "not sent": it taps
    Send Like once at that point and settles *that* send. Only the rose-pill
    path passes it, so the fallback itself is never retried.
    """
    if png is None:
        png = adb.screenshot()
    if vision.find_rose_purchase(png):
        return _purchase_during_settle(recover_purchase, context, png)
    sheet = vision.find_rose_sheet(png)
    if sheet is not None:
        return _resolve_rose_sheet(sheet, prefer_rose, context, png)

    if vision.find_send_like(png) is not None:
        if fallback_send_xy is not None:
            print("  Rose tap had no effect — falling back to Send Like.")
            print(f"Tapping Send Like at {fallback_send_xy} — sending "
                  f"like! 🎯")
            adb.tap(*fallback_send_xy)
            adb.jitter_sleep("after_like_sent")
            if adb.dismiss_keyboard_if_visible():
                print("Keyboard was still open after like — dismissed.")
            # The pill tap itself did nothing, but the rose intent stands:
            # a sheet raised by this Send Like is resolved with "Send a
            # Rose" (with purchase-page protection if the balance is 0).
            # No fallback_send_xy here: the fallback gets one attempt.
            return _settle_after_send(prefer_rose, context,
                                      recover_purchase=recover_purchase)
        time.sleep(1.5)
        png = adb.screenshot()
        if vision.find_rose_purchase(png):
            return _purchase_during_settle(recover_purchase, context, png)
        sheet = vision.find_rose_sheet(png)
        if sheet is not None:
            return _resolve_rose_sheet(sheet, prefer_rose, context, png)
        if vision.find_send_like(png) is not None:
            raise SendLikeFailure(
                "like not confirmed: compose card still open after Send Like tap",
                screenshot_path=_screenshot_or_none(context, png),
            )
        return pill_tapped

    # Neither card nor sheet yet. The sheet renders a beat behind the
    # send tap, so give it one more look before calling the like sent.
    time.sleep(1.5)
    png = adb.screenshot()
    if vision.find_rose_purchase(png):
        return _purchase_during_settle(recover_purchase, context, png)
    sheet = vision.find_rose_sheet(png)
    if sheet is not None:
        # Late sheet: the card already read as closed, so the send state
        # is ambiguous — resolve with the option that cannot overspend.
        return _resolve_rose_sheet(sheet, prefer_rose=False, context=context,
                                   png=png)
    if vision.find_send_like(png) is not None:
        raise SendLikeFailure(
            "like not confirmed: compose card still open after Send Like tap",
            screenshot_path=_screenshot_or_none(context, png),
        )
    return pill_tapped


def _skip_unless_rose_sheet(context: str) -> bool:
    """do_skip, unless a rose sheet or purchase page covers the skip button.

    The sheet spans the lower half of the screen and the purchase page is
    full-screen; skip (89, 1319) lands on their margins and does nothing,
    so an error-path skip taken while either is up stalls the loop on the
    same screen. Back out first; if the sheet survives that, raise
    RoseSheetStuck so main aborts rather than tapping. A purchase page is
    resolved like any other: Back, then a normal like — and since that
    send moves the feed, the profile is not skipped again.

    Returns True when the skip ran, False when a rose UI screen was
    resolved instead. Only error paths pay the screenshot — do_skip
    itself is untouched.
    """
    png = adb.screenshot()
    if vision.find_rose_purchase(png):
        print(f"Rose purchase page is up during {context} — backing out and "
              f"finishing the like instead of skipping.")
        _recover_from_rose_purchase(context, png)
        return False
    if vision.find_rose_sheet(png) is not None:
        print(f"Rose sheet is up during {context} — pressing Back before skip.")
        _recover_from_dialog()
        png = adb.screenshot()
        if vision.find_rose_sheet(png) is not None:
            raise RoseSheetStuck(
                f"rose sheet survived Back during {context}",
                screenshot_path=_screenshot_or_none(
                    f"rose-sheet-{context}", png),
            )
    do_skip()
    return True


def do_like(message: str = "", prefer_rose: bool = False) -> bool:
    """Flow: scroll to top → tap heart → type message → dismiss keyboard →
    press Send Like (or the rose pill) → resolve a rose sheet if one comes up.

    `prefer_rose` is main()'s call: this function does not read the fit
    threshold or the session cap, it just tries the rose when asked and
    falls back to an ordinary like when the pill is missing or shows the
    0-rose count. Returns True when a rose was (or is assumed) spent.
    """
    global _feed_at_top

    if config.DRY_RUN:
        do_skip()
        return False
    scroll_back_to_top()
    # Let the profile settle after scrolling.
    time.sleep(0.5)
    # Dismiss any stale compose card from a previous failed like.
    # (Modal overlay — scroll position doesn't affect it, so check here
    # where the screen is clean and heart detection won't false-match.)
    _dismiss_compose_card_if_visible()
    print("Scrolled to top — running OpenCV scans")

    # ── 4. Find and tap the black heart ──
    heart_xy = vision.find_first_heart(adb.screenshot())
    if heart_xy is None:
        save_error_screenshot("heart-not-found")
        raise RuntimeError("vision: couldn't find photo-1 heart after scroll-back")
    print(f"Clicking heart at {heart_xy}")
    adb.tap(*heart_xy)
    adb.jitter_sleep("after_tap")

    # ── 5. Type the message ──
    if message and not config.DRY_RUN_MESSAGE:
        # Hinge auto-focuses the comment field after the heart tap —
        # the keyboard opens on its own. Wait for the compose card
        # to settle before typing; don't tap the field (tapping away
        # would dismiss the keyboard).
        adb.jitter_sleep("after_tap")
        # ── Density check commented out (never successful) ──
        # est_send_y = heart_xy[1] + int(105 * config.SCALE_Y)
        # empty_pixels = vision.comment_field_text_pixels(
        #     adb.screenshot(), (int(config.SCREEN_WIDTH / 2), est_send_y))
        adb.input_text(message)
        # deadline = time.monotonic() + 15
        # target_pixels = empty_pixels + max(150, 20 * len(message))
        # while time.monotonic() < deadline:
        #     time.sleep(1.0)
        #     current = vision.comment_field_text_pixels(
        #         adb.screenshot(), (int(config.SCREEN_WIDTH / 2), est_send_y))
        #     if current >= target_pixels:
        #         break
        # else:
        #     print(f"WARN: typed text didn't reach expected pixel density "
        #           f"(have {current}, want {target_pixels}) — sending anyway.")

    # ── 6. Dismiss keyboard ──
    if message and not config.DRY_RUN_MESSAGE:
        adb.dismiss_keyboard_if_visible()

    # ── 7. Find the send target; pick rose vs like ──
    # Dismiss keyboard first in case Hinge auto-focused the comment field
    # and the keyboard is covering Send Like.
    adb.dismiss_keyboard_if_visible()
    png = adb.screenshot()
    send_xy = vision.find_send_like(png, log_miss=True)
    if send_xy is None:
        raise SendLikeFailure(
            "vision: couldn't find Send Like after heart tap",
            screenshot_path=_screenshot_or_none("send-like-not-found", png),
        )

    # Read the rose availability now, on the undimmed composer: the "Send a
    # Rose instead?" sheet dims the pill to V~59, so a post-send read cannot
    # see it. Hinge draws the 0-rose count only when the balance is empty,
    # so any other pill means at least one rose. Only run the finder when
    # this like could actually spend a rose.
    pill = vision.find_rose_pill(png, send_xy) if prefer_rose else None
    rose_ready = pill is not None and not pill.empty
    if prefer_rose and not rose_ready:
        why = ("no rose pill on screen" if pill is None
               else "the pill shows 0 roses")
        print(f"  🌹 Rose wanted for this like, but unavailable ({why}) — "
              f"sending a normal like.")

    rose_spent = False
    if prefer_rose and rose_ready:
        print(f"  🌹 Rose wanted — the pill is not the 0-rose count; "
              f"tapping Rose at {pill.center} to send it.")
        adb.tap(*pill.center)
        adb.jitter_sleep("after_like_sent")
        # Clean up any lingering keyboard so it doesn't carry over to the
        # next profile and block heart / Send Like detection.
        if adb.dismiss_keyboard_if_visible():
            print("Keyboard was still open after rose tap — dismissed.")

        # Three outcomes: the rose went out (card closes), the sheet came
        # up (settlement resolves it — as a rose), or the tap did nothing
        # (card still up, no sheet): settlement falls back to Send Like
        # once, keeping the rose intent. A purchase page counts as "send
        # settled", never as a card, so no Send Like tap can land on it.
        rose_spent = _settle_after_send(
            rose_ready, "like-not-confirmed", pill_tapped=True,
            png=adb.screenshot(), fallback_send_xy=send_xy)
    else:
        print(f"Tapping Send Like at {send_xy} — sending like! 🎯")
        adb.tap(*send_xy)
        adb.jitter_sleep("after_like_sent")
        if adb.dismiss_keyboard_if_visible():
            print("Keyboard was still open after like — dismissed.")
        rose_spent = _settle_after_send(False, "like-not-confirmed")

    # ── 8. Confirm the like actually went out (done above) ──
    # _settle_after_send raises SendLikeFailure when the card or the rose
    # sheet is still up. A like that didn't take leaves the card (and its
    # like button) on screen; the sheet adds a second "not sent" screen
    # that find_send_like alone cannot see, because its scrim dims the
    # card below threshold. (Per-template scores are in vision.py.)
    #
    # Confirmed — the feed has advanced to a new profile at the top.
    _feed_at_top = True
    return rose_spent


def _recover_from_dialog() -> None:
    """Attempt to dismiss a dialog by pressing back once."""
    adb.press_back()
    time.sleep(1.0)


def save_error_screenshot(context: str, png: bytes | None = None) -> str:
    """Save the current screen to debug/errors/ for post-mortem.

    `png` is a frame the caller already holds; without it one is captured.

    Returns the path to the saved screenshot."""
    errors_dir = config.DEBUG_DIR / "errors"
    errors_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    if png is None:
        png = adb.screenshot()
    path = errors_dir / f"{ts}_{context}.png"
    path.write_bytes(png)
    return str(path)


def _screenshot_or_none(context: str, png: bytes | None = None) -> str | None:
    """save_error_screenshot, but never let a capture failure replace the
    original error the caller is reporting."""
    try:
        return save_error_screenshot(context, png)
    except Exception as e:
        print(f"WARN: could not save error screenshot ({context}): {e!r}")
        return None


def save_debug(frames: list[bytes], decision, profile_idx: int) -> str | None:
    if not config.SAVE_DEBUG_FRAMES:
        return None
    bucket = "liked" if decision.decision == "like" else "skipped"
    bucket_dir = config.DEBUG_DIR / bucket
    bucket_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^a-z0-9]", "", (decision.name or "unknown").lower()) or "unknown"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = bucket_dir / f"{ts}_{profile_idx:02d}_{safe_name}"
    imgs = folder / "imgs"
    imgs.mkdir(parents=True, exist_ok=True)
    for i, png in enumerate(frames):
        (imgs / f"frame_{i:02d}.png").write_bytes(png)
    (folder / "decision.txt").write_text(

        f"name: {decision.name}\n"
        f"decision: {decision.decision}\n"
        f"fit_score: {decision.fit_score}\n"
        f"confidence: {decision.confidence}\n"
        f"reasoning: {decision.reasoning}\n"
        f"message: {decision.message}\n"
        f"drafted_message: {decision.drafted_message}\n"
        f"opener_anchor: {decision.opener_anchor}\n"
        f"message_archetype: {decision.message_archetype}\n"
        f"prompt_referenced: {decision.prompt_referenced}\n"
        f"dominant_factor: {decision.dominant_factor}\n"
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}\n"
    )
    return folder.name


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HingeAuto loop runner")
    p.add_argument(
        "--mode",
        default=None,
        help="Override config.ACTIVE_MODE for this run (one-shot). "
             "Must match a file under modes/<name>.py.",
    )
    return p.parse_args(argv)


def _acquire_run_lock(serial: str):
    """Take the one-run-at-a-time lock on `serial`, or None if another run
    already holds it.

    hinge-auto and bumble-auto drive the same phone, and an overrun puts
    both on it at once: the second one's launch_app() takes the foreground
    while the first keeps tapping, judging and liking whatever feed is now
    in front of it — which is how the two of them came to report the same
    women on 2026-10-01.

    flock rather than a look at what's on screen, so that nothing here has
    to know another bot exists, let alone which package it uses: a new
    automation joins the contention by running, without every existing bot
    being edited. The kernel releases the lock when its holder dies —
    SIGKILL and reboot included — so a wedged run can't lock out the next
    slot.

    Hold the returned handle for as long as the run lasts: closing it, or
    letting it be collected, releases the lock. Process exit is the only
    unlock, on purpose.
    """
    path = Path(tempfile.gettempdir()) / f"dating-bot-{serial}.lock"
    handle = open(path, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def main() -> int:
    global _feed_at_top

    args = _parse_args()
    load_dotenv()

    if args.mode:
        config.ACTIVE_MODE = args.mode
        config._apply_mode()

    serial = adb.check_device()
    print(f"Connected to: {serial}")

    # Held (never closed) for the rest of the process; see the docstring.
    run_lock = _acquire_run_lock(serial)
    if run_lock is None:
        msg = (f"Another run already has the phone ({serial}) — cancelling "
               f"this one without touching it. The next cron slot picks it "
               f"up.")
        print(f"\n{msg}")
        report.post_error(msg, title="🔒 Hinge Auto — Run Skipped (phone busy)")
        return 1
    age_band = (
        f"age {config.AGE_MIN}-{config.AGE_MAX}"
        if (config.AGE_MIN is not None or config.AGE_MAX is not None)
        else "no age gate"
    )
    print(f"Mode:     {config.MODE_NAME} ({age_band})")
    print(f"Run:      {'DRY RUN (no taps)' if config.DRY_RUN else 'LIVE (will tap)'}")
    if config.DRY_RUN_MESSAGE:
        print(f"Messages:  DRY RUN (generated but not sent)")
    session_like_cap = random.randint(
        config.SESSION_LIKE_MIN,
        config.MAX_LIKES_PER_SESSION,
    )
    print(f"Max likes: {session_like_cap} (randomized "
          f"{config.SESSION_LIKE_MIN}-{config.MAX_LIKES_PER_SESSION}), "
          f"max profiles: {config.MAX_PROFILES_PER_SESSION}")

    if session_like_cap == 0:
        print("Cap is 0 — skipping this session.")
        return 0

    # Wake screen and launch Hinge
    adb.wake_screen()
    adb.launch_app("co.hinge.app")
    # Tap the Discover tab to ensure we're on the main profile feed
    adb.tap(73, 1468)
    time.sleep(2)

    likes_sent = 0
    skips = 0
    roses_sent = 0
    profiles_seen = 0
    total_cost = 0.0
    total_seconds = 0.0
    fit_score_sum = 0
    fit_score_count = 0
    liked_profiles: list[dict] = []  # tracked for the webhook report
    last_frame0_hash: str | None = None
    duplicate_streak = 0
    dialog_streak = 0
    send_like_streak = 0
    send_like_screenshots: list[str] = []
    hit_like_cap = False
    # Set by every path that ends the run on a failure, then consumed once
    # at the bottom of main() to pick the webhook post. Reports used to go
    # out at the failure site and the loop would still fall through to the
    # success post, so an aborted run arrived as a red error *and* a green
    # "Run Complete".
    abort_reason: str | None = None
    abort_screenshots: list[str] = []

    while profiles_seen < config.MAX_PROFILES_PER_SESSION:
        profiles_seen += 1
        print(f"\n--- Profile {profiles_seen} ---")

        t0 = time.monotonic()
        try:
            frames = capture_profile()
        except RuntimeError as e:
            if "loading screen" in str(e):
                dialog_streak += 1
                print(f"\nAPP STUCK ON LOADING SCREEN (streak {dialog_streak})")
                dialog_ss = save_error_screenshot(f"loading-screen-{dialog_streak}")

                # Back press won't help an unloaded app — go straight to restart.
                if dialog_streak >= 3:
                    msg = (f"Loading screen persisted after {dialog_streak} "
                           f"app restarts.")
                    print(f"GIVING UP: {msg}")
                    abort_reason = msg
                    abort_screenshots = [dialog_ss]
                    break

                print(f"  Force-stopping + relaunching Hinge...")
                adb.force_stop_app("co.hinge.app")
                time.sleep(2)
                adb.wake_screen()
                adb.launch_app("co.hinge.app")
                adb.tap(73, 1468)  # Discover tab
                time.sleep(3)

                profiles_seen -= 1
                last_frame0_hash = None
                duplicate_streak = 0
                continue
            raise
        t_capture = time.monotonic() - t0
        print(f"Captured {len(frames)} frames")

        # If frame 0 is identical to the previous profile's frame 0, Hinge
        # didn't advance after our last action — force-skip rather than
        # burning another ~$0.035 re-judging the same person. Escalate the
        # delay if we keep duplicating, in case Hinge needs a beat to
        # recover from an "out of likes" / popup state.
        #
        # Hash a cropped region of frame 0 (excluding status bar at top and
        # nav bar at bottom). The status-bar clock ticks every minute and
        # the nav-bar can show transient badges; both make raw-bytes md5
        # diverge across iterations even when the *profile* is identical,
        # which silently breaks duplicate detection. Cropping isolates the
        # part of the screen that actually identifies the profile.
        frame0_hash = _profile_region_hash(frames[0])
        if frame0_hash == last_frame0_hash:
            duplicate_streak += 1
            print(f"DUPLICATE: frame 0 matches previous profile "
                  f"(streak {duplicate_streak}). Force-skipping without judge.")
            try:
                do_skip()
            except Exception as e:
                print(f"Skip failed during duplicate recovery: {e!r}")
            time.sleep(min(2 + duplicate_streak * 2, 15))
            continue
        last_frame0_hash = frame0_hash
        duplicate_streak = 0

        # Random profile dwell — simulates actually reading
        dwell = random.uniform(2, 8)
        print(f"  Reading profile for {dwell:.1f}s...")
        time.sleep(dwell)

        t1 = time.monotonic()
        decision = None
        fatal_error = None
        network_error = None
        unavailable_error = None
        for attempt in range(3):
            try:
                # Every attempt runs under a wall-clock ceiling. The
                # backend's own httpx timeout is per read *operation*, so a
                # provider that holds the socket open while it queues the
                # request resets it on every byte and never trips it — a 180s
                # httpx timeout did not stop a 947s DeepSeek call. See
                # judge_common.call_with_deadline.
                decision = call_with_deadline(
                    lambda: judge(frames), config.JUDGE_DEADLINE_S
                )
                break
            except Exception as e:
                err = repr(e)
                print(f"Judge attempt {attempt + 1}/3 failed: {e}")
                # Halt on errors that won't recover with a retry — burning
                # through Hinge swipes blind (force-skipping every profile
                # without a real decision) eats the daily quota and looks
                # robotic to Hinge. Classification keys off HTTP status and
                # walks the exception chain, so it covers every backend
                # instead of matching Anthropic's wording only.
                if is_fatal_judge_error(e):
                    fatal_error = err
                    break
                # The provider says it never ran the request at all — it shed
                # load, or held the connection past JUDGE_DEADLINE_S. Unlike
                # a network blip this doesn't get retried: the request sat in
                # the provider's queue, so another attempt buys another full
                # wait, and on 2026-10-01 three of them cost ~47 minutes and
                # ran a 20-minute session to 74. Usually clears by the next
                # cron slot; the profile is left unswiped for it.
                if is_judge_unavailable(e):
                    unavailable_error = err
                    break
                # A network error is different in kind: it usually clears on
                # its own, so it doesn't cut the attempts short. But if it
                # outlasts all three, the internet is down — and skipping is
                # the wrong recovery, because the judge never saw this
                # profile and the skip would spend it for nothing.
                if is_network_error(e):
                    network_error = err
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
        if fatal_error is not None:
            print(f"\nFATAL judge error — halting loop instead of burning "
                  f"Hinge swipes:\n  {fatal_error}")
            abort_reason = ("Fatal judge error — halted instead of burning "
                            f"Hinge swipes.\n{fatal_error}")
            break
        if unavailable_error is not None:
            print(f"\nJUDGE UNAVAILABLE — the provider never ran the request; "
                  f"ending the run. Nothing was skipped; the next cron slot "
                  f"resumes from this profile.\n  {unavailable_error}")
            abort_reason = ("Judge provider unavailable — it shed load or "
                            "stopped answering, so no profile was judged. "
                            "Nothing was skipped; the next cron slot resumes "
                            f"from this profile.\n{unavailable_error}")
            break
        t_judge = time.monotonic() - t1
        if decision is None:
            if network_error is not None:
                print(f"\nNETWORK ERROR on all 3 judge attempts — ending the "
                      f"run. Nothing was skipped; the next cron slot resumes "
                      f"from this profile.\n  {network_error}")
                abort_reason = ("Network error on all 3 judge attempts — "
                                "nothing was skipped; the next cron slot "
                                f"resumes from this profile.\n{network_error}")
                break
            print("Judge failed 3 times — skipping this profile to keep the loop alive.")
            do_skip()
            continue

        print(f"Name:     {decision.name}")

        # ── Pickiness gate: like iff fit_score >= FIT_SCORE_MIN ──
        # The model's decision is advisory; the run's threshold is authoritative.
        # NOT_A_PROFILE is preserved by apply_fit_threshold for recovery below.
        decision = apply_fit_threshold(decision)

        # dominant_factor is symmetric now — it names what drove the score in
        # either direction, so a like shows it alongside the opener archetype.
        label = decision.dominant_factor
        if decision.decision == "like" and decision.message_archetype:
            label = f"{decision.message_archetype} · {label}"
        print(f"Decision: {decision.decision} ({decision.confidence}) [{label}]")
        print(f"Fit:      {decision.fit_score}/100 (threshold {config.FIT_SCORE_MIN})")
        print(f"Reason:   {decision.reasoning}")
        if decision.message:
            print(f"Message:  {decision.message}")
        if decision.drafted_message and decision.drafted_message != decision.message:
            print(f"Drafted:  {decision.drafted_message}  (not sent)")

        # ── Dialog / popup detection & recovery ──
        if decision.decision == "NOT_A_PROFILE":
            dialog_streak += 1
            print(f"\nDIALOG DETECTED (streak {dialog_streak}): {decision.reasoning[:120]}")
            dialog_ss = save_error_screenshot(f"dialog-streak-{dialog_streak}")

            # A rose purchase page left over from an interrupted rose
            # attempt (or a pill tap that found an empty balance): back
            # out, send the interrupted like as an ordinary like, and let
            # the feed advance. Never tap through the page itself.
            try:
                purchase_resolved = _recover_from_rose_purchase("dialog")
            except RosePurchaseStuck as e:
                msg = (f"Rose purchase page is stuck during dialog recovery: "
                       f"Back does not clear it and every control on it buys "
                       f"something. Halting without tapping.\n{e}")
                print(f"GIVING UP: {msg}")
                abort_reason = msg
                abort_screenshots = ([e.screenshot_path]
                                     if e.screenshot_path else [dialog_ss])
                break
            if purchase_resolved:
                print("  Rose purchase page resolved — feed advancing.")
                profiles_seen -= 1
                last_frame0_hash = None
                duplicate_streak = 0
                dialog_streak = 0
                _feed_at_top = True
                continue

            # A "Send a Rose instead?" sheet left over from an interrupted
            # like: the like underneath never went out, so this is not a
            # dialog to back out of — finish it with the non-spending
            # option and let the feed advance. This is the crash/restart
            # safety net for the exact bug the sheet handler fixes.
            sheet = vision.find_rose_sheet(adb.screenshot())
            if sheet is not None and sheet.send_like_anyway is not None:
                print("  Interrupted like found: tapping 'Send Like anyway'.")
                adb.tap(*sheet.send_like_anyway)
                adb.jitter_sleep("after_like_sent")
                png = adb.screenshot()
                if (vision.find_rose_sheet(png) is None
                        and vision.find_send_like(png) is None):
                    print("  Interrupted like resolved — feed advancing.")
                    profiles_seen -= 1
                    last_frame0_hash = None
                    duplicate_streak = 0
                    dialog_streak = 0
                    _feed_at_top = True
                    continue
                print("  Sheet did not clear — continuing with dialog recovery.")

            if dialog_streak == 1:
                # ── Tier 1: press back to dismiss ──
                print(f"  Tier 1: pressing back to dismiss...")
                _recover_from_dialog()
            elif dialog_streak == 2:
                # ── Tier 2: force-stop + relaunch ──
                print(f"  Tier 2: force-stopping + relaunching Hinge...")
                adb.force_stop_app("co.hinge.app")
                time.sleep(1.5)
                adb.wake_screen()
                adb.launch_app("co.hinge.app")
                adb.tap(73, 1468)  # Discover tab
                time.sleep(3)
            else:
                # ── Tier 3: give up ──
                msg = (f"Dialog recovery failed after back button + app restart. "
                       f"Last reason: {decision.reasoning[:200]}")
                print(f"TIER 3: {msg}")
                abort_reason = msg
                abort_screenshots = [dialog_ss]
                break

            profiles_seen -= 1
            last_frame0_hash = None
            duplicate_streak = 0
            continue
        else:
            dialog_streak = 0

        fit_score_sum += decision.fit_score
        fit_score_count += 1

        folder_name = save_debug(frames, decision, profiles_seen)

        t2 = time.monotonic()
        rose_spent = False
        if decision.decision == "like":
            # Rose policy lives here, next to the like/skip decision: spend
            # one only on the top slice of likes, and never more than the
            # session cap even if the whole batch scores above the
            # threshold. do_like only carries the intent out.
            prefer_rose = (
                config.SEND_ROSES
                and decision.fit_score >= config.ROSE_FIT_THRESHOLD
                and roses_sent < config.MAX_ROSES_PER_SESSION
            )
            if (config.SEND_ROSES
                    and decision.fit_score >= config.ROSE_FIT_THRESHOLD):
                if roses_sent < config.MAX_ROSES_PER_SESSION:
                    print(f"🌹 Rose wanted: fit {decision.fit_score} >= "
                          f"{config.ROSE_FIT_THRESHOLD} — specifically "
                          f"trying to send a rose with this like.")
                else:
                    print(f"Rose-worthy fit {decision.fit_score}, but the "
                          f"session rose cap ({roses_sent}/"
                          f"{config.MAX_ROSES_PER_SESSION}) is spent — "
                          f"sending a normal like.")
            try:
                rose_spent = do_like(decision.message, prefer_rose=prefer_rose)
                roses_sent += rose_spent
                if rose_spent:
                    print("🌹 Rose specifically wanted — sent with this like.")
                # A sent like proves the whole compose flow works again.
                send_like_streak = 0
                send_like_screenshots = []
                liked_profiles.append({
                    "name": decision.name,
                    "message": decision.message,
                    "fit_score": decision.fit_score,
                    "index": profiles_seen,
                    "folder": folder_name,
                    "rose": rose_spent,
                })
                likes_sent += 1
                if likes_sent >= session_like_cap:
                    print(f"Hit max likes cap ({session_like_cap}). Stopping.")
                    # Don't break here — the profile still needs its log
                    # record + cost tally below (metrics.log_profile).
                    hit_like_cap = True
            except FeedAlreadyAdvanced as e:
                # _dismiss_compose_card_if_visible already tapped skip and
                # saved its own screenshot, so this profile is gone. The
                # handler below would tap skip again, spending the next
                # profile as well — one no judge ever scored.
                print(f"do_like aborted: {e} — feed already advanced, "
                      f"not skipping again.")
                skips += 1
            except SendLikeFailure as e:
                send_like_streak += 1
                if e.screenshot_path:
                    send_like_screenshots.append(e.screenshot_path)
                print(f"Send Like failed ({send_like_streak}/"
                      f"{SEND_LIKE_STREAK_LIMIT}): {e}")
                if isinstance(e, RosePurchaseStuck):
                    msg = (f"Rose purchase page is stuck: Back does not clear "
                           f"it, and every control on it buys something. "
                           f"Halting without tapping.\n{e}")
                    print(f"GIVING UP: {msg}")
                    abort_reason = msg
                    abort_screenshots = list(send_like_screenshots)
                    break
                if isinstance(e, RoseSheetStuck):
                    msg = (f"Rose sheet is stuck: it covers the composer and "
                           f"neither a resolution tap nor Back clears it. "
                           f"Halting without skipping — the skip X cannot "
                           f"reach through the sheet.\n{e}")
                    print(f"GIVING UP: {msg}")
                    abort_reason = msg
                    abort_screenshots = list(send_like_screenshots)
                    break
                if send_like_streak >= SEND_LIKE_STREAK_LIMIT:
                    msg = (f"Send Like failed {send_like_streak} consecutive "
                           f"like attempts — halted instead of burning more "
                           f"profiles.\nLast error: {e}")
                    print(f"GIVING UP: {msg}")
                    abort_reason = msg
                    abort_screenshots = list(send_like_screenshots)
                    break
                print("  Recovering by skipping this profile.")
                try:
                    if _skip_unless_rose_sheet(f"send-like-{profiles_seen}"):
                        skips += 1
                except (RoseSheetStuck, RosePurchaseStuck) as e2:
                    if e2.screenshot_path:
                        send_like_screenshots.append(e2.screenshot_path)
                    msg = (f"A rose UI screen is stuck during Send Like "
                           f"recovery — Back will not clear it and skip "
                           f"cannot reach the feed. Halting.\n{e2}")
                    print(f"GIVING UP: {msg}")
                    abort_reason = msg
                    abort_screenshots = list(send_like_screenshots)
                    break
                except Exception as e2:
                    print(f"do_skip recovery also failed: {e2!r} — "
                          f"loop will retry next iter.")
            except Exception as e:
                save_error_screenshot(f"do-like-failed-{profiles_seen}")
                print(f"do_like failed: {e!r} — recovering by skipping this profile.")
                try:
                    if _skip_unless_rose_sheet(f"do-like-failed-{profiles_seen}"):
                        skips += 1
                except (RoseSheetStuck, RosePurchaseStuck) as e2:
                    msg = (f"A rose UI screen is stuck during error recovery — "
                           f"halting without skipping.\n{e2}")
                    print(f"GIVING UP: {msg}")
                    abort_reason = msg
                    abort_screenshots = [e2.screenshot_path] if e2.screenshot_path else []
                    break
                except Exception as e2:
                    print(f"do_skip recovery also failed: {e2!r} — loop will retry next iter.")
        else:
            do_skip()
            skips += 1
        t_act = time.monotonic() - t2

        timing = {
            "capture_seconds": round(t_capture, 2),
            "judge_seconds":   round(t_judge, 2),
            "act_seconds":     round(t_act, 2),
            "total_seconds":   round(t_capture + t_judge + t_act, 2),
        }
        metrics.log_profile(profiles_seen, decision, timing,
                            rose_sent=rose_spent)
        total_cost += metrics.estimated_cost(decision.usage)
        total_seconds += timing["total_seconds"]
        avg_fit = (fit_score_sum / fit_score_count) if fit_score_count else 0
        metrics.print_running_totals(
            profiles_seen, likes_sent, skips, total_cost, total_seconds,
            avg_fit_score=avg_fit,
        )

        if hit_like_cap:
            break

    avg_fit = (fit_score_sum / fit_score_count) if fit_score_count else 0
    print(f"\nDone. {likes_sent} likes sent across {profiles_seen} profiles "
          f"(avg fit {avg_fit:.0f}/100).")

    # Exactly one post per run — error or success, never both.
    if abort_reason is not None:
        report.post_error(abort_reason, profiles_seen, likes_sent, skips,
                          roses_sent=roses_sent,
                          screenshot_paths=abort_screenshots)
    else:
        report.post_run(likes_sent, profiles_seen, skips, total_cost,
                        total_seconds, liked_profiles, avg_fit_score=avg_fit)

    # Cleanup: force-stop Hinge so next run starts fresh regardless of app state,
    # then turn screen off.
    adb.force_stop_app("co.hinge.app")
    adb.turn_screen_off()

    # An aborted run is not a success. This used to return 0 either way, so
    # cron.log read "Done (exit 0)" for a run that died halfway — the exit
    # code was the one signal the webhook couldn't stand in for.
    return 1 if abort_reason is not None else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # Nothing inside main() reports a crash: its handlers cover judge
        # failures and stuck screens, but an exception anywhere else (an
        # adb failure, a typo on a rarely-taken branch) used to die with a
        # traceback in cron.log and no webhook post at all. That was the
        # one failure class a channel set to "Only @mentions" would never
        # surface, which is exactly backwards.
        #
        # An import-time error in this file still can't reach here — the
        # module body fails before this block runs — so a missing module
        # or bad dependency stays silent. Worth a launcher if that bites.
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        report.post_crash(tb)
        sys.exit(1)
