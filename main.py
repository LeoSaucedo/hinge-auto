"""HingeAuto orchestrator.

Loop: capture profile frames -> ask Claude -> tap like or skip -> repeat.
"""

import argparse
import hashlib
import random
import re
import sys
import time
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
from judge_common import load_backend

judge = load_backend().judge


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
    for _ in range(swipes):
        adb.scroll_up()
        adb.jitter_sleep("after_scroll")


def do_skip() -> None:
    """Tap the X to advance to the next profile. Always taps (even in dry run);
    advancing is needed for the loop to see new profiles."""
    x, y = config.COORDS["skip_button"]
    adb.tap(x, y)
    adb.jitter_sleep("after_skip")


def _dismiss_compose_card_if_visible() -> None:
    """Dismiss the compose card if it's showing.

    After a failed do_like, the compose card (possibly with keyboard still
    visible) may be left on screen. This blocks find_first_heart on the
    next profile. Checks for the compose card via:
    1. find_send_like() — detects visible Send Like button (pink or grey)
    2. Keyboard visibility — if the keyboard is up, the compose card is
       likely open (Hinge only shows the keyboard during compose).
       Dismiss keyboard first, then close the card.
    """
    ss = adb.screenshot()
    card_visible = vision.find_send_like(ss) is not None
    if not card_visible and adb.dismiss_keyboard_if_visible():
        # Keyboard was covering the button bar. After dismiss, check again.
        adb.jitter_sleep("after_tap")
        ss = adb.screenshot()
        card_visible = vision.find_send_like(ss) is not None
    if card_visible:
        cx, cy = config.COORDS["compose_close"]
        adb.tap(cx, cy)
        adb.jitter_sleep("after_tap")


def do_like(message: str = "") -> None:
    """In live mode: scroll to top, tap heart, type message (if any), tap Send Like.
    In dry run: advance by skipping (so we never send an actual like).

    Send Like / comment input positions are found at tap-time via vision —
    the compose card anchors to whichever heart was tapped and shifts per
    profile, so static COORDS don't survive across profiles.

    CRITICAL: When the compose card opens, Hinge auto-focuses the comment
    input. Dismissing the keyboard causes the button bar to disappear
    entirely (buttons only render when the input has focus or text). So:
    - If there's a message: TYPE FIRST into the auto-focused input, THEN
      find the active (pink) Send Like button.
    - If no message: try to find the disabled (grey) Send Like button;
      fall back to keyboard-dismiss + retry.
    """
    if config.DRY_RUN:
        do_skip()
        return
    # Try to close any leftover compose card from a previous failed
    # like before proceeding. If the card is open, find_first_heart
    # will fail and we'd skip the profile unnecessarily.
    _dismiss_compose_card_if_visible()
    # Scroll back to the top before tapping a heart. The compose box
    # anchors to the tapped element and extends DOWNWARD — if we tap
    # a heart that's already low on screen (which it is after capture),
    # Send Like ends up off-screen and undetectable. Worth the ~14s.
    scroll_back_to_top()
    heart_xy = vision.find_first_heart(adb.screenshot())
    if heart_xy is None:
        save_error_screenshot("heart-not-found")
        raise RuntimeError("vision: couldn't find photo-1 heart after scroll-back")
    adb.tap(*heart_xy)
    adb.jitter_sleep("after_tap")

    if config.DRY_RUN_MESSAGE and message:
        print(f"DRY_RUN_MESSAGE: would send '{message}' — sending like without it.")

    if message and not config.DRY_RUN_MESSAGE:
        # ---------- Message path: type first, then find pink Send Like ----------
        # The comment input is auto-focused after heart tap. Don't dismiss
        # the keyboard first — on the current Hinge version, that removes
        # the button bar entirely. Type directly into the auto-focused field.
        time.sleep(0.8)  # Let keyboard fully render

        adb.input_text(message)
        time.sleep(1.5)  # Let text finish typing

        # Dismiss keyboard — safe now because text was entered, so the Send
        # Like button stays active (pink) even after losing focus.
        adb.dismiss_keyboard_if_visible()
        adb.jitter_sleep("after_tap")

        # Find active (pink) Send Like. Re-find to handle Y shift from
        # multi-line text wrapping.
        send_xy = vision.find_send_like(adb.screenshot())
        if send_xy is None:
            save_error_screenshot("send-like-not-found")
            raise RuntimeError("vision: couldn't find Send Like after typing message")
    else:
        # ---------- No-message path: find disabled/grey Send Like ----------
        send_xy = vision.find_send_like(adb.screenshot())
        if send_xy is None and adb.dismiss_keyboard_if_visible():
            # Keyboard might be covering Send Like. Dismiss to check.
            # The button is grey/disabled but should remain visible
            # as long as the compose card stays open.
            print("Keyboard was blocking Send Like — dismissed and retrying.")
            save_error_screenshot("post-dismiss-send-like-search")
            send_xy = vision.find_send_like(adb.screenshot())
        if send_xy is None:
            save_error_screenshot("send-like-not-found")
            raise RuntimeError("vision: couldn't find Send Like after heart tap")

    adb.tap(*send_xy)
    adb.jitter_sleep("after_like_sent")


def save_error_screenshot(context: str) -> None:
    """Capture the current screen and save to debug/errors/ for post-mortem."""
    errors_dir = config.DEBUG_DIR / "errors"
    errors_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    png = adb.screenshot()
    (errors_dir / f"{ts}_{context}.png").write_bytes(png)


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
        f"confidence: {decision.confidence}\n"
        f"reasoning: {decision.reasoning}\n"
        f"message: {decision.message}\n"
        f"message_archetype: {decision.message_archetype}\n"
        f"prompt_referenced: {decision.prompt_referenced}\n"
        f"skip_reason: {decision.skip_reason}\n"
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
    p.add_argument(
        "--set-filters",
        action="store_true",
        help="Before looping, drive the in-app age slider to match the "
             "active mode's AGE_MIN/AGE_MAX. Requires filter_coords.json "
             "(see calibrate_filters.py).",
    )
    p.add_argument(
        "--location",
        default=None,
        help="Before looping, change Hinge's 'My neighborhood' to the "
             "named city (resolved via locations.json). Orthogonal to "
             "--mode. Requires location_coords.json + Hinge+/X for "
             "out-of-area changes.",
    )
    p.add_argument(
        "--rotate",
        default=None,
        help="Named rotation path from locations.json _rotations (e.g. "
             "'atl'). When Hinge's 'You've seen everyone' screen appears, "
             "advance to the next city in the rotation. Loops back to "
             "start when exhausted.",
    )
    return p.parse_args(argv)


def main() -> int:
    args = _parse_args()
    load_dotenv()

    if args.mode:
        config.ACTIVE_MODE = args.mode
        config._apply_mode()

    serial = adb.check_device()
    print(f"Connected to: {serial}")
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

    if args.set_filters:
        if config.AGE_MIN is None and config.AGE_MAX is None:
            print("--set-filters requested but active mode has no age range; "
                  "skipping in-app filter step.")
        else:
            import filters
            print(f"Setting in-app age filter to {config.AGE_MIN}-{config.AGE_MAX}...")
            filters.set_age_range(config.AGE_MIN, config.AGE_MAX)

    if args.location:
        import locations
        resolved = locations.resolve(args.location)
        print(f"Setting location to '{args.location}' (search: '{resolved}')...")
        locations.set_location(args.location)

    rotation_list: list[str] | None = None
    rotation_idx: int = 0
    if args.rotate:
        import locations
        rotation_list = locations.get_rotation(args.rotate)
        # If --location was also set, start the rotation at that location if
        # it's in the list; otherwise prepend it (don't replace the rotation).
        if args.location and args.location in rotation_list:
            rotation_idx = rotation_list.index(args.location)
        print(f"Rotation '{args.rotate}': {rotation_list} (starting at "
              f"index {rotation_idx} = '{rotation_list[rotation_idx]}')")

    # Wake screen and launch Hinge
    adb.wake_screen()
    adb.launch_app("co.hinge.app")
    # Tap the Discover tab to ensure we're on the main profile feed
    adb.tap(73, 1468)
    time.sleep(2)

    likes_sent = 0
    skips = 0
    profiles_seen = 0
    total_cost = 0.0
    total_seconds = 0.0
    liked_profiles: list[dict] = []  # tracked for the webhook report
    last_frame0_hash: str | None = None
    duplicate_streak = 0

    while profiles_seen < config.MAX_PROFILES_PER_SESSION:
        profiles_seen += 1
        print(f"\n--- Profile {profiles_seen} ---")

        # Dismiss any leftover compose card from a previous failed do_like
        # before capturing. If the card is open (possibly with keyboard),
        # capture_profile will grab frames of the keyboard overlay instead
        # of the actual profile, making the judge skip everything as
        # low_effort.
        _dismiss_compose_card_if_visible()

        t0 = time.monotonic()
        frames = capture_profile()
        t_capture = time.monotonic() - t0
        print(f"Captured {len(frames)} frames")

        # Out-of-candidates detection: if frame 0 shows the "You've seen
        # everyone for now" screen, we're stuck. If --rotate is on, advance
        # to the next city; otherwise just break since further iteration
        # would just force-skip nothing.
        if rotation_list is not None:
            import locations
            if locations.is_out_of_candidates(frames[0]):
                rotation_idx = (rotation_idx + 1) % len(rotation_list)
                next_city = rotation_list[rotation_idx]
                print(f"OUT OF CANDIDATES — rotating to '{next_city}' "
                      f"(index {rotation_idx}/{len(rotation_list) - 1}).")
                try:
                    locations.set_location(next_city)
                except Exception as e:
                    print(f"set_location failed: {e!r} — continuing in place.")
                last_frame0_hash = None
                duplicate_streak = 0
                profiles_seen -= 1  # don't count the empty-state capture
                continue

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
        for attempt in range(3):
            try:
                decision = judge(frames)
                break
            except Exception as e:
                err = repr(e)
                print(f"Judge attempt {attempt + 1}/3 failed: {e}")
                # Halt on errors that won't recover with a retry — burning
                # through Hinge swipes blind (force-skipping every profile
                # without a real decision) eats the daily quota and looks
                # robotic to Hinge. Saw this once when the Anthropic credit
                # balance hit zero mid-run: 124 profiles got blindly skipped
                # before we noticed.
                if any(s in err for s in (
                    "credit balance is too low",
                    "authentication_error",
                    "invalid_api_key",
                    "permission_error",
                )):
                    fatal_error = err
                    break
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
        if fatal_error is not None:
            print(f"\nFATAL judge error — halting loop instead of burning "
                  f"Hinge swipes:\n  {fatal_error}")
            break
        t_judge = time.monotonic() - t1
        if decision is None:
            print("Judge failed 3 times — skipping this profile to keep the loop alive.")
            do_skip()
            continue

        print(f"Name:     {decision.name}")
        print(f"Decision: {decision.decision} ({decision.confidence}) "
              f"[{decision.skip_reason if decision.decision == 'skip' else decision.message_archetype}]")
        print(f"Reason:   {decision.reasoning}")
        if decision.message:
            print(f"Message:  {decision.message}")
        folder_name = save_debug(frames, decision, profiles_seen)

        t2 = time.monotonic()
        if decision.decision == "like":
            try:
                do_like(decision.message)
                liked_profiles.append({
                    "name": decision.name,
                    "message": decision.message,
                    "index": profiles_seen,
                    "folder": folder_name,
                })
                likes_sent += 1
                if likes_sent >= session_like_cap:
                    print(f"Hit max likes cap ({session_like_cap}). Stopping.")
                    break
            except Exception as e:
                save_error_screenshot(f"do-like-failed-{profiles_seen}")
                print(f"do_like failed: {e!r} — recovering by skipping this profile.")
                try:
                    do_skip()
                except Exception as e2:
                    print(f"do_skip recovery also failed: {e2!r} — loop will retry next iter.")
                skips += 1
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
        metrics.log_profile(profiles_seen, decision, timing)
        total_cost += metrics.estimated_cost(decision.usage)
        total_seconds += timing["total_seconds"]
        metrics.print_running_totals(
            profiles_seen, likes_sent, skips, total_cost, total_seconds,
        )

    print(f"\nDone. {likes_sent} likes sent across {profiles_seen} profiles.")

    # Post-run report to Discord webhook (if configured)
    report.post_run(likes_sent, profiles_seen, skips, total_cost, total_seconds,
                    liked_profiles)

    # Cleanup: force-stop Hinge so next run starts fresh regardless of app state,
    # then turn screen off.
    adb.force_stop_app("co.hinge.app")
    adb.turn_screen_off()
    return 0


if __name__ == "__main__":
    sys.exit(main())
