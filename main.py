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
    print(f"Scrolling back to top ({swipes} swipes)...")
    for _ in range(swipes):
        adb.scroll_up()
        adb.jitter_sleep("after_scroll")
    print(f"Scrolling stopped — at top of profile.")


def do_skip() -> None:
    """Tap the X to advance to the next profile. Always taps (even in dry run);
    advancing is needed for the loop to see new profiles."""
    x, y = config.COORDS["skip_button"]
    adb.tap(x, y)
    adb.jitter_sleep("after_skip")


def _dismiss_compose_card_if_visible() -> None:
    """Tap the compose card close button if the card is showing.

    After a failed do_like, the compose card may still be open, which
    blocks find_first_heart on the next profile. Checks for the pink
    Send Like text before tapping.
    """
    ss = adb.screenshot()
    if vision.find_send_like(ss) is not None:
        cx, cy = config.COORDS["compose_close"]
        adb.tap(cx, cy)
        adb.jitter_sleep("after_tap")


def do_like(message: str = "") -> None:
    """Flow: scroll to top → tap heart → type message → dismiss keyboard → find + click Send Like.
    """
    if config.DRY_RUN:
        do_skip()
        return
    _dismiss_compose_card_if_visible()
    scroll_back_to_top()
    # Let the profile settle after scrolling — momentum scrolling can
    # leave the view mid-animation if we screenshot too fast.
    time.sleep(0.5)
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
        # Comment input is roughly center-x, a fixed offset above the
        # expected Send Like position (which sits below the heart).
        comment_x = int(config.SCREEN_WIDTH / 2)
        comment_y = heart_xy[1] - 66
        adb.tap(comment_x, comment_y)
        adb.jitter_sleep("after_tap")
        # Estimate Send Like Y to crop the text-field pixel check.
        est_send_y = heart_xy[1] + 105
        empty_pixels = vision.comment_field_text_pixels(
            adb.screenshot(), (comment_x, est_send_y))
        adb.input_text(message)
        deadline = time.monotonic() + 15
        target_pixels = empty_pixels + max(150, 20 * len(message))
        while time.monotonic() < deadline:
            time.sleep(1.0)
            current = vision.comment_field_text_pixels(
                adb.screenshot(), (comment_x, est_send_y))
            if current >= target_pixels:
                break
        else:
            print(f"WARN: typed text didn't reach expected pixel density "
                  f"(have {current}, want {target_pixels}) — sending anyway.")

    # ── 6. Dismiss keyboard ──
    if message and not config.DRY_RUN_MESSAGE:
        adb.dismiss_keyboard_if_visible()

    # ── 7. Find and click Send Like ──
    # Dismiss keyboard first in case Hinge auto-focused the comment field
    # and the keyboard is covering Send Like.
    adb.dismiss_keyboard_if_visible()
    send_xy = vision.find_send_like(adb.screenshot())
    if send_xy is None:
        save_error_screenshot("send-like-not-found")
        raise RuntimeError("vision: couldn't find Send Like after heart tap")
    print(f"Tapping Send Like at {send_xy} — sending like! 🎯")
    adb.tap(*send_xy)
    adb.jitter_sleep("after_like_sent")

    # Clean up any lingering keyboard so it doesn't carry over to the
    # next profile and block heart / Send Like detection.
    if adb.dismiss_keyboard_if_visible():
        print("Keyboard was still open after like — dismissed.")


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
