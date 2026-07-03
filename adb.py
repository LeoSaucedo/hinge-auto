"""Thin wrapper around `adb` commands."""

import random
import shlex
import subprocess
import time

import config


def _run(args: list[str], capture: bool = False) -> bytes | None:
    cmd = ["adb"] + args
    if capture:
        result = subprocess.run(cmd, capture_output=True, check=True)
        return result.stdout
    subprocess.run(cmd, check=True)
    return None


def check_device() -> str:
    """Raise if no device is connected. Return the serial of the first device."""
    out = _run(["devices"], capture=True).decode()
    lines = [l for l in out.splitlines()[1:] if l.strip() and "\tdevice" in l]
    if not lines:
        raise RuntimeError(
            "No ADB device found. Start your emulator and run `adb devices`."
        )
    return lines[0].split("\t")[0]


def screenshot() -> bytes:
    """Return PNG bytes of the current screen."""
    return _run(["exec-out", "screencap", "-p"], capture=True)


def tap(x: int, y: int, jitter: int = 10) -> None:
    """Tap at (x, y) with ±`jitter` px randomness."""
    x += random.randint(-jitter, jitter)
    y += random.randint(-jitter, jitter)
    _run(["shell", "input", "tap", str(x), str(y)])


def input_text(text: str) -> None:
    """Type `text` into the currently focused field via `adb shell input text`.

    Single-quotes the payload with shlex so $, `, ", and spaces survive the
    device shell. Caller should keep text plain ASCII — emoji and the chars
    \\ " $ ` should be filtered upstream (the model is instructed to avoid
    them) since `input text` itself only sends keyevents.
    """
    quoted = shlex.quote(text)
    try:
        _run(["shell", f"input text {quoted}"])
    except subprocess.CalledProcessError as e:
        # Some devices/IME implementations crash `input text` with a
        # NullPointerException (Moto e20 / Gboard). When this happens
        # the keyboard is in a stuck state — dismiss it so the rest
        # of the pipeline can recover.
        print(f"adb: input_text failed (device IME crash) — dismissing"
              f" keyboard\n    text={text!r}")
        try:
            _run(["shell", "input", "keyevent", "4"])
        except subprocess.CalledProcessError:
            pass


def swipe(x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
    _run(["shell", "input", "swipe",
          str(x1), str(y1), str(x2), str(y2), str(duration_ms)])


def scroll_down() -> None:
    c = config.COORDS
    # Vary scroll distance 80-120% to avoid identical gestures
    scale = random.uniform(0.8, 1.2)
    sx = int(c["scroll_from"][0])
    sy = int(c["scroll_from"][1])
    ex = int(c["scroll_to"][0])
    ey = int(c["scroll_to"][1])
    dy = int((sy - ey) * scale)
    swipe(sx, sy, sx, sy - dy, c["scroll_duration_ms"])


def scroll_up() -> None:
    c = config.COORDS
    scale = random.uniform(0.8, 1.2)
    sx = int(c["scroll_to"][0])
    sy = int(c["scroll_to"][1])
    ex = int(c["scroll_from"][0])
    ey = int(c["scroll_from"][1])
    dy = int((ey - sy) * scale)
    swipe(sx, sy, sx, sy + dy, c["scroll_duration_ms"])


def jitter_sleep(key: str) -> None:
    lo, hi = config.DELAYS[key]
    time.sleep(random.uniform(lo, hi))


def is_screen_awake() -> bool:
    """Check if the device screen is on."""
    out = _run(["shell", "dumpsys", "power"], capture=True).decode()
    return "mWakefulness=Awake" in out


def wake_screen() -> None:
    """Turn the screen on if it's off."""
    if not is_screen_awake():
        _run(["shell", "input", "keyevent", "26"])
        time.sleep(1.0)


def force_stop_app(package: str) -> None:
    """Force-stop the app so the next launch starts fresh."""
    _run(["shell", "am", "force-stop", package])
    time.sleep(0.5)


def dismiss_keyboard_if_visible() -> bool:
    """Dismiss the soft keyboard if it's currently visible.

    Taps in the compose card header area (above the text field, below
    the status bar) to dismiss the IME via focus loss. Avoids sending
    KEYCODE_BACK, which on some Hinge versions / Android builds also
    closes the compose card.

    Safe to call when the keyboard isn't up — does nothing and returns False.
    """
    out = _run(["shell", "dumpsys", "input_method"], capture=True)
    if out and b"mInputShown=true" in out:
        # Tap just below the status bar, inside the app area, to
        # dismiss the keyboard via tap-on-background. This is the
        # compose card photo header area — tapping here causes
        # focus loss on the text field without KEYCODE_BACK side effects.
        _run(["shell", "input", "tap", "360", "200"])
        time.sleep(0.8)
        # Check if keyboard is actually gone
        out2 = _run(["shell", "dumpsys", "input_method"], capture=True)
        keyboard_still_visible = out2 and b"mInputShown=true" in out2
        if keyboard_still_visible:
            # If tap didn't work, try Android's internal IME hide
            _run(["shell", "input", "keyevent", "111"])  # KEYCODE_ESCAPE
            time.sleep(0.5)
        return not keyboard_still_visible
    return False


def launch_app(package: str, activity: str = ".ui.AppActivity") -> None:
    """Launch an Android app by package name. Force-stops first for a clean slate.

    Uses `am start` instead of `monkey` because the latter can hang over
    TCP / Tailscale ADB connections (subprocess.run waits forever).
    Falls back to monkey if am start fails.
    """
    force_stop_app(package)
    intent = f"{package}/{activity}"
    try:
        _run(["shell", "am", "start", "-n", intent])
    except subprocess.CalledProcessError:
        _run(["shell", "monkey", "-p", package, "1"])
    time.sleep(3.0)


def go_home() -> None:
    """Send the home key."""
    _run(["shell", "input", "keyevent", "3"])
    time.sleep(0.5)


def turn_screen_off() -> None:
    """Turn the screen off."""
    _run(["shell", "input", "keyevent", "26"])
