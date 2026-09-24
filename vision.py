"""Image-based detection for UI elements whose position varies per profile.

Uses OpenCV template matching for reliable detection immune to profile
photo colors and Hinge UI updates.
"""

import io
from pathlib import Path

import config

import cv2
import numpy as np
from PIL import Image


# Scale factor: resolution-independent constants are written in the
# reference resolution's pixels (config.REF_WIDTH) and scaled at import time.
_S = config.SCALE_X

# Template images (one-time load at module init)
_TEMPLATE_DIR = Path(__file__).parent / "templates"
_heart_template: np.ndarray | None = None
_sendlike_template: np.ndarray | None = None

_SENDLIKE_CONFIDENCE = 0.85
_HEART_CONFIDENCE = 0.85


def _load_templates() -> None:
    """Lazy-load template images from disk. Called once per session."""
    global _heart_template, _sendlike_template
    heart_path = _TEMPLATE_DIR / "heart_template.jpg"
    sendlike_path = _TEMPLATE_DIR / "sendlike_template.jpg"
    if _heart_template is None and heart_path.exists():
        _heart_template = cv2.imread(str(heart_path), cv2.IMREAD_COLOR)
    if _sendlike_template is None and sendlike_path.exists():
        _sendlike_template = cv2.imread(str(sendlike_path), cv2.IMREAD_COLOR)


def _png_to_ndarray(png: bytes) -> np.ndarray:
    """Decode PNG bytes to BGR numpy array (OpenCV format)."""
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)


def _png_to_array(png: bytes) -> np.ndarray:
    """Decode PNG bytes to RGB numpy array (PIL format)."""
    return np.array(Image.open(io.BytesIO(png)).convert("RGB"))


# ============================================================
# find_send_like — OpenCV template matching
# ============================================================

def find_send_like(png: bytes, log_miss: bool = False) -> tuple[int, int] | None:
    """Locate the 'Send Like' button via OpenCV template matching.

    Searches the right half of the screen. Returns (x, y) center of the
    best match, or None if confidence is below threshold or the template
    file is missing.

    A miss is the *healthy* result at three of the four call sites — the
    stale-card check and both like-confirmation checks are asking "is a
    card still on screen?", and "no" is what a working like looks like. So
    the miss line is opt-in; do_like's own lookup, the one place a miss is
    a genuine failure, passes log_miss=True. Printing it unconditionally
    meant every profile logged a line that read like a fault.
    """
    _load_templates()
    if _sendlike_template is None:
        print("  Send Like: no template file — skipping")
        return None

    screen = _png_to_ndarray(png)
    scr_h, scr_w = screen.shape[:2]
    tmpl_h, tmpl_w = _sendlike_template.shape[:2]

    # Compose card sits in right half, middle 50% vertically
    y0 = int(scr_h * 0.25)
    y1 = int(scr_h * 0.75)
    x_crop = int(scr_w * 0.25)
    roi = screen[y0:y1, x_crop:]

    result = cv2.matchTemplate(roi, _sendlike_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= _SENDLIKE_CONFIDENCE:
        cx = x_crop + max_loc[0] + tmpl_w // 2
        cy = y0 + max_loc[1] + tmpl_h // 2
        print(f"  Send Like: template match at ({cx}, {cy}) conf={max_val:.3f}")
        return (cx, cy)

    if log_miss:
        print(f"  Send Like: not found (conf={max_val:.3f} < {_SENDLIKE_CONFIDENCE})")
    return None


# ============================================================
# find_first_heart — OpenCV template matching
# ============================================================

def find_first_heart(png: bytes) -> tuple[int, int] | None:
    """Locate the heart icon on photo 1 via OpenCV template matching.

    Searches the right ~40% of the screen across the full height.
    Finds all heart matches above confidence and returns the top-most
    one (smallest Y = highest on screen = photo 1).
    """
    _load_templates()
    if _heart_template is None:
        print("  Heart vision: no template file — skipping")
        return None

    screen = _png_to_ndarray(png)
    scr_h, scr_w = screen.shape[:2]
    tmpl_h, tmpl_w = _heart_template.shape[:2]

    x0 = int(scr_w * 0.60)
    x1 = scr_w
    y0 = int(scr_h * 0.10)
    y1 = int(scr_h * 0.90)
    roi = screen[y0:y1, x0:x1]

    result = cv2.matchTemplate(roi, _heart_template, cv2.TM_CCOEFF_NORMED)

    # Find all matches above threshold, pick the top-most (smallest Y).
    locations = np.where(result >= _HEART_CONFIDENCE)
    if len(locations[0]) == 0:
        best = result.max()
        print(f"  Heart vision: not found (best conf={best:.3f} < {_HEART_CONFIDENCE})")
        return None

    best_idx = locations[0].argmin()
    match_y = locations[0][best_idx]
    match_x = locations[1][best_idx]
    best_conf = result[match_y, match_x]

    cx = x0 + match_x + tmpl_w // 2
    cy = y0 + match_y + tmpl_h // 2
    print(f"  Heart vision: top match at ({cx}, {cy}) conf={best_conf:.3f} "
          f"({len(locations[0])} total above threshold)")
    return (cx, cy)


# ============================================================
# Comment field / input helpers
# ============================================================

def comment_field_text_pixels(png: bytes, send_like_xy: tuple[int, int]) -> int:
    """Count dark (text) pixels in the comment input area above Send Like."""
    arr = _png_to_array(png)
    sx, sy = send_like_xy
    y0 = max(0, sy - int(230 * _S))
    y1 = max(0, sy - int(60 * _S))
    x0 = max(0, sx - int(350 * _S))
    x1 = min(arr.shape[1], sx + int(350 * _S))
    region = arr[y0:y1, x0:x1]
    if region.size == 0:
        return 0
    dark = (region.max(axis=-1) < 130).sum()
    return int(dark)


def find_comment_input(send_like_xy: tuple[int, int]) -> tuple[int, int]:
    """Comment input sits at a fixed offset above the Send Like button."""
    _, send_y = send_like_xy
    return (int(540 * _S), send_y - int(171 * _S))


# ============================================================
# is_app_loading — detect stuck loading / splash screen
# ============================================================

def is_app_loading(png: bytes) -> bool:
    """Check if Hinge is stuck on the loading/splash screen.

    The loading screen is a white backdrop with a small animated logo
    in the center — the animation creates some pixel variation at
    center, but the overwhelming majority of the content area is still
    near-white. A loaded profile has a photo card, text overlays, and
    UI buttons that fill most of the screen with non-white pixels.

    Uses a simple white-pixel ratio across the full content area
    (excluding status bar and nav bar). If >75% of pixels are
    near-white (≥230), it's a loading screen. On a real profile,
    photos and text bring this below the threshold.

    Measured at 720x1600 over 200 sampled profiles: real profile = 0.44
    median, 0.62 max at the capture position. That margin is narrower
    than the Bumble sibling's (profile ~0.17, splash ~0.98) because
    Hinge floats each photo card on a white background rather than
    filling the screen with it. Mid-scroll frames reach 0.94, so this
    test is only safe because the guard runs before the first scroll —
    do not move the call site later in capture_profile().
    """
    im = np.array(Image.open(io.BytesIO(png)).convert("L"))
    h, w = im.shape

    # Full-screen sanity check — reject thumbnails and crops. Derived
    # from config rather than hard-coded, so it tracks the device if the
    # phone changes.
    #
    # This read "if h < 1500 or w < 950" until 2026-09-24: bounds written
    # for the 1080-wide reference resolution, and left unscaled when the
    # screen changed to 720 wide in 8d3a736. Every frame captured was 720
    # wide, so the gate returned False unconditionally and the white-ratio
    # test below was unreachable: in the eight weeks this guard existed it
    # never fired once, and the "loading screen" recovery branch in main.py
    # was equally dead. Same fix as the Bumble sibling, which was ported
    # with the config-derived form.
    if h < config.SCREEN_HEIGHT * 0.9 or w < config.SCREEN_WIDTH * 0.9:
        return False

    # Full content area: exclude status bar (~y=0-150) and nav bar
    # (~bottom 250px). The animated logo occupies a small fraction of
    # this region — it won't push white_ratio below the threshold.
    content = im[150:h - 250, 50:w - 50]
    if content.size == 0:
        return False

    white_ratio = (content > 230).mean()
    return white_ratio > 0.75
