"""Image-based detection for UI elements whose position varies per profile.

All pixel-size constants are defined for a 1080px-wide screen and scaled
at import time to match config.SCREEN_WIDTH so detection works on any
device (emulator, real phone, different resolutions).
"""

import io

import config

import numpy as np
from PIL import Image
from scipy.ndimage import label, find_objects


# Scale factor: resolution-independent constants are defined for 1080px
# (Pixel 10 reference width) and scaled at import time.
_S = config.SCREEN_WIDTH / 1080.0


def _png_to_array(png: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(png)).convert("RGB"))


def find_send_like(png: bytes) -> tuple[int, int] | None:
    """Locate the 'Send Like' button. Returns (x, y) center or None.

    Hinge changed the compose card button in late June 2026 — replaced
    the filled peach button with a white button + pink 'Send Like' text
    (RGB ~255, 159, 191). As of early July 2026 the button is back to
    a filled warm peach color (RGB ~238, 225, 219). This finds the
    peach blob and returns its centroid as a proxy for the button
    position.
    """
    arr = _png_to_array(png)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    # Peach filled button: r ~238, g ~225, b ~219. r > g > b, light/muted.
    peach = (
        (r > 225) & (r < 250)
        & (g > 210) & (g < 240)
        & (b > 205) & (b < 230)
        & (r > g) & (g > b)
    )
    labeled, num = label(peach)
    candidates = []
    for i in range(1, num + 1):
        sl = find_objects((labeled == i).astype(np.int32))
        if sl is None or sl[0] is None:
            continue
        y0, y1 = sl[0][0].start, sl[0][0].stop
        x0, x1 = sl[0][1].start, sl[0][1].stop
        h, w = y1 - y0, x1 - x0
        area = ((labeled == i).astype(np.int32))[sl[0]].sum()
        # Button is larger than text; use button-sized filter.
        if not (h > 30 and area > 500):
            continue
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        # Button is in the right half of the compose card.
        if cx < int(500 * _S):
            continue
        candidates.append((cy, cx))
    if not candidates:
        return None
    candidates.sort()
    cy, cx = candidates[0]
    return (cx, cy)


def find_first_heart(png: bytes) -> tuple[int, int] | None:
    """Locate the heart icon on photo 1 (topmost heart in current view).

    Verifies that the dark circle of the new white-on-black heart design
    (July 2026) is actually at the expected position before returning it.
    Falls back to None if the position doesn't look like it has the dark
    circle background — caller uses the static coordinate instead.
    """
    arr = _png_to_array(png)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # Check the expected heart position: dark circle (#1A1A1A = ~26,26,26)
    # at bottom-right of the action row. Sample a 5x5 kernel around the
    # static coordinate — if the majority of pixels are the dark button
    # color, the heart is present and the coord is likely valid.
    cx, cy = config.COORDS["heart_photo_1"]
    r_patch = r[cy-2:cy+3, cx-2:cx+3]
    g_patch = g[cy-2:cy+3, cx-2:cx+3]
    b_patch = b[cy-2:cy+3, cx-2:cx+3]
    dark = (r_patch < 60) & (g_patch < 60) & (b_patch < 60)
    if dark.sum() >= 15:  # at least 15/25 pixels are dark circle
        return (cx, cy)

    # The new dark button isn't where expected. Try the old white-circle
    # design as a fallback (black heart on white circular background).
    mask = (r > 235) & (g > 235) & (b > 235)
    labeled, _ = label(mask)
    hearts = []
    for i, sl in enumerate(find_objects(labeled), 1):
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        h, w = y1 - y0, x1 - x0
        area = (labeled[sl] == i).sum()
        if not (int(100 * _S) < h < int(140 * _S)
                and int(100 * _S) < w < int(140 * _S)):
            continue
        if abs(w - h) >= 15:
            continue
        if area < int(8500 * _S * _S) or area > int(12000 * _S * _S):
            continue
        cx = (x0 + x1) // 2
        if cx <= int(800 * _S):
            continue
        cy = (y0 + y1) // 2
        hearts.append((cy, cx))
    if not hearts:
        return None
    hearts.sort()
    cy, cx = hearts[0]
    return (cx, cy)


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
