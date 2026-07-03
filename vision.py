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


def _find_pink_send_like(arr: np.ndarray) -> tuple[int, int] | None:
    """Look for hot-pink 'Send Like' text (active/enabled button).

    Pink: r~255, g~160, b~190 with r - g > 40.
    """
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    pink = (
        (r > 240) & (g > 140) & (g < 200) & (b > 170) & (b < 210)
        & (r > g) & (r > b)
        & ((r.astype(np.int32) - g.astype(np.int32)) > 40)
    )
    labeled, num = label(pink)
    candidates = []
    for i in range(1, num + 1):
        sl = find_objects((labeled == i).astype(np.int32))
        if sl is None or sl[0] is None:
            continue
        y0, y1 = sl[0][0].start, sl[0][0].stop
        x0, x1 = sl[0][1].start, sl[0][1].stop
        h, w = y1 - y0, x1 - x0
        area = ((labeled == i).astype(np.int32))[sl[0]].sum()
        if not (h > 15 and area > 30):
            continue
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        if cx < int(500 * _S):
            continue
        candidates.append((cy, cx))
    if not candidates:
        return None
    candidates.sort()
    cy, cx = candidates[0]
    return (cx, cy)


def _find_grey_send_like(arr: np.ndarray) -> tuple[int, int] | None:
    """Fallback: look for disabled Send Like button (grey pill, no pink text).

    When the compose card input field is empty, the Send Like button is a
    light-grey pill (RGB ~232,232,232) with medium-grey text, sitting in
    the compose card's bottom bar. This searches for the pill by looking
    for a region of near-uniform grey that differs from the bar background.
    """
    h, w = arr.shape[0], arr.shape[1]
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # The compose card bottom bar is roughly the lower-right area.
    # Search the bottom half of screen for a grey pill in the right 40%.
    roi_top = int(h * 0.55)
    roi_bottom = int(h * 0.94)
    roi_left = int(w * 0.45)

    r_roi = r[roi_top:roi_bottom, roi_left:]
    g_roi = g[roi_top:roi_bottom, roi_left:]
    b_roi = b[roi_top:roi_bottom, roi_left:]

    # Medium grey: all channels ~150-210, roughly equal (disabled text)
    grey_text = (
        (r_roi > 140) & (r_roi < 220)
        & (g_roi > 140) & (g_roi < 220)
        & (b_roi > 140) & (b_roi < 220)
        & (np.abs(r_roi.astype(np.int32) - g_roi.astype(np.int32)) < 15)
        & (np.abs(g_roi.astype(np.int32) - b_roi.astype(np.int32)) < 15)
    )
    labeled, num = label(grey_text)
    candidates = []
    for i in range(1, num + 1):
        sl = find_objects((labeled == i).astype(np.int32))
        if sl is None or sl[0] is None:
            continue
        y0, y1 = sl[0][0].start, sl[0][0].stop
        x0, x1 = sl[0][1].start, sl[0][1].stop
        bh, bw = y1 - y0, x1 - x0
        # The disabled Send Like text should be roughly 15-25px tall and
        # 80-200px wide (wider than tall — a text line, not noise).
        min_h, max_h = int(12 * _S), int(45 * _S)
        min_w, max_w = int(60 * _S), int(250 * _S)
        if not (min_h < bh < max_h and min_w < bw < max_w):
            continue
        cx = roi_left + (x0 + x1) // 2
        cy = roi_top + (y0 + y1) // 2
        # Must be on the right side of screen
        if cx < int(w * 0.5):
            continue
        candidates.append((cy, cx))
    if not candidates:
        return None
    candidates.sort()
    cy, cx = candidates[0]
    return (cx, cy)


def find_send_like(png: bytes) -> tuple[int, int] | None:
    """Locate the 'Send Like' button. Returns (x, y) center or None.

    Tries two approaches in order:
    1. Pink text detection (active/enabled button after typing)
    2. Grey pill detection (disabled button when input field is empty)

    The button may be disabled (grey) or active (pink text) depending on
    whether the comment field has text.
    """
    arr = _png_to_array(png)
    result = _find_pink_send_like(arr)
    if result is not None:
        return result
    return _find_grey_send_like(arr)


def find_first_heart(png: bytes) -> tuple[int, int] | None:
    """Locate the heart icon on photo 1 (topmost heart in current view)."""
    arr = _png_to_array(png)
    mask = (arr[..., 0] > 235) & (arr[..., 1] > 235) & (arr[..., 2] > 235)
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


def estimate_send_y(png: bytes) -> int:
    """Estimate the Send Like button Y position from the compose card layout.

    The compose card bottom bar is a uniform grey strip just above the
    nav area. When the buttons are not rendered (post-keyboard-dismiss), we
    can still find the bar to estimate where the button WOULD be.
    Falls back to a proportional estimate if no bar is detected.
    """
    arr = _png_to_array(png)
    h = arr.shape[0]
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # Scan from bottom up for the compose card's uniform grey bottom bar.
    # The bar is RGB(~246,~246,~246) on the Moto e20 — near-white but
    # darker than the card's white background. Scan the middle 50% of width
    # to avoid side shadows.
    bar_top = None
    bar_bot = None
    in_bar = False
    left = arr.shape[1] // 4
    right = 3 * arr.shape[1] // 4

    for y in range(h - 1, h // 2, -1):
        band = arr[y, left:right, :]
        avg = band.mean(axis=0)
        is_bar = (238 < avg[0] < 252) and (238 < avg[1] < 252) and (238 < avg[2] < 252)
        if is_bar and not in_bar:
            bar_bot = y
            in_bar = True
        elif not is_bar and in_bar:
            bar_top = y + 1
            break

    if bar_top and bar_bot and (bar_bot - bar_top) >= 15:
        return (bar_top + bar_bot) // 2

    # Fallback: proportional based on known layout
    return int(h * 0.86)


def comment_text_pixels(png: bytes, estimated_send_y: int) -> int:
    """Count dark text pixels in the comment input area using an estimated
    Send Like Y. Unlike comment_field_text_pixels(), this doesn't require
    knowing the exact send_xy — it uses a broader ROI based on the estimate."""
    arr = _png_to_array(png)
    y0 = max(0, estimated_send_y - int(280 * _S))
    y1 = max(0, estimated_send_y - int(50 * _S))
    x0 = max(0, int(540 * _S) - int(350 * _S))
    x1 = min(arr.shape[1], int(540 * _S) + int(350 * _S))
    region = arr[y0:y1, x0:x1]
    if region.size == 0:
        return 0
    dark = (region.max(axis=-1) < 130).sum()
    return int(dark)
