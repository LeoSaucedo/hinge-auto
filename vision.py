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

    Hinge changed the compose card button color in July 2026 — replaced
    the pink 'Send Like' text with a warm beige/cream text (RGB ~238, 225,
    219). This finds the beige text blob and returns its centroid as a
    proxy for the button position.
    """
    arr = _png_to_array(png)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    # Beige "Send Like" text: r ~238, g ~225, b ~219. r>g>b, all close.
    beige = (
        (r > 230) & (r < 248) &
        (g > 215) & (g < 235) &
        (b > 208) & (b < 230) &
        (r > g) & (g > b)
    )
    labeled, num = label(beige)
    candidates = []
    y_cutoff = int(arr.shape[0] * 0.10)  # ignore top 10% (three-dot menu area)
    for i in range(1, num + 1):
        sl = find_objects((labeled == i).astype(np.int32))
        if sl is None or sl[0] is None:
            continue
        y0, y1 = sl[0][0].start, sl[0][0].stop
        x0, x1 = sl[0][1].start, sl[0][1].stop
        h, w = y1 - y0, x1 - x0
        area = ((labeled == i).astype(np.int32))[sl[0]].sum()
        # The text is ~25-30px tall; filter out noise.
        if not (h > 15 and area > 30):
            continue
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        if cy < y_cutoff:
            continue
        # Button text is in the right half of the compose card.
        if cx < int(500 * _S):
            continue
        candidates.append((cy, cx))
    if not candidates:
        return None
    candidates.sort()
    cy, cx = candidates[0]
    return (cx, cy)


def _heart_button_score(arr: np.ndarray, cx: int, cy: int) -> float | None:
    """Score a candidate heart button at (cx, cy).

    Returns a 0-1 score or None if the candidate fails basic checks.
    The heart button is a dark circle (~25px diameter) with a white
    heart icon inside. We check:
      - Dark ring at the boundary
      - White interior (the heart icon cutout)
      - Moderate dark-fill ratio (~0.50-0.85 for icon cutout)
    """
    scr_h, scr_w = arr.shape[:2]
    radius = int(13 * _S)  # button radius in pixels
    margin = 2
    x0 = max(0, cx - radius - margin)
    x1 = min(scr_w, cx + radius + margin + 1)
    y0 = max(0, cy - radius - margin)
    y1 = min(scr_h, cy + radius + margin + 1)

    patch = arr[y0:y1, x0:x1]
    if patch.shape[0] < 10 or patch.shape[1] < 10:
        return None

    # Create a circular mask for the interior (slightly smaller than radius)
    py, px = patch.shape[:2]
    cy_p, cx_p = py // 2, px // 2
    yy, xx = np.ogrid[:py, :px]
    dist = np.sqrt((yy - cy_p) ** 2 + (xx - cx_p) ** 2)
    inner_mask = dist <= (radius - 3) * 0.9
    ring_mask = (dist >= radius * 0.7) & (dist <= radius * 1.1)

    dark = (patch[..., 0] < 60) & (patch[..., 1] < 60) & (patch[..., 2] < 60)
    white = (patch[..., 0] > 180) & (patch[..., 1] > 180) & (patch[..., 2] > 180)

    # Dark ring: should have high dark-pixel density
    ring_pixels = ring_mask.sum()
    if ring_pixels < 10:
        return None
    ring_dark = (dark & ring_mask).sum() / ring_pixels
    if ring_dark < 0.30:
        return None

    # White interior: the heart icon is white
    inner_pixels = inner_mask.sum()
    if inner_pixels < 5:
        return None
    inner_white = (white & inner_mask).sum() / inner_pixels
    if inner_white < 0.20:
        return None

    # Overall dark fill in the whole window (not just inner/ring)
    total_dark = dark.sum() / patch[..., 0].size
    if not (0.15 < total_dark < 0.75):
        return None

    # Score: combine ring darkness + interior whiteness
    return ring_dark * 0.6 + inner_white * 0.4


def find_first_heart(png: bytes) -> tuple[int, int] | None:
    """Locate the heart icon on photo 1 (topmost heart in current view).

    The heart button is a dark circle (~25px) with a white heart icon
    inside, positioned on the right side of the screen. Its Y position
    varies per profile (photo size, text length).

    Two-tier detection:
      1. Quick 5x5 dark-pixel check at the calibrated static coord.
         Returns instantly if the heart is exactly where expected.
      2. Scanning-window search across the right side of the screen.
         Scores each candidate by dark-ring + white-interior pattern.
         Immune to connected-component blob-merging issues.
    """
    arr = _png_to_array(png)
    scr_h, scr_w = arr.shape[:2]
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # ---- 1. Quick static-coord check ----
    sx, sy = config.COORDS["heart_photo_1"]
    if 2 <= sy < scr_h - 2 and 2 <= sx < scr_w - 2:
        r_patch = r[sy-2:sy+3, sx-2:sx+3]
        g_patch = g[sy-2:sy+3, sx-2:sx+3]
        b_patch = b[sy-2:sy+3, sx-2:sx+3]
        dark_patch = (r_patch < 60) & (g_patch < 60) & (b_patch < 60)
        if dark_patch.sum() >= 15:
            return (sx, sy)

    # ---- 2. Scanning-window search across right half of screen ----
    step = max(int(5 * _S), 3)
    x_start = int(scr_w * 0.78)
    x_end = int(scr_w * 0.96)
    y_start = int(scr_h * 0.25)
    y_end = int(scr_h * 0.88)
    best_score = 0.0
    best_xy = None
    for cy in range(y_start, y_end, step):
        for cx in range(x_start, x_end, step):
            score = _heart_button_score(arr, cx, cy)
            if score is not None and score > best_score:
                best_score = score
                best_xy = (cx, cy)

    if best_xy is not None and best_score > 0.40:
        print(f"  Heart vision: scan hit at {best_xy} score={best_score:.2f}")
        return best_xy

    return None


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
