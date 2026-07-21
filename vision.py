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


# Scale factor: resolution-independent constants are defined for 1080px
# (Pixel 10 reference width) and scaled at import time.
_S = config.SCREEN_WIDTH / 1080.0

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

def find_send_like(png: bytes) -> tuple[int, int] | None:
    """Locate the 'Send Like' button via OpenCV template matching.

    Searches the right half of the screen. Returns (x, y) center of the
    best match, or None if confidence is below threshold or the template
    file is missing.
    """
    _load_templates()
    if _sendlike_template is None:
        print("  Send Like: no template file — skipping")
        return None

    screen = _png_to_ndarray(png)
    scr_w = screen.shape[1]
    tmpl_h, tmpl_w = _sendlike_template.shape[:2]

    # Narrow search to the compose-card region (right half)
    x_crop = int(scr_w * 0.25)
    roi = screen[:, x_crop:]

    result = cv2.matchTemplate(roi, _sendlike_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= _SENDLIKE_CONFIDENCE:
        cx = x_crop + max_loc[0] + tmpl_w // 2
        cy = max_loc[1] + tmpl_h // 2
        print(f"  Send Like: template match at ({cx}, {cy}) conf={max_val:.3f}")
        return (cx, cy)

    print(f"  Send Like: not found (conf={max_val:.3f} < {_SENDLIKE_CONFIDENCE})")
    return None


# ============================================================
# find_first_heart — OpenCV template matching
# ============================================================

def find_first_heart(png: bytes) -> tuple[int, int] | None:
    """Locate the heart icon on photo 1 via OpenCV template matching.

    Searches the right ~40% of the screen (photo 1 area). Returns
    (x, y) center of the best match, or None if confidence is below
    threshold or the template file is missing.
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
    y0 = int(scr_h * 0.20)
    y1 = int(scr_h * 0.50)  # top half only — avoid matching photo 2
    roi = screen[y0:y1, x0:x1]

    result = cv2.matchTemplate(roi, _heart_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= _HEART_CONFIDENCE:
        cx = x0 + max_loc[0] + tmpl_w // 2
        cy = y0 + max_loc[1] + tmpl_h // 2
        print(f"  Heart vision: template match at ({cx}, {cy}) conf={max_val:.3f}")
        return (cx, cy)

    print(f"  Heart vision: not found (conf={max_val:.3f} < {_HEART_CONFIDENCE})")
    return None


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
