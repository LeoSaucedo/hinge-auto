"""Image-based detection for UI elements whose position varies per profile.

Uses OpenCV template matching for reliable detection immune to profile
photo colors and Hinge UI updates.
"""

import io
from pathlib import Path
from typing import NamedTuple

import config

import cv2
import numpy as np
from PIL import Image


# Scale factor: resolution-independent constants are written in the
# reference resolution's pixels (config.REF_WIDTH) and scaled at import time.
_S = config.SCALE_X

# Template images (one-time load at module init)
_TEMPLATE_DIR = Path(__file__).parent / "templates"
_heart_templates: list[tuple[str, np.ndarray]] | None = None
_sendlike_templates: list[tuple[str, np.ndarray]] | None = None
_rose_sheet_templates: list[tuple[str, np.ndarray]] | None = None
_rose_zero_template: np.ndarray | None = None
_rose_zero_count_mask: np.ndarray | None = None
_rose_zero_count_box: tuple[int, int, int, int] | None = None
_rose_purchase_template: np.ndarray | None = None

_SENDLIKE_CONFIDENCE = 0.85
_HEART_CONFIDENCE = 0.85
_ROSE_SHEET_CONFIDENCE = 0.85
_ROSE_ZERO_COUNT_IOU = 0.90
_ROSE_ZERO_COUNT_SEARCH_PAD = 3
_ROSE_PURCHASE_CONFIDENCE = 0.85

# The button's appearance varies twice over: the label depends on the
# account's subscription (Hinge X relabels it "Send Priority Like"), and
# Hinge 10.8.0 (auto-updated 2026-10-14 14:50) restyled that button from
# dark-on-light to white-on-black. Each variant is tried and the higher
# score wins, because they are mutually exclusive at this threshold.
#
# 2026-10-06, the label split — the only frames carrying the new label
# are the six error screenshots from that day's 11:00 outage, and live
# profiles are a 260-frame sample:
#
#   template                  Hinge X card   plain card   live profile
#   send_priority_like.jpg        0.999      0.553-0.578     <=0.386
#   send_like.jpg                 0.707       0.997         <=0.448
#
# 2026-10-15, the restyle — the composer screenshots that day's failing
# runs saved, the four 10-06 frames, and live profile frames from both
# UIs:
#
#   template                      10.8.0 card   10.6-era card   profiles
#   send_priority_like_dark.jpg     1.000          <=0.53        <=0.56
#   send_priority_like.jpg          <=0.35          1.000        <=0.36
#   send_like.jpg                   <=0.43          0.708        <=0.49
#
# Keeping all three costs two extra ~90 ms matchTemplate passes and
# covers a lapsed subscription, which would otherwise fail every like —
# a failure that only shows up as a run which skips every profile. Those
# off-card ceilings matter too: this same lookup is the "is a card still
# on screen?" signal at the stale-card check and both like-confirmation
# checks, so every score on a card-less frame has to stay well clear of
# 0.85 or a like that never went out reads as sent.
#
# The filename stem is the label in the log line, so there is no second
# place to update when a label is added.
_SENDLIKE_TEMPLATE_FILES = (
    "send_priority_like_dark.jpg",
    "send_priority_like.jpg",
    "send_like.jpg",
)

# The photo-1 like glyph moves with the app version too: Hinge 10.8.0
# (auto-updated 2026-10-14 14:50) added a small nested "priority like"
# heart inside the existing outline. Both are tried and the higher score
# wins, same as the composer labels above — mutually exclusive at this
# threshold, measured over the saved corpus (the 63 heart-not-found
# error frames from the first evening on 10.8.0, and the 28 liked
# profiles from that morning's runs):

#   template                    10.8.0 frames     pre-10.8.0 frames
#   heart_template_priority.jpg  0.870 (63/63)      0.628 best
#   heart_template.jpg           0.621 best         0.874 (28/28)

# "hit" is the top-most in-feed match, the score the threshold actually
# gates — exactly 0.870 on all 63 frames, 0.874 on all 28. The other
# template's *best* score anywhere in the search ROI stays <= 0.63, so
# only one template can clear 0.85 on a given frame.
#
# A stale single template fails every like with "couldn't find photo-1
# heart" (2026-10-14 afternoon: 63 like decisions, 0 sent), so this
# costs one extra matchTemplate pass to keep covering the old glyph.
_HEART_TEMPLATE_FILES = (
    "heart_template_priority.jpg",
    "heart_template.jpg",
)

# The "Send a Rose instead?" bottom sheet that sometimes intercepts a Send
# Like. Both buttons are cropped from 20261022-081850_dialog-streak-1.png,
# and both score 1.000 on all 11 sheet frames in the debug corpus at
# identical coordinates. Off-sheet, the best score over 330 error frames
# and 800 profile frames is 0.608 (send_a_rose) / 0.619 (send_like_anyway),
# so 0.85 is a wide margin.
#
# The sheet's heading is deliberately NOT a template. Its wording changed
# between app versions — the 2026-10-09 frames score 0.627 against the
# 2026-10-22 heading crop — while the buttons did not move or restyle.
# Detection is the OR of the two buttons, never the AND: a missed sheet is
# the dangerous direction, because the scrim dims the compose card enough
# that find_send_like scores it <=0.59 and the like reads as sent.
_ROSE_SHEET_TEMPLATE_FILES = (
    ("send_a_rose", "rose_dialog_send.jpg"),
    ("send_like_anyway", "rose_dialog_like_anyway.jpg"),
)

# The lavender pill left of Send Like shows the rose count. The pill itself
# is found by color (see find_rose_pill).
#
# The zero template is a whole-pill crop of the "0 roses" composer, uploaded
# 2026-10-23. The pill's icon and fill are pixel-identical at every count —
# measured HSV mean (145.0, 32.9, 191.6) at count 0 vs (145.0, 32.8, 191.7)
# at count 1 — so only the count text differs.
#
# Do NOT match the whole pill. A live 3-rose pill on 2026-10-23 scored
# 0.965 against this template: the pill is 175x71 while the count is only
# ~13x17, so the flat lavender background dominates the correlation and a
# whole-pill threshold cannot tell 0 from any other count. (The 0.6744 the
# old corpus showed on count-1 pills came from icon misalignment in pre-10.8
# frames, not from the digit — an accident, not a signal.)
#
# Do NOT match only a tight crop around the "0" either: a two-digit count
# like "10" can right-align its "0" exactly where the single "0" sits, and
# the "1" then falls mostly outside a 23x27 crop. Instead, compare the dark
# ink of the whole count area to the right of the icon (see
# _locate_zero_count). A count of 0 reproduces that ink exactly (IoU 1.0);
# "1" scores 0.24 and "10" at most ~0.66 with normal kerning (~0.76 even if
# its digits overlap), because the extra digit's ink grows the union.
# Threshold 0.90.
_ROSE_ZERO_TEMPLATE_FILE = "zero_rose_pill.jpg"

# The rose purchase page's heading (title + subtitle), cropped from the
# screenshot uploaded 2026-10-23. This is what the rose pill opens when the
# balance is 0, and what tapping "Send a Rose" on the sheet opens at 0 — a
# page of buy buttons that must never be tapped through. The heading block
# is the distinctive part: 1.0000 self-match, best score on any of the 341
# error frames 0.2904 and on 200 profile frames 0.3096, so 0.85 is a wide
# margin. Calling it once costs ~0.23 s per frame on the Pi.
_ROSE_PURCHASE_TEMPLATE_FILE = "rose_purchase_title.jpg"


def _locate_zero_count(
        tmpl: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Cut the whole count area out of the 0-rose pill template.

    The pill carries exactly two dark components: the rose icon on the left
    and the count on the right. The rightmost is the count. The box spans
    from just right of the icon to just inside the pill's right edge, and
    vertically the count's glyph plus a margin, so any number of digits —
    "0", "1", "10", "99" — is inside it. Returns the box's dark-ink mask
    (bool) and the box (x0, y0, x1, y1), or None when the template has no
    usable dark components.

    Measured on the 175x71 template: icon (56,20)-(84,52), count "0"
    (103,27)-(116,44), so the box is (88,19)-(170,52).
    """
    dark = (cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY) < 100).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    comps = [i for i in range(1, n)
             if stats[i, cv2.CC_STAT_AREA] >= 20]
    if len(comps) < 2:
        return None
    icon = max(comps, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    count = max(comps, key=lambda i: stats[i, cv2.CC_STAT_LEFT])
    icon_right = int(stats[icon, cv2.CC_STAT_LEFT]
                     + stats[icon, cv2.CC_STAT_WIDTH])
    count_top = int(stats[count, cv2.CC_STAT_TOP])
    count_h = int(stats[count, cv2.CC_STAT_HEIGHT])
    x0 = max(0, icon_right + 4)
    x1 = max(x0, tmpl.shape[1] - 5)
    y0 = max(0, count_top - 8)
    y1 = min(tmpl.shape[0], count_top + count_h + 8)
    if x1 <= x0 or y1 <= y0:
        return None
    return dark[y0:y1, x0:x1].astype(bool), (x0, y0, x1, y1)


def _best_mask_iou(tmpl_mask: np.ndarray,
                   live_mask: np.ndarray,
                   pad: int) -> float:
    """Highest ink-mask IoU between two count areas over ±pad px shifts.

    `live_mask` must already be padded by `pad` on every side. IoU is
    intersection over union of the dark pixels, so extra ink (a second
    digit) lowers the score even when the "0" lines up perfectly. Returns
    0.0 when the masks never overlap.
    """
    h, w = tmpl_mask.shape
    best = 0.0
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            win = live_mask[pad + dy:pad + dy + h, pad + dx:pad + dx + w]
            if win.shape != tmpl_mask.shape:
                continue
            union = int(np.logical_or(tmpl_mask, win).sum())
            if union == 0:
                continue
            inter = int(np.logical_and(tmpl_mask, win).sum())
            best = max(best, inter / union)
    return best


def _zero_count_score(pill: np.ndarray) -> float | None:
    """IoU of a pill crop's count-area ink against the 0-rose template.

    The crop must be the template's size within a few pixels; anything
    else (another device, another pill) returns None, which callers treat
    as "not a confirmed 0". See `_locate_zero_count` for why the whole
    count area is compared instead of the digit alone.
    """
    if _rose_zero_template is None or _rose_zero_count_mask is None:
        return None
    t_h, t_w = _rose_zero_template.shape[:2]
    if abs(pill.shape[0] - t_h) > 4 or abs(pill.shape[1] - t_w) > 4:
        return None
    gx0, gy0, gx1, gy1 = _rose_zero_count_box
    region = cv2.cvtColor(pill[gy0:gy1, gx0:gx1],
                          cv2.COLOR_BGR2GRAY) < 100
    pad = _ROSE_ZERO_COUNT_SEARCH_PAD
    padded = np.pad(region.astype(np.uint8), pad).astype(bool)
    return _best_mask_iou(_rose_zero_count_mask, padded, pad)


def _load_templates() -> None:
    """Lazy-load template images from disk. Called once per session."""
    global _heart_templates, _sendlike_templates
    global _rose_sheet_templates, _rose_zero_template, _rose_purchase_template
    global _rose_zero_count_mask, _rose_zero_count_box
    if _heart_templates is None:
        _heart_templates = [
            (Path(name).stem,
             cv2.imread(str(_TEMPLATE_DIR / name), cv2.IMREAD_COLOR))
            for name in _HEART_TEMPLATE_FILES
            if (_TEMPLATE_DIR / name).exists()
        ]
    if _sendlike_templates is None:
        _sendlike_templates = [
            (Path(name).stem,
             cv2.imread(str(_TEMPLATE_DIR / name), cv2.IMREAD_COLOR))
            for name in _SENDLIKE_TEMPLATE_FILES
            if (_TEMPLATE_DIR / name).exists()
        ]
    if _rose_sheet_templates is None:
        _rose_sheet_templates = [
            (label,
             cv2.imread(str(_TEMPLATE_DIR / name), cv2.IMREAD_COLOR))
            for label, name in _ROSE_SHEET_TEMPLATE_FILES
            if (_TEMPLATE_DIR / name).exists()
        ]
    if _rose_zero_template is None:
        zero_path = _TEMPLATE_DIR / _ROSE_ZERO_TEMPLATE_FILE
        if zero_path.exists():
            _rose_zero_template = cv2.imread(str(zero_path), cv2.IMREAD_COLOR)
            if _rose_zero_template is not None:
                located = _locate_zero_count(_rose_zero_template)
                if located is not None:
                    _rose_zero_count_mask, _rose_zero_count_box = located
    if _rose_purchase_template is None:
        purchase_path = _TEMPLATE_DIR / _ROSE_PURCHASE_TEMPLATE_FILE
        if purchase_path.exists():
            _rose_purchase_template = cv2.imread(str(purchase_path),
                                                 cv2.IMREAD_COLOR)


# Decoding a 720x1600 screencap costs ~20 ms, and the send-settlement path
# runs several finders on the *same* frame (sheet, purchase page, card).
# Callers pass the exact bytes object from one screenshot() to all of them,
# so an identity check turns those extra decodes into a lookup. The finders
# only ever read from the returned array.
_last_png: bytes | None = None
_last_screen: np.ndarray | None = None


def _png_to_ndarray(png: bytes) -> np.ndarray:
    """Decode PNG bytes to BGR numpy array (OpenCV format)."""
    global _last_png, _last_screen
    if png is _last_png and _last_screen is not None:
        return _last_screen
    screen = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    _last_png = png
    _last_screen = screen
    return screen


def _png_to_array(png: bytes) -> np.ndarray:
    """Decode PNG bytes to RGB numpy array (PIL format)."""
    return np.array(Image.open(io.BytesIO(png)).convert("RGB"))


# ============================================================
# find_send_like — OpenCV template matching
# ============================================================

def find_send_like(png: bytes, log_miss: bool = False) -> tuple[int, int] | None:
    """Locate the like button via OpenCV template matching.

    Searches the right half of the screen, trying each label's template
    (see _SENDLIKE_TEMPLATE_FILES) and keeping the higher score. Returns
    (x, y) center of the best match, or None if confidence is below
    threshold or no template file is present.

    A miss is the *healthy* result at three of the four call sites — the
    stale-card check and both like-confirmation checks are asking "is a
    card still on screen?", and "no" is what a working like looks like. So
    the miss line is opt-in; do_like's own lookup, the one place a miss is
    a genuine failure, passes log_miss=True. Printing it unconditionally
    meant every profile logged a line that read like a fault.
    """
    _load_templates()
    if not _sendlike_templates:
        print("  Send Like: no template file — skipping")
        return None

    screen = _png_to_ndarray(png)
    scr_h, scr_w = screen.shape[:2]

    # Compose card sits in right half, middle 50% vertically
    y0 = int(scr_h * 0.25)
    y1 = int(scr_h * 0.75)
    x_crop = int(scr_w * 0.25)
    roi = screen[y0:y1, x_crop:]

    best: tuple[float, str, tuple[int, int]] | None = None
    for label, tmpl in _sendlike_templates:
        tmpl_h, tmpl_w = tmpl.shape[:2]
        if roi.shape[0] < tmpl_h or roi.shape[1] < tmpl_w:
            continue
        result = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if best is None or max_val > best[0]:
            best = (max_val, label,
                    (x_crop + max_loc[0] + tmpl_w // 2,
                     y0 + max_loc[1] + tmpl_h // 2))

    if best is not None and best[0] >= _SENDLIKE_CONFIDENCE:
        conf, label, (cx, cy) = best
        print(f"  Send Like: template match at ({cx}, {cy}) conf={conf:.3f} "
              f"[{label}]")
        return (cx, cy)

    if log_miss:
        conf, label = best[:2] if best else (0.0, "none")
        print(f"  Send Like: not found (conf={conf:.3f} < {_SENDLIKE_CONFIDENCE} "
              f"[best: {label}])")
    return None


# ============================================================
# Rose sheet / rose pill
# ============================================================

class RoseSheet(NamedTuple):
    """"Send a Rose instead?" sheet, as the buttons that matched.

    Each field is the (x, y) center of that button, or None when the
    template did not match. A returned RoseSheet always has at least one
    of the two — find_rose_sheet returns None when neither matches.
    """
    send_a_rose: tuple[int, int] | None
    send_like_anyway: tuple[int, int] | None


class RosePill(NamedTuple):
    """The composer's lavender rose-count pill.

    `center` is the tap target; `empty` is True only when the count area
    matched the 0-rose template's. Hinge draws a "0" only when the balance
    is empty, so a pill whose count does not match it has at least one rose
    behind it — including counts 2+ and pills whose crop is not the
    template's size. If the template (or its count area) failed to load,
    find_rose_pill returns None instead: no rose is spent without a
    working zero check.
    """
    center: tuple[int, int]
    empty: bool


# Lavender pill fill measured at BGR (192,167,188) -> HSV (145,33,192).
# The range is broad on purpose: the pill is the only saturated control
# in the composer toolbar, so a false positive would have to come from a
# photo pixel that scrolled into this row. The size filter and the
# 0-rose count match are the real gates.
_ROSE_PILL_HSV_LO = np.array([130, 15, 105])
_ROSE_PILL_HSV_HI = np.array([165, 130, 255])

# Measured pill size as a fraction of the screen: 175x71 on 720x1600.
_ROSE_PILL_W_RANGE = (0.15, 0.40)
_ROSE_PILL_H_RANGE = (0.02, 0.10)


def find_rose_sheet(png: bytes) -> RoseSheet | None:
    """Locate the "Send a Rose instead?" sheet, or None if it is not up.

    Detection is the OR of the two button templates (see
    _ROSE_SHEET_TEMPLATE_FILES): the sheet interrupts a Send Like, and a
    missed sheet is what makes the like look sent — the scrim dims the
    compose card below find_send_like's threshold.

    Both buttons sit in the sheet's lower half; the ROI skips the scrim
    and the card above. The sheet is pinned to the bottom edge, so this
    window does not need to track the profile layout.
    """
    _load_templates()
    if not _rose_sheet_templates:
        return None

    screen = _png_to_ndarray(png)
    scr_h, scr_w = screen.shape[:2]
    y0 = int(scr_h * 0.65)
    y1 = int(scr_h * 0.98)
    x0 = int(scr_w * 0.10)
    x1 = int(scr_w * 0.90)
    roi = screen[y0:y1, x0:x1]

    found: dict[str, tuple[int, int]] = {}
    for label, tmpl in _rose_sheet_templates:
        tmpl_h, tmpl_w = tmpl.shape[:2]
        if roi.shape[0] < tmpl_h or roi.shape[1] < tmpl_w:
            continue
        result = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val >= _ROSE_SHEET_CONFIDENCE:
            found[label] = (x0 + max_loc[0] + tmpl_w // 2,
                            y0 + max_loc[1] + tmpl_h // 2)
            print(f"  Rose sheet: '{label}' at {found[label]} "
                  f"conf={max_val:.3f}")

    if not found:
        return None
    return RoseSheet(send_a_rose=found.get("send_a_rose"),
                     send_like_anyway=found.get("send_like_anyway"))


def find_rose_pill(png: bytes,
                   send_like_xy: tuple[int, int]) -> RosePill | None:
    """Locate the lavender rose-count pill left of Send Like.

    Returns the pill's tap center and whether it shows the 0-rose count,
    or None when no pill is on screen.

    Only the count area is compared, not the whole pill. The pill is
    175x71 with ~13x17 of it being the count, so a whole-pill correlation
    is dominated by the flat lavender fill: a live 3-rose pill scored 0.965
    against the 0-rose image on 2026-10-23, while a true 0 is pixel-perfect.
    `_locate_zero_count` cuts the whole count area — everything right of
    the icon, wide enough for two digits — out of the template, and its
    dark-ink mask is compared by IoU within a ±3 px alignment window. A
    count of 0 reads 1.0; a count-1 corpus frame 0.24; a two-digit count
    stays at ~0.66 even when its "0" lands exactly on the template's
    (~0.76 if the digits overlap). Anything under `_ROSE_ZERO_COUNT_IOU`
    means at least one rose, which includes counts 2+.

    Must be called on the *undimmed* composer, before any send tap: the
    rose sheet dims the pill to V~59, well below the mask's floor, so a
    post-send read would report "no pill" on exactly the frames where a
    rose is available.

    Geometry on a 720x1600 frame: Send Like centered near x=456, the pill
    at x=63-238 on the same row.
    """
    _load_templates()
    if _rose_zero_template is None or _rose_zero_count_mask is None:
        print("  Rose pill: no readable 0-rose template — cannot read "
              "availability")
        return None

    screen = _png_to_ndarray(png)
    scr_h, scr_w = screen.shape[:2]
    sx, sy = send_like_xy

    # Reference-pixel scales for the measured offsets (720x1600). Config's
    # SCALE_* describe the 1080x2424 authoring ruler, not this measurement.
    ref_x = scr_w / 720.0
    ref_y = scr_h / 1600.0

    y0 = max(0, sy - int(80 * ref_y))
    y1 = min(scr_h, sy + int(80 * ref_y))
    x1 = max(0, sx - int(40 * ref_x))
    if y1 <= y0 or x1 <= 20:
        return None

    band = screen[y0:y1, :x1]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _ROSE_PILL_HSV_LO, _ROSE_PILL_HSV_HI)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    px, py, pw, ph = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if not (_ROSE_PILL_W_RANGE[0] * scr_w <= pw <= _ROSE_PILL_W_RANGE[1] * scr_w
            and _ROSE_PILL_H_RANGE[0] * scr_h <= ph <= _ROSE_PILL_H_RANGE[1] * scr_h):
        print(f"  Rose pill: lavender blob at ({px},{py}) size {pw}x{ph} "
              f"is not pill-shaped — ignoring")
        return None

    center = (px + pw // 2, y0 + py + ph // 2)

    zero_score = _zero_count_score(band[py:py + ph, px:px + pw])
    if zero_score is None:
        print(f"  Rose pill: crop {pw}x{ph} does not match the 0-rose "
              f"template — assuming at least one rose")

    empty = (zero_score is not None
             and zero_score >= _ROSE_ZERO_COUNT_IOU)
    score_text = f"{zero_score:.3f}" if zero_score is not None else "n/a"
    print(f"  Rose pill: at {center} "
          f"{'0 roses' if empty else 'not the 0-rose count'} "
          f"(count-match={score_text})")
    return RosePill(center=center, empty=empty)


def find_rose_purchase(png: bytes) -> bool:
    """True when the rose purchase page is on screen.

    A full-frame match of the page's heading block (see
    _ROSE_PURCHASE_TEMPLATE_FILE). This page is what a pill tap opens when
    the balance is 0 even though the pill was supposed to have a rose, and
    what tapping "Send a Rose" on the sheet opens in the same situation.
    It is a wall of buy buttons, so the caller backs out of it and sends a
    normal like — it must never be tapped on.

    Called from the send-settlement path and both recovery paths, so it
    pays one extra full-frame match (~0.23 s) per check.
    """
    _load_templates()
    if _rose_purchase_template is None:
        return False

    screen = _png_to_ndarray(png)
    tmpl_h, tmpl_w = _rose_purchase_template.shape[:2]
    if screen.shape[0] < tmpl_h or screen.shape[1] < tmpl_w:
        return False

    result = cv2.matchTemplate(screen, _rose_purchase_template,
                               cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    if max_val >= _ROSE_PURCHASE_CONFIDENCE:
        print(f"  Rose purchase screen: heading matched conf={max_val:.3f}")
        return True
    return False


# ============================================================
# find_first_heart — OpenCV template matching
# ============================================================

def find_first_heart(png: bytes) -> tuple[int, int] | None:
    """Locate the heart icon on photo 1 via OpenCV template matching.

    Searches the right ~40% of the screen across the full height, trying
    each glyph's template (see _HEART_TEMPLATE_FILES). Finds all matches
    above confidence and returns the top-most one (smallest Y = highest
    on screen = photo 1).
    """
    _load_templates()
    if not _heart_templates:
        print("  Heart vision: no template file — skipping")
        return None

    screen = _png_to_ndarray(png)
    scr_h, scr_w = screen.shape[:2]

    x0 = int(scr_w * 0.60)
    x1 = scr_w
    y0 = int(scr_h * 0.10)
    y1 = int(scr_h * 0.90)
    roi = screen[y0:y1, x0:x1]

    # Per template, the top-most match above threshold; across templates
    # the top-most of those. Candidates compare by glyph center Y, not by
    # match top edge — the two crops carry different margins, so top
    # edges would rank the templates a few pixels apart on equal glyphs.
    best_miss: tuple[float, str] = (0.0, "none")
    candidates: list[tuple[int, int, float, str, int]] = []
    for label, tmpl in _heart_templates:
        tmpl_h, tmpl_w = tmpl.shape[:2]
        if roi.shape[0] < tmpl_h or roi.shape[1] < tmpl_w:
            continue
        result = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
        if float(result.max()) > best_miss[0]:
            best_miss = (float(result.max()), label)
        locations = np.where(result >= _HEART_CONFIDENCE)
        if len(locations[0]) == 0:
            continue
        best_idx = locations[0].argmin()
        match_y = int(locations[0][best_idx])
        match_x = int(locations[1][best_idx])
        candidates.append((y0 + match_y + tmpl_h // 2,
                           x0 + match_x + tmpl_w // 2,
                           float(result[match_y, match_x]), label,
                           len(locations[0])))

    if not candidates:
        conf, label = best_miss
        print(f"  Heart vision: not found (best conf={conf:.3f} < "
              f"{_HEART_CONFIDENCE} [best: {label}])")
        return None

    cy, cx, conf, label, above = min(candidates, key=lambda c: c[0])
    print(f"  Heart vision: top match at ({cx}, {cy}) conf={conf:.3f} "
          f"[{label}] ({above} total above threshold)")
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

    Uses two measurements over the full content area (excluding status
    bar and nav bar), both of which must hold:

        white_ratio > 0.75 — the area is overwhelmingly near-white (≥230)
        dark_ratio  < 0.02 — and carries none of the profile chrome

    Measured at 720x1600 at the capture position:

        real profiles    white 0.44 median, 0.59 max; dark 0.37 median
        profile "K"      white 0.88-0.93;              dark 0.029-0.076
        genuine loading  white 0.987;                  dark 0.012

    The white test alone cannot separate the middle two. A profile
    whose photo card is a white-background line-art illustration is as
    white as a splash screen — a 1.06x gap — and on 2026-09-27 that
    tripped the guard in three separate runs, each of which restarted
    the app onto the same unchanged profile and gave up after three
    strikes. Nothing about that is transient, so the restart ladder
    could never clear it. The dark test separates the two by 2.45x:
    every real profile carries name text, prompt text, or the black
    action buttons, and a blank splash screen carries none. Across
    2,490 captured frames no profile fell below dark 0.036, so 0.02
    sits in an empty band with margin on both sides.

    That white margin is still narrower than the Bumble sibling's
    (profile ~0.17, splash ~0.98) because Hinge floats each photo card
    on a white background rather than filling the screen with it.
    Mid-scroll frames reach 0.94, so this test is only safe because
    the guard runs before the first scroll — do not move the call site
    later in capture_profile().
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

    # Both must hold. Whiteness alone flags a white-background photo
    # card as a splash screen; the dark-pixel floor is what separates
    # a blank screen from a rendered profile. 130 matches the text
    # threshold used by comment_field_text_pixels.
    white_ratio = (content > 230).mean()
    dark_ratio = (content < 130).mean()
    return white_ratio > 0.75 and dark_ratio < 0.02
