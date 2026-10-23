"""Offline feasibility harness for like-targeting. NOT part of the loop.

Nothing imports this; the swipe loop, judge, config, and cron path are
untouched. It reads the saved debug corpus read-only and measures the
mechanical claims the like-targeting design rests on:

  1. Heart detection at scale: card hearts per frame, outliers, junk matches.
  2. Signature crops: which detected hearts get a usable content patch
     (the region we would template-match back onto the card at tap time).
  3. Relocalization: does a heart's content patch re-find its card in a
     DIFFERENT real screenshot of the same profile (conf + position error)?
     This is exactly what the tap-time scan would do, minus the swiping.
     Tested forward (matching the real top-down scan); "below" hearts,
     which have scrolled out forward, are tested backwards as well.
  4. Animation suspects: hearts that should have been findable in a
     neighboring frame (pair overlap proven by other hearts) but weren't.

Model-side behavior (picking a badge, writing the anchored opener) is NOT
testable here — that is a live-call question.

Usage:
    python eval_targeting.py [--profiles 100] [--seed 42] [--debug-dir PATH]
Artifacts (examples + per-profile CSV): /tmp/eval_targeting/
"""

import argparse
import csv
import random
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent
HEART_TEMPLATE = REPO / "templates" / "heart_template.jpg"
DEFAULT_CORPUS = Path("/mnt/sda1/hinge-auto/debug")

# Detection — same template + threshold as production find_first_heart.
HEART_CONF = 0.85
NMS_PX = 60
DETECT_X_MIN = 420      # nav-bar heart sits at x~360; card hearts at x~600-634
SUSPICIOUS_X = 540      # a "card heart" left of this is an outlier
Y_MIN, Y_MAX = 150, 1485

SCREEN_W = 720          # corpus capture width; the harness is 720x1600-only

# Signature crop geometry (relative to the heart center).
CROP_X0, CROP_X1 = -244, -34      # left of the heart glyph
CROP_ABOVE = (-330, -45)          # primary: content above the heart
CROP_BELOW = (45, 330)            # fallback: content below the heart
UI_TOP, UI_BOT = 175, 1480        # keep off the top bar / nav bar
MIN_CROP = 120                    # min height/width to bother matching
LOW_TEXTURE = 15.0                # grayscale std below this = flat/white patch

# Relocalization gates (what the executor would demand before tapping).
# LOCATABLE_CONF: a match this strong against static content is the found
# signal (non-overlap noise measured <= ~0.76). TAP_CONF: the executor's
# conf gate before trusting a heart at the derived spot.
LOCATABLE_CONF = 0.97
TAP_CONF = 0.90
LINK_PX = 8

TEMPLATE = cv2.imread(str(HEART_TEMPLATE))
assert TEMPLATE is not None, f"missing template: {HEART_TEMPLATE}"
TH, TW = TEMPLATE.shape[:2]


def detect_hearts(img: np.ndarray) -> list[tuple[float, int, int]]:
    """All card-heart matches >= HEART_CONF, NMS'd, sorted top-to-bottom."""
    res = cv2.matchTemplate(img, TEMPLATE, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= HEART_CONF)
    pts = sorted(zip(res[ys, xs], xs, ys), reverse=True)
    kept = []
    for s, x, y in pts:
        if all(abs(int(x) - kx) > NMS_PX or abs(int(y) - ky) > NMS_PX
               for _, kx, ky in kept):
            kept.append((float(s), int(x), int(y)))
    out = []
    for s, x, y in kept:
        cx, cy = x + TW // 2, y + TH // 2
        if cx < DETECT_X_MIN or not (Y_MIN < cy < Y_MAX):
            continue
        out.append((s, cx, cy))
    return sorted(out, key=lambda h: h[2])


def pick_signature(hx: int, hy: int):
    """Return (variant, x0, x1, y0, y1) for a content patch near the heart."""
    variants = [
        ("above", CROP_X0, CROP_X1, hy + CROP_ABOVE[0], hy + CROP_ABOVE[1]),
        ("below", CROP_X0, CROP_X1, hy + CROP_BELOW[0], hy + CROP_BELOW[1]),
    ]
    for name, rx0, rx1, ry0, ry1 in variants:
        x0, x1 = max(0, hx + rx0), min(SCREEN_W, hx + rx1)
        y0, y1 = max(UI_TOP, ry0), min(UI_BOT, ry1)
        if y1 - y0 >= MIN_CROP and x1 - x0 >= MIN_CROP:
            return name, x0, x1, y0, y1
    return None


def match_conf(frame_gray: np.ndarray, sig_gray: np.ndarray):
    res = cv2.matchTemplate(frame_gray, sig_gray, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    return float(mx), loc


def annotate(frame: np.ndarray, hx: int, hy: int, rect=None, label=""):
    img = frame.copy()
    if rect:
        x0, x1, y0, y1 = rect
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 3)
    cv2.circle(img, (hx, hy), 42, (255, 0, 255), 3)
    if label:
        cv2.putText(img, label, (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (255, 0, 255), 3)
    return img


def load_frames(folder: Path):
    frames = []
    for i in range(7):
        img = cv2.imread(str(folder / "imgs" / f"frame_{i:02d}.png"))
        if img is None:
            return None
        frames.append(img)
    return frames


def run_profile(folder: Path, bucket: str, out_dir: Path, examples: dict) -> dict:
    frames = load_frames(folder)
    if frames is None:
        return {"folder": folder.name, "bucket": bucket, "frames_ok": 0}

    hearts = {k: detect_hearts(f) for k, f in enumerate(frames)}
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    n_hearts = sum(len(v) for v in hearts.values())

    # Signature crop per heart + relocalization into the next 2 frames.
    # records: (conf, heart_found_at_spot, variant, std)
    # links: tap-ready relocalizations (strong conf AND heart at the spot)
    records = []
    links = []            # (k, i, j, i2, conf, step)
    unusable = 0
    crops = {}            # (k, i) -> (variant, x0, x1, y0, y1, std)
    for k in range(7):
        for i, (c, hx, hy) in enumerate(hearts[k]):
            sig_info = pick_signature(hx, hy)
            if sig_info is None:
                unusable += 1
                continue
            variant, x0, x1, y0, y1 = sig_info
            sig = gray[k][y0:y1, x0:x1]
            std = float(sig.std())
            crops[(k, i)] = (variant, x0, x1, y0, y1, std)
            dx, dy = hx - x0, hy - y0
            # Forward = the direction the production scan runs (start at
            # top, scroll down): a card enters low on screen and rises.
            # The "below" variant is only used when there is no room above
            # the heart (heart y < ~340) -- and a heart that high at
            # capture has scrolled out by k+1, so forward tests can never
            # validate it. Test it backwards too, where the card sits
            # lower; that is also the geometry the real scan sees when the
            # card first comes into view.
            js = [k + 1, k + 2]
            if variant == "below":
                js += [k - 1, k - 2]
            for j in js:
                if not (0 <= j <= 6):
                    continue
                conf, loc = match_conf(gray[j], sig)
                ex, ey = loc[0] + dx, loc[1] + dy
                found = None
                for i2, (c2, hx2, hy2) in enumerate(hearts[j]):
                    if abs(hx2 - ex) <= LINK_PX and abs(hy2 - ey) <= LINK_PX:
                        found = i2
                        break
                records.append((k, i, j, conf, found is not None,
                                variant, std, ey))
                if found is not None and conf >= LOCATABLE_CONF:
                    step = (hy - hearts[j][found][2]) / (j - k)
                    links.append((k, i, j, found, conf, step))

    # Cluster hearts into unique cards via the link graph (union-find).
    parent = {}
    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for k in range(7):
        for i in range(len(hearts[k])):
            find((k, i))
    for k, i, j, i2, conf, step in links:
        union((k, i), (j, i2))
    cards = len({find(n) for n in parent})

    # Animation suspects: pair overlap proven by >=1 link, but this heart
    # (usable, textured crop) had no link to j at its expected position.
    suspects = []
    if links:
        med_step = float(np.median([l[5] for l in links]))
        linked = {(k, i) for k, i, *_ in links}
        rec_by = {}
        for k, i, j, conf, found, _v, _std, _ey in records:
            rec_by.setdefault((k, i), []).append((conf, found))
        for k in range(7):
            for i, (c, hx, hy) in enumerate(hearts[k]):
                if (k, i) in linked or (k, i) not in crops:
                    continue
                variant, _, _, _, _, std = crops[(k, i)]
                if std < LOW_TEXTURE:
                    continue
                js = [k + 1, k + 2]
                if variant == "below":
                    js += [k - 1, k - 2]
                for j in js:
                    if not (0 <= j <= 6):
                        continue
                    pair_proven = any(lk[0] == k and lk[2] == j for lk in links)
                    expected = hy - med_step * (j - k)
                    # Only a fair test inside the detection band: outside
                    # it the heart is clipped and couldn't be found even
                    # if its content relocalized perfectly.
                    if pair_proven and Y_MIN <= expected <= Y_MAX:
                        # Classification: best match conf this heart got
                        # over its tested pairs, and whether a heart was
                        # detected at the derived spot.
                        bc, bf = max(rec_by.get((k, i), [(0.0, False)]))
                        suspects.append((k, i, j, folder.name, bc, bf))
                        break

    # Save one annotated relocalization example per profile, up to a cap.
    if links and examples["links"] < 4:
        k, i, j, i2, conf, _ = links[0]
        _, hx, hy = hearts[k][i]
        variant, x0, x1, y0, y1, std = crops[(k, i)]
        src = annotate(frames[k], hx, hy, (x0, x1, y0, y1),
                       f"crop {conf:.2f}")
        cv2.imwrite(str(out_dir / f"link_{folder.name}_f{k}->f{j}.png"), src)
        examples["links"] += 1
    if suspects and examples["suspects"] < 3:
        k, i, j, _, bc, bf = suspects[0]
        _, hx, hy = hearts[k][i]
        cv2.imwrite(str(out_dir / f"suspect_{folder.name}_f{k}.png"),
                    annotate(frames[k], hx, hy,
                             label=f"no link (best {bc:.2f}"
                                   f"{', heart found' if bf else ''})"))
        examples["suspects"] += 1

    per_frame = [len(hearts[k]) for k in range(7)]
    return {
        "folder": folder.name, "bucket": bucket, "frames_ok": 7,
        "hearts": n_hearts, "hearts_max_frame": max(per_frame),
        "unusable": unusable, "tests": len(records),
        "tap_ready": len(links), "cards": cards, "suspects": len(suspects),
        "records": records, "crops": crops, "steps": [l[5] for l in links],
        "suspect_recs": suspects,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profiles", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--debug-dir", type=Path, default=DEFAULT_CORPUS)
    args = ap.parse_args()

    corpus = args.debug_dir
    rng = random.Random(args.seed)

    def sample(bucket, n):
        root = corpus / bucket
        folders = [d for d in root.iterdir()
                   if (d / "imgs" / "frame_00.png").exists()]
        rng.shuffle(folders)
        return folders[:n]

    half = args.profiles // 2
    picked = ([(d, "liked") for d in sample("liked", half)]
              + [(d, "skipped") for d in sample("skipped", args.profiles - half)])
    rng.shuffle(picked)

    out_dir = Path("/tmp/eval_targeting")
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = {"links": 0, "suspects": 0}

    results, all_records, all_parts, suspects_all = [], [], [], []
    t0 = time.time()
    for n, (folder, bucket) in enumerate(picked, 1):
        r = run_profile(folder, bucket, out_dir, examples)
        results.append(r)
        all_records.extend(r.get("records", []))
        all_parts.extend(r.get("steps", []))
        suspects_all.extend(r.get("suspect_recs", []))
        if n % 10 == 0 or n == len(picked):
            print(f"  [{n}/{len(picked)}] {time.time()-t0:.0f}s")

    ok = [r for r in results if r.get("frames_ok") == 7]
    hearts = sum(r["hearts"] for r in ok)
    unusable = sum(r["unusable"] for r in ok)
    tests = sum(r["tests"] for r in ok)
    tap_ready_n = sum(r["tap_ready"] for r in ok)
    suspects = sum(r["suspects"] for r in ok)
    cards = [r["cards"] for r in ok]
    per_frame = [r["hearts"] / 7 for r in ok]

    # Split every test into: tap-ready (strong conf + heart at spot),
    # strong-but-heart-missing (content found, heart detection missed it --
    # e.g. edge-clipped; the executor would just resume scanning), weak
    # match that happened to land near a heart anyway, and no-match (this
    # is where "frames don't overlap" lands -- it must look clearly weaker
    # than the tap-ready set or the gate is unsafe).
    tap_confs, strong_missing, weak_coincident, no_match = [], [], [], []
    by_variant = {}
    below_fwd_on = below_fwd_off = 0
    for k_, i_, j_, c, found, v, _std, ey in all_records:
        d = by_variant.setdefault(v, [0, 0])
        d[0] += 1
        if found and c >= LOCATABLE_CONF:
            tap_confs.append(c)
            d[1] += 1
        elif c >= LOCATABLE_CONF:
            strong_missing.append(c)
        elif found:
            weak_coincident.append(c)
        else:
            no_match.append(c)
        if v == "below" and j_ > k_:
            if Y_MIN < ey < Y_MAX:
                below_fwd_on += 1
            else:
                below_fwd_off += 1

    print(f"\n== Like-targeting feasibility harness ==")
    print(f"corpus: {corpus}")
    print(f"profiles: {len(ok)} ok / {len(picked)} sampled   "
          f"({time.time()-t0:.0f}s)")
    if not ok:
        return 1
    print(f"\n-- Detection --")
    print(f"card hearts: {hearts}   per frame: mean "
          f"{np.mean(per_frame):.1f}  max-frame {max(r['hearts_max_frame'] for r in ok)}")
    print(f"unique cards/profile (link-clustered): median "
          f"{np.median(cards):.0f}  range {min(cards)}-{max(cards)}")
    print(f"\n-- Signature crops --")
    print(f"unusable (no crop fits): {unusable} ({unusable/max(hearts,1)*100:.1f}%)")
    for v, (n_, tr) in sorted(by_variant.items()):
        print(f"  {v:6s}: {n_:4d} tests, tap-ready {tr/n_*100:.1f}%")
    if below_fwd_on + below_fwd_off:
        print(f"  below-variant forward tests with the derived heart "
              f"off-screen: {below_fwd_off}/{below_fwd_on + below_fwd_off} "
              f"(heart had scrolled out by then -- geometry, not the crop)")
    stds = [std for r in ok for (_, _, _, _, _, std) in r["crops"].values()]
    print(f"crop texture std: median {np.median(stds):.1f}  "
          f"p10 {np.percentile(stds, 10):.1f}")
    print(f"\n-- Relocalization (real renders, adjacent/next-next frames) --")
    print(f"tests: {tests}  (note: most pairs share no content -- big "
          f"scroll steps -- so 'no match' is mostly geometry, not failure)")
    print(f"tap-ready (conf>={LOCATABLE_CONF} AND heart at derived spot): "
          f"{tap_ready_n} ({tap_ready_n/max(tests,1)*100:.1f}% of tests)")
    if tap_confs:
        print(f"  tap-ready conf: min {min(tap_confs):.3f}  "
              f"p5 {np.percentile(tap_confs, 5):.3f}  "
              f"median {np.median(tap_confs):.3f}")
    if strong_missing:
        print(f"strong match but heart not detected at spot: "
              f"{len(strong_missing)}  conf median {np.median(strong_missing):.3f}"
              f"  (edge-clipped card; scan resumes, not a failure)")
    if weak_coincident:
        print(f"weak match landing near a heart anyway: {len(weak_coincident)}"
              f"  conf median {np.median(weak_coincident):.3f}  "
              f"max {max(weak_coincident):.3f}"
              f"   [>= {TAP_CONF}: "
              f"{sum(1 for c in weak_coincident if c >= TAP_CONF)}]")
    if no_match:
        print(f"no match (no overlap / failure): {len(no_match)}  "
              f"conf median {np.median(no_match):.3f}  "
              f"p95 {np.percentile(no_match, 95):.3f}  max {max(no_match):.3f}"
              f"   [>= {TAP_CONF}: {sum(1 for c in no_match if c >= TAP_CONF)}]")
    if all_parts:
        big = [s for s in all_parts if abs(s) > 100]
        print(f"tap-ready pairs: {len(all_parts)}  step median "
              f"{np.median(all_parts):.0f}px  range "
              f"{min(all_parts):.0f}..{max(all_parts):.0f}px  "
              f"|step|>100px: {len(big)}")
    print(f"\n-- Animation suspects --")
    print(f"hearts expected-visible but unlinked: {suspects} "
          f"(classified by best match conf over their tested pairs)")
    absent = [s for s in suspects_all if s[4] < 0.70]
    near = [s for s in suspects_all if 0.70 <= s[4] < LOCATABLE_CONF and s[5]]
    weak = [s for s in suspects_all if 0.70 <= s[4] < LOCATABLE_CONF and not s[5]]
    odd = [s for s in suspects_all if s[4] >= LOCATABLE_CONF]
    print(f"  content absent (conf<0.70 -- animation/edited card): {len(absent)}")
    print(f"  0.70-0.97, heart found at spot (near-miss): {len(near)}")
    print(f"  0.70-0.97, no heart at spot: {len(weak)}")
    if odd:
        print(f"  >= {LOCATABLE_CONF}, heart at derived spot not "
              f"detected (edge-clipped): {len(odd)}")
    print(f"\nartifacts: {out_dir}/  (examples + per_profile.csv)")

    with open(out_dir / "per_profile.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["folder", "bucket", "hearts", "hearts_max_frame",
                    "unusable", "tests", "tap_ready", "cards", "suspects"])
        for r in results:
            w.writerow([r["folder"], r["bucket"], r.get("hearts", ""),
                        r.get("hearts_max_frame", ""), r.get("unusable", ""),
                        r.get("tests", ""), r.get("tap_ready", ""),
                        r.get("cards", ""), r.get("suspects", "")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
