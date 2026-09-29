"""One-off: scroll to the top, then progressively capture frames top-to-bottom."""

import shutil
import time
from pathlib import Path

import adb
import config

out = Path(__file__).parent / "debug" / "full_profile"
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)

adb.check_device()

# Reset: scroll up 8 times to guarantee we land at the top
print("Scrolling to top...")
for _ in range(8):
    adb.scroll_up()
    time.sleep(0.4)

time.sleep(1.0)

# Capture frame 0 at the top
(out / "frame_00.png").write_bytes(adb.screenshot())
print("frame_00 saved (top)")

# Scroll distance is in reference pixels, scaled below like the rest of the
# tree. It needs the scaling: unscaled it began at y=1700, 100px below the
# bottom of a 1600px screen, so every frame after the first was a duplicate
# of the top of the profile.
#
# Scaled, this is ~660px on a 720x1600 device — the same gesture main.py's
# capture_profile makes. The note here used to call it "smaller than the
# global scroll", which only held while both were unscaled.
NUM_FRAMES = 7
_x = int(540 * config.SCALE_X)
for i in range(1, NUM_FRAMES):
    adb.swipe(_x, int(1700 * config.SCALE_Y), _x, int(700 * config.SCALE_Y), 350)
    time.sleep(0.9)
    (out / f"frame_{i:02d}.png").write_bytes(adb.screenshot())
    print(f"frame_{i:02d} saved")

print(f"\nDone -> {out}")
