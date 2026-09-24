"""Saves a screenshot so you can read pixel coords for config.COORDS.

Open the resulting PNG in any image viewer that shows cursor position
(MS Paint, IrfanView, etc.) and hover over each UI element you need
to record.
"""

from pathlib import Path

import adb


def main() -> None:
    adb.check_device()
    path = Path(__file__).parent / "calibrate.png"
    path.write_bytes(adb.screenshot())
    print(f"Saved: {path}")
    print()
    print("Open the file and read pixel coords for:")
    print("  1. Skip (X) button         -> COORDS['skip_button']")
    print()
    print("The heart and the 'Send Like' button are not in COORDS — vision.py")
    print("locates both by template matching at tap-time, so there is nothing")
    print("to read off for them here.")
    print()
    print("Then edit config.py.")


if __name__ == "__main__":
    main()
