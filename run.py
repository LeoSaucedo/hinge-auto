"""Cron entry point — the outer crash net.

`python main.py` still works and reports its own runtime crashes, but it
cannot report an import-time failure: main's module body runs before its
`__main__` handler exists, so a missing module or a bad dependency dies
with a traceback in cron.log and nothing in Discord. That is the failure
mode a module-removal refactor produces, and it is exactly the one a
channel set to "Only @mentions" would never surface.

Importing main *inside* the try moves that case into reach. When run this
way main.py's own handler never executes (its module is imported, not
executed as a script), so there is no chance of reporting a crash twice.
"""

import sys
import traceback


def _report(tb: str) -> None:
    """Best-effort crash report. Never raises: a failure to report a
    failure must not mask the original traceback."""
    try:
        import report
        report.post_crash(tb)
    except Exception as e:
        print(f"[run] could not report crash: {e!r}", file=sys.stderr)


def main() -> int:
    try:
        import main as app
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        _report(tb)
        return 1

    try:
        return app.main()
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        _report(tb)
        return 1


if __name__ == "__main__":
    sys.exit(main())
