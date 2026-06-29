"""Post run results to Discord via webhook.

Requires DISCORD_WEBHOOK_URL in the environment. No-op when unset.
Sends a stats embed + profile photos as attachments on the same message.
"""

import json
import os
import time
from pathlib import Path
from urllib import request as urllib_request

import config


_USER_AGENT = "HingeAuto/1.0"


def _send_multipart_payload(webhook_url: str, payload: dict,
                             files: list[tuple[str, bytes]]) -> None:
    """Send a Discord webhook payload with optional file attachments.

    Uses multipart/form-data so files appear as message attachments.
    """
    import uuid
    boundary = uuid.uuid4().hex

    body_parts = []

    # payload_json field
    body_parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="payload_json"\r\n'
        f"Content-Type: application/json\r\n\r\n"
        f"{json.dumps(payload)}\r\n"
    )

    # File fields
    for i, (filename, data) in enumerate(files):
        body_parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="files[{i}]"; '
            f'filename="{filename}"\r\n'
            f"Content-Type: image/png\r\n\r\n".encode()
            + data
            + b"\r\n"
        )

    body_parts.append(f"--{boundary}--\r\n".encode())

    body = b"".join(
        p.encode() if isinstance(p, str) else p for p in body_parts
    )

    req = urllib_request.Request(
        webhook_url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": _USER_AGENT,
        },
        method="POST",
    )
    try:
        urllib_request.urlopen(req)
    except urllib_request.HTTPError as e:
        print(f"[report] webhook failed: {e.code} {e.read().decode()[:200]}")


def post_run(likes_sent: int, profiles_seen: int, skips: int,
             total_cost: float, total_duration_s: float,
             liked_profiles: list[dict] | None = None) -> None:
    """Post a stats embed + profile photos to the configured webhook.

    Reads DISCORD_WEBHOOK_URL from the environment. No-op when unset.
    """
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return  # silent no-op

    liked_profiles = liked_profiles or []

    # Collect profile photos from this run
    liked_dir = config.DEBUG_DIR / "liked"
    files: list[tuple[str, bytes]] = []
    profile_texts: list[str] = []

    for i, profile in enumerate(liked_profiles):
        name = profile.get("name", "unknown").capitalize()
        msg = profile.get("message", "") or "(no message)"
        safe = "".join(c for c in name.lower() if c.isalnum()) or "unknown"

        profile_texts.append(f"{i + 1}. **{name}** — {msg}")

        # Find & read the first photo
        if liked_dir.is_dir():
            for folder in sorted(liked_dir.iterdir()):
                if folder.is_dir() and folder.name.endswith(f"_{safe}"):
                    photo = folder / "imgs" / "frame_00.png"
                    if photo.is_file():
                        files.append((f"{safe}_frame_00.png", photo.read_bytes()))
                        break

    embed = {
        "title": "Hinge Auto — Run Complete",
        "color": 0x57F287,
        "fields": [
            {"name": "👀 Seen",  "value": str(profiles_seen), "inline": True},
            {"name": "❤️ Likes", "value": str(likes_sent),    "inline": True},
            {"name": "⏭️ Skip",  "value": str(skips),         "inline": True},
            {
                "name": "Profiles Liked",
                "value": "\n".join(profile_texts) if profile_texts else "None",
                "inline": False,
            },
        ],
        "footer": {"text": f"${total_cost:.2f} · {total_duration_s:.0f}s"},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
    }

    payload = {"embeds": [embed]}
    if files:
        payload["attachments"] = [
            {"id": i, "filename": fn, "description": f"Photo {i + 1}"}
            for i, (fn, _) in enumerate(files)
        ]

    _send_multipart_payload(webhook_url, payload, files)
