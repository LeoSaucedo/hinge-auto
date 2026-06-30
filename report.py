"""Post run results to Discord via webhook.

Requires DISCORD_WEBHOOK_URL in the environment. No-op when unset.
Sends profile photos with stats in the first batch, no separate summary.
"""

import json
import os
import time
from pathlib import Path
from urllib import request as urllib_request

import config


_USER_AGENT = "HingeAuto/1.0"
_DISCORD_ATTACHMENT_LIMIT = 10


def _send_multipart_payload(webhook_url: str, payload: dict,
                             files: list[tuple[str, bytes]]) -> None:
    """Send a Discord webhook payload with optional file attachments."""
    import uuid
    boundary = uuid.uuid4().hex

    body_parts = []
    body_parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="payload_json"\r\n'
        f"Content-Type: application/json\r\n\r\n"
        f"{json.dumps(payload)}\r\n"
    )
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


def _send_embed_only(webhook_url: str, embed: dict) -> None:
    """Send a single embed with no file attachments."""
    body = json.dumps({"embeds": [embed]}).encode("utf-8")
    req = urllib_request.Request(
        webhook_url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
        },
        method="POST",
    )
    try:
        urllib_request.urlopen(req)
    except urllib_request.HTTPError as e:
        print(f"[report] webhook embed failed: {e.code} {e.read().decode()[:200]}")


def post_run(likes_sent: int, profiles_seen: int, skips: int,
             total_cost: float, total_duration_s: float,
             liked_profiles: list[dict] | None = None) -> None:
    """Post profile photos with stats in the first batch, no separate summary."""
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return

    liked_profiles = liked_profiles or []

    liked_dir = config.DEBUG_DIR / "liked"
    profile_data: list[dict] = []
    for i, profile in enumerate(liked_profiles):
        name = profile.get("name", "unknown").capitalize()
        msg = profile.get("message", "") or "(no message)"
        folder_name = profile.get("folder")
        photo_bytes = None
        if folder_name and liked_dir.is_dir():
            candidate = liked_dir / folder_name / "imgs" / "frame_00.png"
            if candidate.is_file():
                photo_bytes = candidate.read_bytes()
        if photo_bytes is None:
            safe = "".join(c for c in name.lower() if c.isalnum()) or "unknown"
            if liked_dir.is_dir():
                for folder in sorted(liked_dir.iterdir()):
                    if folder.is_dir() and f"_{safe}" in folder.name:
                        candidate = folder / "imgs" / "frame_00.png"
                        if candidate.is_file():
                            photo_bytes = candidate.read_bytes()
                            break
        profile_data.append({"name": name, "msg": msg, "bytes": photo_bytes})

    if not profile_data:
        embed = {
            "title": "Hinge Auto — Run Complete",
            "color": 0x57F287,
            "fields": [
                {"name": "👀 Seen",  "value": str(profiles_seen), "inline": True},
                {"name": "❤️ Likes", "value": str(likes_sent),    "inline": True},
                {"name": "⏭️ Skip",  "value": str(skips),         "inline": True},
            ],
            "footer": {"text": f"${total_cost:.2f} · {total_duration_s:.0f}s"},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        }
        _send_embed_only(webhook_url, embed)
        return

    # Send profile photos in batches, stats in the first batch
    batches = [
        profile_data[i:i + _DISCORD_ATTACHMENT_LIMIT]
        for i in range(0, len(profile_data), _DISCORD_ATTACHMENT_LIMIT)
    ]

    for batch_idx, batch in enumerate(batches):
        files_batch = [(f"{p['name']}_frame_00.png", p["bytes"])
                       for p in batch if p["bytes"]]

        start_num = batch_idx * _DISCORD_ATTACHMENT_LIMIT + 1
        end_num = start_num + len(batch) - 1
        profile_lines = "\n".join(
            f"{start_num + i}. **{p['name']}** — {p['msg']}"
            for i, p in enumerate(batch)
        )

        if batch_idx == 0:
            embed = {
                "title": f"Hinge Auto{f' ({start_num}-{end_num})' if len(batches) > 1 else ''}",
                "color": 0x57F287,
                "fields": [
                    {"name": "👀 Seen",  "value": str(profiles_seen), "inline": True},
                    {"name": "❤️ Likes", "value": str(likes_sent),    "inline": True},
                    {"name": "⏭️ Skip",  "value": str(skips),         "inline": True},
                    {"name": "Liked", "value": profile_lines, "inline": False},
                ],
                "footer": {"text": f"${total_cost:.2f} · {total_duration_s:.0f}s"},
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            }
        else:
            embed = {
                "title": f"Hinge Auto ({start_num}-{end_num})",
                "color": 0x57F287,
                "fields": [
                    {"name": "Liked", "value": profile_lines, "inline": False},
                ],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            }

        payload = {"embeds": [embed]}
        if files_batch:
            payload["attachments"] = [
                {"id": i, "filename": fn, "description": f"Photo {start_num + i}"}
                for i, (fn, _) in enumerate(files_batch)
            ]
        _send_multipart_payload(webhook_url, payload, files_batch)

        if batch_idx + 1 < len(batches):
            time.sleep(0.5)
