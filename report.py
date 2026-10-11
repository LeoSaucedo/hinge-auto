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
import metrics


_USER_AGENT = "HingeAuto/1.0"
_DISCORD_ATTACHMENT_LIMIT = 10
# Discord caps an embed field *value* at 1024 chars, and the `Liked` field
# carries every liked opener in a single value. Batch against a budget below
# the real cap so a long run splits instead of losing the whole embed.
_DISCORD_FIELD_LIMIT = 1000


def _format_duration(seconds: float) -> str:
    """Duration for the embed footer: "42s", "3m 42s", "1h 5m".

    Seconds are dropped past the hour mark because they stop carrying
    information at that scale — the footer is a glance, and a run that
    overran its slot should read as "1h 14m" at a glance, not "4470s".
    """
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def _footer(total_cost: float, total_duration_s: float,
            avg_fit_score: float) -> dict:
    """Embed footer: cost, duration, average fit, and the judge model — so a
    run's backend is identifiable from Discord when comparing backends."""
    return {
        "text": (
            f"${total_cost:.2f} · {_format_duration(total_duration_s)} · "
            f"avg fit {avg_fit_score:.0f}/100 · {metrics.active_model()}"
        )
    }


def _liked_line(n: int, p: dict) -> str:
    """One line of the `Liked` field, exactly as it will be rendered."""
    return (f"{n}. **{p['name']}** — {p['msg']} "
            f"(fit {p.get('fit_score', 0)}/100)")


def _batch_by_size(profile_data: list[dict]) -> list[list[tuple[int, dict]]]:
    """Split liked profiles into webhook-sized batches, numbered globally.

    Two independent limits apply to one message: Discord takes at most 10
    attachments, and the `Liked` field text rides along in the same embed
    under its 1024-char per-field cap. Bounding only the attachment count
    let a run with long openers overflow the field, and Discord rejects the
    entire embed with `400 {"embeds": ["0"]}` — the stats and all ten photos
    lost with it, while the run still exits 0.

    Lines are measured as rendered so the boundary follows actual opener
    length rather than a guessed profiles-per-batch number. Numbering is
    assigned here, before splitting, so a boundary landing somewhere new
    doesn't renumber the list.
    """
    batches: list[list[tuple[int, dict]]] = []
    current: list[tuple[int, dict]] = []
    used = 0
    for n, p in enumerate(profile_data, start=1):
        line_len = len(_liked_line(n, p))
        separator = 1 if current else 0  # the newline joining the lines
        if current and (
            len(current) >= _DISCORD_ATTACHMENT_LIMIT
            or used + separator + line_len > _DISCORD_FIELD_LIMIT
        ):
            batches.append(current)
            current, used, separator = [], 0, 0
        current.append((n, p))
        used += separator + line_len
    if current:
        batches.append(current)
    return batches


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


def _mention() -> str:
    """Mention prefix for error posts, or "" when no ID is configured.

    This has to ride in the top-level `content`, never in the embed:
    Discord notifies on mentions in a message's content but not on ones
    rendered inside an embed. That is what lets the channel sit at
    "Only @mentions" — silent for every routine post, and loud only for
    the errors that carry this prefix.
    """
    uid = os.environ.get("DISCORD_MENTION_USER_ID", "").strip()
    return f"<@{uid}> " if uid else ""


def post_error(message: str, profiles_seen: int | None = None,
               likes_sent: int | None = None, skips: int | None = None,
               screenshot_path: str | None = None,
               title: str = "❌ Hinge Auto — Run Aborted") -> None:
    """Send a fatal-error embed to the Discord webhook, mentioning the user.

    If screenshot_path is provided, the screenshot is attached as a file.

    The run stats are optional because the crash handler at the bottom of
    main.py has no frame to read them from — omitting the fields beats
    posting a row of zeroes that reads like a real (empty) run."""
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return

    embed = {
        "title": title,
        "color": 0xED4245,
        "description": message,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
    }
    fields = [{"name": name, "value": str(value), "inline": True}
              for name, value in (("👀 Seen", profiles_seen),
                                  ("❤️ Likes", likes_sent),
                                  ("⏭️ Skips", skips))
              if value is not None]
    if fields:
        embed["fields"] = fields

    # Parse "users" only: the mention must ping, and no error post should
    # ever be able to page a role or @everyone.
    allowed_mentions = {"parse": ["users"]}
    content = _mention() + title

    if screenshot_path:
        try:
            screenshot_bytes = Path(screenshot_path).read_bytes()
            payload = {"content": content, "embeds": [embed],
                       "allowed_mentions": allowed_mentions, "attachments": [
                {"id": 0, "filename": "dialog_screenshot.png",
                 "description": "Dialog that blocked the run"}
            ]}
            _send_multipart_payload(
                webhook_url, payload,
                [("dialog_screenshot.png", screenshot_bytes)])
            return
        except Exception as e:
            print(f"[report] failed to attach screenshot: {e}")

    _send_embed_only(webhook_url, embed,
                     content=content, allowed_mentions=allowed_mentions)


def post_crash(tb: str, title: str = "💥 Hinge Auto — Crashed") -> None:
    """Report an unhandled exception, with the traceback tail as the body.

    Shared by main.py's own handler and the run.py launcher, which is the
    only one of the two that can see an import-time failure."""
    post_error(
        f"Unhandled exception — the run died before finishing.\n"
        f"```\n{tb[-1500:]}\n```",
        title=title,
    )


def _send_embed_only(webhook_url: str, embed: dict, content: str = "",
                     allowed_mentions: dict | None = None) -> None:
    """Send a single embed with no file attachments."""
    payload = {"embeds": [embed]}
    if content:
        payload["content"] = content
    if allowed_mentions:
        payload["allowed_mentions"] = allowed_mentions
    body = json.dumps(payload).encode("utf-8")
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
             liked_profiles: list[dict] | None = None,
             avg_fit_score: float = 0.0) -> None:
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
        profile_data.append({
            "name": name,
            "msg": msg,
            "fit_score": profile.get("fit_score", 0),
            "bytes": photo_bytes,
        })

    if not profile_data:
        embed = {
            "title": "Hinge Auto — Run Complete",
            "color": 0x57F287,
            "fields": [
                {"name": "👀 Seen",  "value": str(profiles_seen), "inline": True},
                {"name": "❤️ Likes", "value": str(likes_sent),    "inline": True},
                {"name": "⏭️ Skip",  "value": str(skips),         "inline": True},
            ],
            "footer": _footer(total_cost, total_duration_s, avg_fit_score),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        }
        _send_embed_only(webhook_url, embed)
        return

    # Send profile photos in batches, stats in the first batch
    batches = _batch_by_size(profile_data)

    for batch_idx, batch in enumerate(batches):
        # Keep the profile number with each photo: files_batch drops entries
        # with no bytes, so an index-derived number would drift past a gap.
        photos = [(f"{p['name']}_frame_00.png", p["bytes"], n)
                  for n, p in batch if p["bytes"]]
        files_batch = [(fn, data) for fn, data, _ in photos]

        start_num, end_num = batch[0][0], batch[-1][0]
        profile_lines = "\n".join(_liked_line(n, p) for n, p in batch)

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
                "footer": _footer(total_cost, total_duration_s, avg_fit_score),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            }
        else:
            embed = {
                "title": f"Hinge Auto ({start_num}-{end_num})",
                "color": 0x57F287,
                "fields": [
                    {"name": "Liked", "value": profile_lines, "inline": False},
                ],
                "footer": _footer(total_cost, total_duration_s, avg_fit_score),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            }

        payload = {"embeds": [embed]}
        if files_batch:
            payload["attachments"] = [
                {"id": i, "filename": fn, "description": f"Photo {n}"}
                for i, (fn, _, n) in enumerate(photos)
            ]
        _send_multipart_payload(webhook_url, payload, files_batch)

        if batch_idx + 1 < len(batches):
            time.sleep(0.5)
