<p align="center">
  <strong>HINGEAUTO</strong>
</p>

<p align="center">
  <strong>An LLM that swipes Hinge for you.</strong><br/>
  A stitched-vision + LLM-judge loop that drives a real Hinge install on an Android device via ADB — it reads each profile, judges it against <em>your</em> rubric, and skips or likes with a personalized opener.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-FF4FD8.svg"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-FF4FD8?logo=python&logoColor=white">
  <img alt="Judge: Claude, Gemini, or Ollama" src="https://img.shields.io/badge/judge-Claude%20%C2%B7%20Gemini%20%C2%B7%20Ollama-FF4FD8">
  <img alt="Drives Android via ADB" src="https://img.shields.io/badge/device-Android%20%C2%B7%20ADB-FF4FD8">
  <a href="#-read-this-first"><img alt="Violates Hinge ToS — use at your own risk" src="https://img.shields.io/badge/%E2%9A%A0-violates%20Hinge%20ToS-red"></a>
</p>

<p align="center">
  <a href="#-read-this-first">Read this first</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#failure-alerts">Failure alerts</a> ·
  <a href="#writing-your-own-mode">Modes</a> ·
  <a href="#backends">Backends</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#whats-improved">What's improved</a> ·
  <a href="LICENSE">License</a>
</p>

---

HingeAuto is what you get when you point a vision-LLM at a phone screen and let
it date for you. An Android device runs a real Hinge install; this repo drives
it over ADB, judging every profile against a rubric **you** write and acting on
the verdict — skip, or like with a personalized opener. It runs on **Claude**,
**Gemini**, or a local **Ollama** model. The fun part is the AI engineering —
stitched-frame vision plus a forced structured decision; the swiping is just
the demo.

## ⚠ Read this first

**This project automates Hinge, which violates Hinge's Terms of Service.**
Real risk of account ban with no appeal. Treat this as an educational toy
for a single throwaway account, not a dating strategy.

- **One account only.**
- **Get Hinge+ if you're going to use this seriously.** It removes the daily
  like cap, and this repo is basically the most efficient way to use the
  subscription — the bot does the liking for you, so you get full value without
  ever opening the app.
- **No warranty. No support.** Your account, your problem.
- The repo exists because automating a stitched-vision + LLM-judge loop on a
  phone UI is an interesting AI engineering exercise — not because anyone
  thinks bot-swiping is good dating advice.

## What it does

Drives an Android device running Hinge through ADB. For each profile it
scrolls top-to-bottom, screenshots the frames, asks a vision LLM to judge
against a user-defined rubric, and either taps Skip or taps the heart and
types a personalized opener.

## Quickstart

### 1. Set up Android device with ADB

You need an Android phone or emulator with ADB debugging enabled and Hinge
installed. For a real phone:

1. Enable **Developer options** → **USB debugging** on the phone.
2. Connect via USB or set up ADB over TCP/IP:
   ```bash
   adb tcpip 5555
   adb connect <phone-ip>:5555
   ```
3. Confirm with `adb devices` — should show your phone as `device`.

### 2. Install repo dependencies

```bash
git clone https://github.com/LeoSaucedo/hinge-auto.git
cd hinge-auto
pip install -r requirements.txt
```

### 3. Configure your `.env`

```bash
cp .env.example .env
```

Add at minimum an API key for your chosen backend (see [Backends](#backends)).

### 4. Pick a mode

Three example rubrics ship in `modes/` (see [Writing your own mode](#writing-your-own-mode)).
Set `ACTIVE_MODE` in `config.py` to the file's `NAME` field.

### 5. Run

```bash
python run.py
```

Watch the first few decisions print live. If a decision or opener looks
wrong: Ctrl-C, edit `PREFERENCES` in your mode file, re-run.

`run.py` is a thin wrapper over `main.py` that also catches failures
raised before the loop starts — see [Failure alerts](#failure-alerts).
`python main.py` still works if you'd rather run the loop directly.

## Failure alerts

An aborted run is not a success, and a silent one is worse. Two things
cover that:

- **`main()` exits 1 on abort, 0 on a clean run.** `run_random_window.sh`
  logs it as `Done (exit N)`, so `cron.log` distinguishes a run that
  finished from one that died halfway.
- **Failures notify; routine posts stay quiet.** Set
  `DISCORD_MENTION_USER_ID` in `.env` to your own user ID (Discord
  Developer Mode → Copy User ID) and set the channel's notification
  setting to **Only @mentions**. "Run Complete" embeds still arrive but
  make no noise; errors — loading-screen give-up, dialog recovery
  exhausted, fatal judge error, network outage, or an unhandled crash —
  carry a real mention and ping. Leave the variable unset for no mention.

The mention rides in the message *content* because that's the only place
Discord notifies from; one rendered inside an embed is inert.
`allowed_mentions` is pinned to `users`, so no error post can page a role
or `@everyone`.

`run.py` is the cron entry point and the outer crash net. `main.py`
reports its own runtime crashes, but it can't report an import-time
failure — its module body fails before its handler exists — so `run.py`
imports it inside a `try`. Launched that way main's own handler never
runs, so a crash can't double-post.

## Writing your own mode

Three example rubrics ship in `modes/`:

- `example_lenient.py` — default-LIKE, generic. Good starting point.
- `example_strict.py` — default-SKIP, generic. The antonym.
- `carlos.py` — the author's personal mode with custom voice settings.

To write your own:

1. Copy `modes/template.py.example` to `modes/<your_name>.py`.
2. Edit `PREFERENCES` to describe what should and shouldn't get a like.
3. Optionally set `MESSAGE_VOICE` to one of the templates under `voice/`.
4. Optionally populate `PREMADES` with verbatim openers.
5. Point `ACTIVE_MODE` in `config.py` at the new file, or pass `--mode <your_name>`.

## Backends

The judge pipeline supports four interchangeable backends set via
`JUDGE_BACKEND` in `config.py`. All share the same system prompt, schema, and
`Decision` shape (in `judge_common.py`) — only the API call differs.

### `"anthropic"` (default, best quality)

Uses Claude with vision + forced tool calling. Best quality decisions and
best opener writing. Roughly **$0.02–$0.05 per profile** on Sonnet.

Setup: `ANTHROPIC_API_KEY` in `.env`.

### `"gemini"` (cheap)

Uses Google Gemini via Gemini API. Good quality at lower cost.
**$0.00025–$0.0015 per profile** on Flash Lite.

Setup: `GEMINI_API_KEY` in `.env`. Override model via `GEMINI_MODEL`.

### `"deepseek"` (cheap, OpenAI-compatible)

Uses DeepSeek's OpenAI-compatible API (`deepseek-flash`, vision-capable)
with a forced tool call. Measured **~$0.0015–$0.003 per profile** on a
7-frame profile (off-peak rates are half the peak ones) — roughly a tenth
of the Anthropic backend's cost, and cheaper than the mid-tier Gemini
flash models.

Setup: `DEEPSEEK_API_KEY` in `.env`. Override model via `DEEPSEEK_MODEL`
(only `deepseek-flash` has vision — `deepseek-v4-pro` does not).
`DEEPSEEK_THINKING=true` trades the guaranteed tool call for extra
reasoning; see the header of `judge_deepseek.py` for the tradeoff.

### `"ollama"` (free — local or cloud)

Uses an open-weight vision model through Ollama. No per-token cost.

Setup: `OLLAMA_API_KEY` + `OLLAMA_HOST` for the cloud tier, or local
Ollama at `http://localhost:11434`.

## Architecture

```
ADB capture    →  frame stitching  →  LLM judge         →  action
   adb.py          config / main       judge.py              main.py
                                       judge_deepseek.py     adb.py
                                       judge_gemini.py
                                       judge_ollama.py
                                       vision.py
                                       judge_common.py
```

### Module breakdown

| Module | Role |
|---|---|
| **`adb.py`** | Wraps the `adb` CLI: screenshot, tap, swipe, type, keyboard dismiss. Handles device IME failures gracefully. |
| **`main.py`** | The orchestration loop. For each profile: scroll-to-top, capture N frames, run through judge, then skip or like + message. Error handling with screenshot capture, recovery via skip. Every abort sets `abort_reason`, consumed once at the end so a run posts exactly one webhook — error or success — and exits 1. |
| **`run.py`** | Cron entry point and outer crash net. Imports `main` inside a `try` so an import-time failure still reaches Discord, then passes `main()`'s exit code through. |
| **`judge_common.py`** | Backend-agnostic pipeline: system prompt template, JSON tool schema, `Decision` dataclass, voice resolver, and `load_backend()` dispatcher. |
| **`judge.py`** | Anthropic Claude backend — vision + forced tool call. |
| **`judge_deepseek.py`** | DeepSeek backend — vision + forced tool call (OpenAI-compatible REST, no SDK). |
| **`judge_gemini.py`** | Google Gemini backend — vision + function declaration. |
| **`judge_ollama.py`** | Ollama backend (cloud or local). |
| **`vision.py`** | Finds UI elements via pixel-level template matching: heart icon on photo 1, Send Like button, comment input area. |
| **`modes/`** | Rubric files. Loaded by `config._apply_mode()` which populates `PREFERENCES`, `AGE_MIN/MAX`, `MESSAGE_VOICE`, and `PREMADES`. |
| **`voice/`** | Opener-tone templates that modes can reference by name. |
| **`metrics.py`** | Tracks per-profile cost (model-aware pricing), timing, and writes JSONL to `debug/session_log.jsonl`. |
| **`report.py`** | Discord webhook reporting with batched attachments (10 per message) — stats embed + profile photos. Error and crash posts @mention the configured user in the top-level `content`, which is the only place Discord notifies from. |
| **`config.py`** | All settings with `.env` override support via `_apply_env_overrides()`. |
| **`matches_scan.py`** | Scrapes the Matches tab via a separate Claude vision pass — for analytics, not for the swipe loop. |
| **`scan_self.py`** | Captures your own profile and asks the judge for improvement suggestions. The one feature that doesn't violate Hinge ToS. |
| **`run_random_window.sh`** | Cron wrapper with 0-20min random jitter for session randomization. |

### Session flow

1. ADB captures 7 profile screenshots top-to-bottom
2. Frames + system prompt sent to the active judge backend
3. Judge returns structured `Decision` (like/skip, confidence, reasoning, message)
4. `do_like()` taps heart, opens compose card, types opener, taps Send Like
5. Keyboard issues handled automatically — dismissed if covering UI, NPEs caught gracefully
6. Metrics logged, Discord stats posted
7. Loops until like cap hit or profiles exhausted

## What's improved

This repo is a fork and continuation of the original
[TerraByte-Dev/hinge-auto](https://github.com/TerraByte-Dev/hinge-auto). Major
improvements beyond the original:

- **Gemini backend** — cheaper alternative to Claude with competitive quality
- **Webhook batching** — Discord embeds split into 10-attachment batches to stay within rate limits
- **Keyboard dismiss** — automatic keyboard handling via `dumpsys input_method` detection
- **NPE recovery** — `input_text` crashes on certain devices/IMEs caught gracefully with keyboard dismiss
- **Cost tracking** — per-profile and per-session cost estimates in metrics and Discord reports
- **Randomized cron** — 0-20min jitter per session for anti-detection
- **Session caps** — configurable min/max likes per session with random roll
- **Error screenshots** — debug captures on failure saved to `debug/errors/` (override the whole debug output dir via `DEBUG_DIR` in `.env`)
- **Failure alerting** — error posts @mention a configured user so the channel can sit at "Only @mentions"; aborts post exactly once (no red-and-green pair) and exit non-zero, and crashes are reported instead of only landing in `cron.log`
- **Bot detection evasion** — randomized scroll distance, tap jitter, action delays, session timing

## License

MIT. See [`LICENSE`](LICENSE).

## Credits

Originally created by [TerraByte Solutions LLC](https://github.com/TerraByte-Dev) —
the stitched-vision + tool-calling pipeline is their design. This fork extends
it with additional backends, reliability improvements, and anti-detection
measures.

Built with ❤️ by [LeoSaucedo](https://github.com/LeoSaucedo)
