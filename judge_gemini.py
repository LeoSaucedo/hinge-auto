"""Gemini backend for HingeAuto judging.

Sends profile screenshots to a Gemini vision model with a forced
function call to extract a structured Decision. Same interface as `judge.py`.

Usage: set JUDGE_BACKEND = "gemini" in config.py, add GEMINI_API_KEY to .env.
"""

import os

from dotenv import load_dotenv

from google import genai
from google.genai import types

import config
from judge_common import (
    DECIDE_INPUT_SCHEMA,
    Decision,
    build_system_prompt,
    enforce_premade_verbatim,
)


DECIDE_DECLARATION = types.FunctionDeclaration(
    name="submit_decision",
    description="Submit a like/skip decision for this Hinge profile.",
    parameters=DECIDE_INPUT_SCHEMA,
)


def _image_part(png_bytes: bytes) -> types.Part:
    return types.Part(
        inline_data=types.Blob(mime_type="image/png", data=png_bytes)
    )


def judge(frames: list[bytes]) -> Decision:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Add it to .env or export it.")

    client = genai.Client(api_key=api_key)
    model = getattr(config, "GEMINI_MODEL", "gemini-3.1-flash-lite")

    parts = [_image_part(f) for f in frames]
    parts.append(types.Part(
        text=(
            f"Above are {len(frames)} screenshots of one Hinge profile, in "
            "order from top to bottom. Decide whether to like or skip."
        )
    ))

    response = client.models.generate_content(
        model=model,
        contents=types.Content(role="user", parts=parts),
        config=types.GenerateContentConfig(
            system_instruction=build_system_prompt(),
            temperature=0.2,
            max_output_tokens=2000,
            tools=[types.Tool(function_declarations=[DECIDE_DECLARATION])],
        ),
    )

    if not response.candidates:
        raise RuntimeError(f"No candidates. feedback={response.prompt_feedback}")

    candidate = response.candidates[0]
    if not candidate.content or not candidate.content.parts:
        raise RuntimeError(f"No content. finish_reason={candidate.finish_reason}")

    for part in candidate.content.parts:
        if part.function_call and part.function_call.name == "submit_decision":
            args = {k: v for k, v in part.function_call.args.items()}
            decision = Decision(**args)
            enforce_premade_verbatim(decision)
            return decision

    raise RuntimeError(
        f"No submit_decision call. finish_reason={candidate.finish_reason}"
    )
