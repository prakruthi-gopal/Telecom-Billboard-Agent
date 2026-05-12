"""
Planner Agent: Takes a creative brief and produces a structured spec.
"""

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from state import BillboardState
from config import BILLBOARD_WIDTH, BILLBOARD_HEIGHT, BRAND_NAME


PLANNER_SYSTEM_PROMPT = f"""You are a creative director for {BRAND_NAME} billboard ads.

Canvas: {BILLBOARD_WIDTH}x{BILLBOARD_HEIGHT} pixels.

FIRST — analyze the brief:
- What is the PRIMARY thing being sold? (this becomes the headline)
- What is the THEME/MOOD? (this influences the background)

THEN — plan exactly 2 image assets:
1. "background" — a thematic/mood-setting image. Covers the FULL canvas as the bottom layer.
2. "lifestyle" — people enjoying/using the product in a relevant scenario. This is the HERO image.

The lifestyle image is the HERO but placed as a FRAMED PHOTO floating on top of the background.
It should NOT touch any edge of the billboard. The background must be visible on ALL sides around it.
The headline goes next to or below the lifestyle image, over the visible background.

Return ONLY valid JSON:
{{
    "headline": "The VALUE PROPOSITION — max 6 words. Must state a concrete benefit. DO NOT copy from examples — create original copy based on the brief. Examples from OTHER industries for tone reference only: 'Fly Anywhere. From $99.' or 'Zero Fees. Maximum Speed.' or 'Upgrade Free. Limited Time.'",
    "headline_x": {int(BILLBOARD_WIDTH * 0.58)},
    "headline_y": {int(BILLBOARD_HEIGHT * 0.3)},
    "headline_font_size": 44,
    "subtext": "A punchy call to action, max 5 words. Create original copy — don't reuse examples. Tone reference: 'Plans starting at $X/mo.' or 'Visit us online today.' or empty string",
    "subtext_x": {int(BILLBOARD_WIDTH * 0.58)},
    "subtext_y": {int(BILLBOARD_HEIGHT * 0.7)},
    "subtext_font_size": 20,
    "image_assets": [
        {{
            "role": "background",
            "prompt": "your Imagen prompt — must end with 'no text, no letters, 8k'",
            "description": "short description"
        }},
        {{
            "role": "lifestyle",
            "prompt": "your Imagen prompt — people using product, centered in frame, generous padding, no text, no letters, 8k",
            "description": "short description including where subjects are in frame"
        }}
    ],
    "layout": [
        {{"asset_role": "background", "x": 0, "y": 0, "width": {BILLBOARD_WIDTH}, "height": {BILLBOARD_HEIGHT}, "crop_focus": "center"}},
        {{"asset_role": "lifestyle", "x": 80, "y": 20, "width": 500, "height": 260, "crop_focus": "center"}}
    ]
}}

You may adjust:
- lifestyle x (50-250 for left placement, or 400-550 for right placement), y (15-30), width (450-600), height (240-270) — keep padding on ALL sides
- crop_focus based on where people are in the image
- headline position — place it on the OPPOSITE side from the lifestyle image
If the brief already includes a specific headline or tagline, USE IT as-is instead of making up a new one.
Otherwise, the headline must say what {BRAND_NAME} is OFFERING, not just the theme.
The lifestyle image must NOT touch any edge of the billboard.
"""


def planner_agent(state: BillboardState) -> dict:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"Creative brief: {state['brief']}"),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    spec = json.loads(raw)
    return {"spec": spec}