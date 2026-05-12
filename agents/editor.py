"""
Editor Agent: The hero of the system.
ReAct loop: REASON → ACT → OBSERVE
"""

import json
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from state import BillboardState
from config import BRAND_GUIDELINES, BILLBOARD_WIDTH, BILLBOARD_HEIGHT, BRAND_NAME
from tools import (
    create_canvas,
    place_asset_on_canvas,
    add_text_overlay,
    add_subtext,
    place_logo,
    apply_brand_overlay,
    resolve_overlap,
)


EDITOR_REASON_PROMPT = f"""You are an expert billboard compositor for {BRAND_NAME}.

You have image assets to compose onto a {BILLBOARD_WIDTH}x{BILLBOARD_HEIGHT} pixel canvas.

Available tools:
1. create_canvas — Create blank canvas. Parameters: bg_color [r, g, b]
2. place_asset — Place an image on the canvas. Parameters: asset_role (str), x (int), y (int), width (int), height (int), crop_focus ("center"|"left"|"right"|"top"|"bottom")
3. text_overlay — Add headline. Parameters: headline (str), x (int), y (int), font_size (36-56)
4. subtext — Add secondary text. Parameters: text (str), x (int), y (int), font_size (18-28)
5. logo — Place brand logo. No parameters needed (position is fixed).

Return ONLY valid JSON:
{{
    "reasoning": "Your thinking about composition",
    "actions": [
        {{"tool": "tool_name", "parameters": {{}}, "why": "reason"}}
    ]
}}

EVERY iteration must include the FULL sequence:
create_canvas → place background (full canvas) → place lifestyle (on top) → text_overlay → subtext (if needed) → logo

COMPOSITION RULES:
- Background: ALWAYS x=0, y=0, width={BILLBOARD_WIDTH}, height={BILLBOARD_HEIGHT}. No exceptions.
- Lifestyle: A FRAMED PHOTO floating on top of the background. Must NOT touch any edge of the billboard.
  Acceptable ranges: x=50-250, y=15-30, width=450-600, height=240-270
  The background must be visible on ALL four sides around the lifestyle image.
- Headline: Place where the background is visible (usually the opposite side from the lifestyle image)
- Use the spec's planned layout coordinates as your starting point. Adjust only if compliance violations require it.
"""


COMPLIANCE_PROMPT = f"""You are a brand compliance reviewer for {BRAND_NAME} billboards.

{BRAND_GUIDELINES}

Review the image and return ONLY valid JSON:
{{
    "status": "pass" or "fail",
    "violations": ["list of issues"],
    "suggestions": ["fixes"]
}}

PASS the billboard if it looks reasonably professional and readable.
Only FAIL for these critical issues:
- Headline text is completely missing or unreadable
- Billboard is blank or has a completely broken layout
- Offensive or inappropriate content

DO NOT fail for:
- Text being close to edges (as long as it's readable)
- Subtext being partially near the bottom edge
- Minor spacing or alignment imperfections
- The lifestyle image not being perfectly centered
"""


def _get_asset_path(state: BillboardState, role: str) -> str:
    for asset in state["image_assets"]:
        if asset["role"] == role:
            return asset["path"]
    return None


def _reason_about_edits(state: BillboardState) -> dict:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    asset_info = [
        {"role": a["role"], "description": a["description"]}
        for a in state["image_assets"]
    ]

    context = {
        "planned_layout": state["spec"].get("layout", []),
        "planned_headline": {
            "text": state["spec"].get("headline", ""),
            "x": state["spec"].get("headline_x"),
            "y": state["spec"].get("headline_y"),
            "font_size": state["spec"].get("headline_font_size", 48),
        },
        "planned_subtext": {
            "text": state["spec"].get("subtext", ""),
            "x": state["spec"].get("subtext_x"),
            "y": state["spec"].get("subtext_y"),
            "font_size": state["spec"].get("subtext_font_size", 24),
        },
        "available_assets": asset_info,
        "edits_completed": state["edit_history"],
        "iteration": state["iteration_count"],
        "compliance_violations": state["compliance_violations"],
        "is_first_pass": len(state["edit_history"]) == 0,
    }

    messages = [
        SystemMessage(content=EDITOR_REASON_PROMPT),
        HumanMessage(content=f"Current state:\n{json.dumps(context, indent=2)}"),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    return json.loads(raw)


def _execute_action(action: dict, state: BillboardState, current_path: str, edit_history: list[str], text_bottom_y: int = None) -> dict:
    tool = action["tool"]
    params = action.get("parameters", {})
    output_dir = state["output_dir"]

    if tool == "create_canvas":
        bg_color = tuple(params.get("bg_color", (0, 61, 165)))
        return create_canvas(output_dir, edit_history, bg_color)

    elif tool == "place_asset":
        asset_role = params.get("asset_role", "background")
        asset_path = _get_asset_path(state, asset_role)
        if not asset_path:
            return {"new_image_path": current_path, "edit_description": f"Asset '{asset_role}' not found — skipped"}

        if asset_role == "background":
            x, y = 0, 0
            w, h = BILLBOARD_WIDTH, BILLBOARD_HEIGHT
            mode = "stretch"
        else:
            x = params.get("x", 0)
            y = params.get("y", 0)
            w = params.get("width", BILLBOARD_WIDTH)
            h = params.get("height", BILLBOARD_HEIGHT)
            mode = "crop"

        return place_asset_on_canvas(
            canvas_path=current_path, asset_path=asset_path,
            output_dir=output_dir, edit_history=edit_history,
            x=x, y=y, width=w, height=h,
            crop_focus=params.get("crop_focus", "center"), resize_mode=mode,
        )

    elif tool == "brand_overlay":
        return apply_brand_overlay(
            current_path, output_dir, edit_history,
            region=params.get("region", "bottom-strip"), opacity=params.get("opacity", 0.3),
        )

    elif tool == "text_overlay":
        return add_text_overlay(
            current_path, output_dir, edit_history,
            headline=params.get("headline", "Your Brand Here"),
            x=params.get("x", int(BILLBOARD_WIDTH * 0.1)),
            y=params.get("y", int(BILLBOARD_HEIGHT * 0.4)),
            font_size=params.get("font_size", 48),
        )

    elif tool == "subtext":
        subtext_y = params.get("y", int(BILLBOARD_HEIGHT * 0.65))
        if text_bottom_y is not None:
            subtext_y = max(subtext_y, text_bottom_y + 8)
        if subtext_y > BILLBOARD_HEIGHT - 15:
            return {"new_image_path": current_path, "edit_description": "Subtext skipped — no room below headline"}
        return add_subtext(
            current_path, output_dir, edit_history,
            text=params.get("text", ""), x=params.get("x", int(BILLBOARD_WIDTH * 0.1)),
            y=subtext_y, font_size=params.get("font_size", 24),
        )

    elif tool == "logo":
        return place_logo(current_path, output_dir, edit_history)

    else:
        return {"new_image_path": current_path, "edit_description": f"Unknown tool '{tool}' — skipped"}


def _check_compliance(image_path: str) -> dict:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    messages = [
        SystemMessage(content=COMPLIANCE_PROMPT),
        HumanMessage(content=[
            {"type": "text", "text": "Review this billboard image for brand compliance."},
            {"type": "image_url", "image_url": f"data:image/png;base64,{image_data}"},
        ]),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"status": "fail", "violations": ["Could not parse compliance response"], "suggestions": []}


def editor_agent(state: BillboardState) -> dict:
    """Editor node. One iteration of the ReAct loop. Every iteration starts from a fresh canvas."""
    iteration = state["iteration_count"] + 1

    plan = _reason_about_edits(state)
    actions = plan.get("actions", [])

    has_canvas = any(a.get("tool") == "create_canvas" for a in actions)
    if not has_canvas:
        actions.insert(0, {"tool": "create_canvas", "parameters": {"bg_color": [0, 61, 165]}, "why": "Fresh canvas"})

    current_path = None
    edit_history = list(state["edit_history"])
    placed_regions: list[dict] = []
    text_bottom_y = None

    for action in actions:
        tool = action.get("tool", "")
        params = action.get("parameters", {})

        if tool == "place_asset" and params.get("asset_role") != "background":
            proposed = {"x": params.get("x", 0), "y": params.get("y", 0),
                        "width": params.get("width", 200), "height": params.get("height", 200)}
            adjusted = resolve_overlap(proposed, placed_regions)
            if adjusted != proposed:
                action = dict(action)
                action["parameters"] = dict(params)
                action["parameters"].update({"x": adjusted["x"], "y": adjusted["y"],
                                             "width": adjusted["width"], "height": adjusted["height"]})
            placed_regions.append(adjusted)

        if tool == "text_overlay":
            headline = params.get("headline", "")
            fs = params.get("font_size", 48)
            placed_regions.append({"x": params.get("x", 0), "y": params.get("y", 0),
                                   "width": len(headline) * int(fs * 0.6), "height": int(fs * 1.3)})

        result = _execute_action(action, state, current_path, edit_history, text_bottom_y)
        current_path = result["new_image_path"]

        if tool == "text_overlay" and "text_bottom_y" in result:
            text_bottom_y = result["text_bottom_y"]

        edit_history.append(f"[iter {iteration}] {result['edit_description']} | reason: {action.get('why', 'N/A')}")

    compliance = _check_compliance(current_path)

    return {
        "current_image_path": current_path,
        "edit_history": edit_history,
        "compliance_status": compliance.get("status", "fail"),
        "compliance_violations": compliance.get("violations", []),
        "iteration_count": iteration,
    }