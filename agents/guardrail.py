"""
Guardrail Agent: First node in the pipeline.
Validates the user's brief before any other agents run.
"""

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from state import BillboardState


GUARDRAIL_PROMPT = """You are a content safety and relevance validator for a billboard advertisement generation system.

Review the creative brief and determine if it is:
1. SAFE — no profanity, hate speech, self-harm, violence, sexually explicit content
2. APPROPRIATE — suitable for public billboard advertising (visible to all ages)
3. RELEVANT — actually a request for a billboard or advertisement

Return ONLY valid JSON:
{
    "approved": true or false,
    "reason": "Brief explanation",
    "category": "safe" or "profanity" or "harmful" or "explicit" or "off_topic" or "ad_violation"
}

When in doubt, approve — the planner agent will handle creative direction.
"""


def guardrail_agent(state: BillboardState) -> dict:
    """Guardrail node in the pipeline."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    messages = [
        SystemMessage(content=GUARDRAIL_PROMPT),
        HumanMessage(content=f"Creative brief to validate:\n\n{state['brief']}"),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"approved": False, "reason": "Could not validate brief. Please try rephrasing.", "category": "safe"}

    return {
        "guardrail_passed": result.get("approved", False),
        "guardrail_message": result.get("reason", ""),
    }