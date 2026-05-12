"""
Billboard Ad Agent — Streamlit UI

Shows the full agentic pipeline:
1. User enters a creative brief
2. Planner agent produces a structured spec
3. Generator agent creates multiple image assets via Imagen
4. Editor agent composes them into a billboard (with ReAct loop)
5. Final billboard displayed with full edit history
"""

import os
import json
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Support both .env (local) and Streamlit secrets (cloud deployment)
if not os.environ.get("GEMINI_API_KEY"):
    try:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.error(
            "Missing GEMINI_API_KEY. "
            "Add it to `.env` (local) or Streamlit Secrets (cloud)."
        )
        st.stop()

from state import BillboardState
from agents.planner import planner_agent
from agents.generator import generator_agent
from agents.editor import editor_agent
from config import BILLBOARD_WIDTH, BILLBOARD_HEIGHT


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Billboard Ad Agent",
    page_icon="🎨",
    layout="wide",
)

st.title("Billboard Ad Agent")
st.markdown(
    "An agentic system that generates and composes billboard advertisements. "
    "The **Editor Agent** uses a ReAct loop to compose multiple image assets, "
    "check brand compliance, and iterate until the billboard passes."
)

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    max_iterations = st.slider("Max editor iterations", 1, 5, 3)
    st.markdown("---")
    st.markdown(
        "**Tech stack**\n"
        "- Gemini 2.5 Flash (reasoning)\n"
        "- Google Imagen (image generation)\n"
        "- Pure Python (orchestration)\n"
        "- Pillow (image composition)\n"
    )
    st.markdown("---")
    st.caption(
        f"Canvas: {BILLBOARD_WIDTH}x{BILLBOARD_HEIGHT}px "
        f"(~3.4:1 bulletin ratio)"
    )

# ---------------------------------------------------------------------------
# Brief input — wrapped in a form so button activates on submit
# ---------------------------------------------------------------------------
with st.form("brief_form", clear_on_submit=False):
    brief = st.text_area(
        "Enter your creative brief",
        placeholder="e.g., Holiday promotion billboard — family audience, winter theme, showcase a new smartphone",
        height=100,
        key="brief_input",
    )
    run_button = st.form_submit_button("Generate Billboard", type="primary")

if run_button and brief:
    # Clear previous results
    st.session_state.pop("final_state", None)

    # Create temp dir for this run
    output_dir = tempfile.mkdtemp(prefix="billboard_")

    # Initialize state
    state: BillboardState = {
        "brief": brief,
        "guardrail_passed": None,
        "guardrail_message": None,
        "spec": None,
        "image_assets": [],
        "current_image_path": None,
        "edit_history": [],
        "compliance_status": None,
        "compliance_violations": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "output_dir": output_dir,
    }

    # ---- STEP 0: Guardrail ----
    with st.status("🛡️ Guardrail Agent — validating brief...", expanded=False) as status:
        try:
            from agents.guardrail import guardrail_agent
            guardrail_result = guardrail_agent(state)
            state.update(guardrail_result)

            if state["guardrail_passed"]:
                status.update(label="✅ Guardrail — brief approved", state="complete")
            else:
                status.update(label="🛡️ Guardrail — brief not accepted", state="complete")
                st.markdown("---")
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        border: 1px solid #e94560;
                        border-radius: 12px;
                        padding: 32px;
                        text-align: center;
                        margin: 20px 0;
                    ">
                        <div style="font-size: 48px; margin-bottom: 12px;">🛡️</div>
                        <h3 style="color: #e94560; margin: 0 0 12px 0;">Brief not accepted</h3>
                        <p style="color: #a8a8b3; font-size: 16px; margin: 0 0 20px 0;">
                            {state['guardrail_message']}
                        </p>
                        <p style="color: #6c6c80; font-size: 14px; margin: 0;">
                            Please enter a valid billboard advertisement brief and try again.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.stop()

        except Exception as e:
            status.update(label="❌ Guardrail Agent failed", state="error")
            st.error(f"Guardrail error: {e}")
            st.stop()

    # ---- STEP 1: Planner ----
    with st.status("🧠 Planner Agent — analyzing brief...", expanded=False) as status:
        try:
            planner_result = planner_agent(state)
            state.update(planner_result)
            status.update(label="✅ Planner Agent — spec ready", state="complete")

            st.markdown(f"**Headline:** {state['spec'].get('headline', 'N/A')}")
            subtext = state['spec'].get('subtext', '')
            if subtext:
                st.markdown(f"**Subtext:** {subtext}")

            assets_planned = state["spec"].get("image_assets", [])
            st.markdown(f"**Assets planned:** {len(assets_planned)}")
            for a in assets_planned:
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• *{a['role']}* — {a.get('description', '')}")

        except Exception as e:
            status.update(label="❌ Planner Agent failed", state="error")
            st.error(f"Planner error: {e}")
            st.stop()

    # ---- STEP 2: Generator ----
    with st.status("🎨 Generator Agent — creating image assets with Imagen...", expanded=False) as status:
        try:
            generator_result = generator_agent(state)
            state.update(generator_result)
            status.update(label=f"✅ Generator Agent — {len(state['image_assets'])} assets created", state="complete")

            asset_cols = st.columns(len(state["image_assets"]))
            for i, asset in enumerate(state["image_assets"]):
                with asset_cols[i]:
                    st.image(asset["path"], caption=asset['role'], use_container_width=True)

        except Exception as e:
            status.update(label="❌ Generator Agent failed", state="error")
            st.error(f"Generator error: {e}")
            st.stop()

    # ---- STEP 3: Editor (ReAct loop) ----
    editor_container = st.container()

    with editor_container:
        for iteration in range(1, max_iterations + 1):
            with st.status(
                f"🔄 Editor iteration {iteration}/{max_iterations}...",
                expanded=False,
            ) as status:
                try:
                    editor_result = editor_agent(state)
                    state.update(editor_result)

                    # Show what the editor did this iteration
                    iter_edits = [e for e in state["edit_history"] if e.startswith(f"[iter {iteration}]")]

                    if state["compliance_status"] == "pass" or state["iteration_count"] >= max_iterations:
                        label = "compliance passed" if state["compliance_status"] == "pass" else "max iterations reached"
                        status.update(label=f"✅ Editor — {label} (iteration {iteration})", state="complete")

                        st.markdown(f"**Compliance:** ✅ {state['compliance_status'].upper()}")
                        with st.expander(f"Composition steps (iteration {iteration})"):
                            for edit in iter_edits:
                                # Clean up the log line for display
                                clean = edit.split("] ", 1)[1] if "] " in edit else edit
                                st.markdown(f"• {clean}")
                        break
                    else:
                        status.update(label=f"⚠️ Editor iteration {iteration} — refining...", state="complete")

                        st.markdown(f"**Compliance:** ⚠️ FAIL — {len(state['compliance_violations'])} violation(s)")
                        with st.expander(f"Details (iteration {iteration})"):
                            st.markdown("**Violations:**")
                            for v in state["compliance_violations"]:
                                st.markdown(f"• {v}")
                            st.markdown("**Actions taken:**")
                            for edit in iter_edits:
                                clean = edit.split("] ", 1)[1] if "] " in edit else edit
                                st.markdown(f"• {clean}")

                except Exception as e:
                    status.update(label=f"❌ Editor iteration {iteration} failed", state="error")
                    st.error(f"Editor error: {e}")
                    break

    # ---- Final result ----
    # Save to session state so it persists across reruns (e.g., download button click)
    st.session_state["final_state"] = dict(state)

# Show final result from session state (persists across reruns)
if "final_state" in st.session_state:
    state = st.session_state["final_state"]
    st.markdown("---")
    st.subheader("Final Billboard")

    # Auto-scroll to the billboard
    st.markdown('<div id="final-billboard"></div>', unsafe_allow_html=True)
    st.markdown(
        '<script>document.getElementById("final-billboard").scrollIntoView({behavior: "smooth"});</script>',
        unsafe_allow_html=True,
    )

    if state["current_image_path"] and os.path.exists(state["current_image_path"]):
        st.image(state["current_image_path"], use_container_width=True)

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            with open(state["current_image_path"], "rb") as f:
                st.download_button(
                    label="Download Billboard",
                    data=f.read(),
                    file_name="bell_billboard.png",
                    mime="image/png",
                )
        with col2:
            st.metric("Iterations", state["iteration_count"])
        with col3:
            st.metric("Compliance", state["compliance_status"].upper() if state["compliance_status"] else "N/A")

        with st.expander("Edit history"):
            for edit in state["edit_history"]:
                st.text(edit)
    else:
        st.error("No billboard was produced. Check the errors above.")