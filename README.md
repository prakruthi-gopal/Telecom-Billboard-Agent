# Billboard Ad Agent

An agentic system that generates and composes billboard advertisements for any telecom brand.

## Core idea

The billboard canvas is a **2D grid** (1024×300 pixels). Image generation models produce raw assets. The **Editor Agent** treats the canvas as a coordinate space — reasoning about where each asset goes, what size, how to crop — then uses image manipulation tools to compose the final billboard.

## Architecture

```
User Brief → Guardrail → Planner → Generator → Editor (ReAct loop) → Final Billboard
                |                                  ↑        |
           rejected → END                          |   fail + iterations remain
                                                   └────────┘
```

The pipeline is orchestrated in pure Python with a shared `BillboardState` dict passed between agents. Each agent reads what it needs from the state and writes its output back. The editor agent's ReAct loop uses a simple while condition — no framework abstraction hiding the logic. This keeps the agent behavior explicit and debuggable. In production, frameworks like LangGraph or CrewAI would add value for state persistence, parallel execution, and more complex multi-agent workflows.

**Guardrail Agent** — Validates the brief for safety, relevance, and advertising appropriateness. Blocks harmful or off-topic requests before any images are generated (no wasted API credits).

**Planner Agent** — Takes the creative brief and produces a structured spec: what image assets to generate (background, product, lifestyle), where each goes on the canvas (layout coordinates), headline text, brand overlay placement.

**Generator Agent** — Makes one Imagen API call per asset (2 assets = 2 calls). Each asset is saved separately with its role tagged. Includes automatic fallback between Imagen models if quota limits are hit.

**Editor Agent (the hero)** — Implements a ReAct loop (Reason → Act → Observe):
- **Reason**: LLM analyzes the assets, the planner's layout, and any compliance violations from previous iterations. Decides what tools to call with what parameters.
- **Act**: Composes the billboard — creates a fresh canvas, crops/resizes each asset to its target grid region, places them in layer order, adds brand overlay, headline text, and logo.
- **Observe**: Sends the composed billboard to Gemini Vision for brand compliance checking. If it fails, loops back with the violations as context.

Every iteration starts from a fresh canvas to prevent ghosting and double-layering.

## What the Editor Agent actually does (2D grid operations)

This is the part I stumbled on in the interview. The key insight: image generation models are **tools**, not agents. The editor agent **orchestrates** them.

Each placement is a grid operation:
- `place_asset_on_canvas(x=512, y=40, width=380, height=220)` — maps an asset to a rectangular region on the canvas
- `crop_focus="center"` — decides which part of the source image survives the crop to match the target region's aspect ratio
- `resolve_overlap()` — rectangle intersection math to prevent assets from overlapping
- `validate_crop_ratio()` — catches extreme crops that would mangle faces or key content (e.g., cropping a portrait into a wide thin strip = 83% content loss)

These validations are **hardcoded in Python**, not LLM prompt suggestions. The LLM can output any coordinates it wants — the code enforces sanity.

## Tech stack

| Component | Technology |
|-----------|-----------|
| LLM reasoning | Gemini Flash |
| Image generation | Google Imagen (same API key) |
| Orchestration | Pure Python (shared state dict) |
| Image composition | Pillow |
| UI | Streamlit |

The model string for Imagen is a constant — swapping to Nano Banana means changing one line.

## Running locally

```bash
git clone <repo-url>
cd billboard_agent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your API key to `.env`:
```
GEMINI_API_KEY=your_key_here
```

Run:
```
streamlit run app.py
```

## Trade-offs: agency vs consistency

Building this over the weekend revealed a core tension — giving the LLM full creative freedom produced inconsistent layouts (50/50 splits, overlapping text, assets off-canvas), but hardcoding everything removed the point of having an agent. Here's where I landed:

**What the LLM decides (creative agency):**
- Headline and subtext content (what to say, how to sell)
- Image generation prompts (what scene, what people, what mood)
- Lifestyle image positioning within bounded ranges
- Crop focus direction (which part of the image to preserve)
- Headline and subtext placement
- Whether to include subtext at all
- Canvas background color

**What's hardcoded (infrastructure/brand constraints):**
- Background always covers the full canvas, blurred to not compete with the hero
- Logo always top-right, fixed size (brand constant, not a creative decision)
- Max 2 image assets (tested 1, 2, and 3 — 2 consistently produced the best results on a 300px-tall canvas)
- Fresh canvas every iteration (prevents ghosting from previous layers)
- Text auto-wrapping and font-size reduction (rendering safety, not creative choice)
- Subtext always positioned below headline (prevents overlap)
- Feathered edges on lifestyle image (visual polish)
- Overlap prevention, boundary clamping, crop validation (safety nets)

**Why this balance:**
The hardcoded constraints are things a human designer wouldn't freestyle either — logo placement, canvas dimensions, text rendering rules. These are brand and physics constraints. The LLM handles what actually matters for a billboard: what to show, what to say, and how to frame it.

The main limitation is spatial reasoning. LLMs reason about layout from text descriptions without seeing the images, which leads to inconsistent compositions. In production, feeding the actual generated images into a multimodal model for layout decisions — rather than planning layout blind — would significantly improve consistency.

## Production considerations (not built, but thought about)

- **Brand guidelines from config store** — currently hardcoded as a string constant. In production, load from a database or brand management system.
- **Agent frameworks (LangGraph, CrewAI, AutoGen)** — for state persistence, conditional routing, and multi-agent coordination at scale. For this POC, pure Python orchestration keeps the ReAct pattern explicit and the agent logic readable.
- **Persistent asset storage** — currently uses temp directories. Production would use S3/GCS.
- **Layout balance heuristics** — programmatic checks for visual weight distribution across the canvas.
- **A/B variant generation** — generate multiple billboard variants from one brief for testing.

## Project structure

```
billboard_agent/
├── app.py                  # Streamlit UI — standalone mode (agents in-process)
├── state.py                # Shared state schema
├── config.py               # Brand guidelines + constants
├── tools.py                # Pillow tools + hard validations
├── requirements.txt
├── packages.txt            # System dependencies (fonts)
├── .env                    # API key (gitignored)
└── agents/
    ├── guardrail.py        # Safety + relevance check
    ├── planner.py          # Brief → structured spec
    ├── generator.py        # Spec → Imagen assets
    └── editor.py           # ReAct composition loop
```

## Running

**Standalone (Streamlit Cloud or local):**
```bash
streamlit run app.py
```
Runs everything in one process. No backend needed.

`app.py` runs the pipeline directly (agents in-process).
