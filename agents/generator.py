"""
Generator Agent: Produces image assets for the billboard using Imagen.
"""

import os
from google import genai
from google.genai import types

from state import BillboardState, ImageAsset


FAST_MODEL = "imagen-4.0-fast-generate-001"
FALLBACK_MODEL = "imagen-4.0-generate-001"


def _generate_with_fallback(client, prompt: str):
    """Try fast model first. If quota/resource error, fallback to full model."""
    try:
        return client.models.generate_images(
            model=FAST_MODEL,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                output_mime_type="image/png",
                aspect_ratio="16:9",
            ),
        )
    except Exception as e:
        error_str = str(e).lower()
        if any(kw in error_str for kw in ["429", "quota", "resource_exhausted", "rate limit"]):
            print(f"[Generator] Fast model quota hit — falling back to full model")
            return client.models.generate_images(
                model=FALLBACK_MODEL,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    output_mime_type="image/png",
                    aspect_ratio="16:9",
                ),
            )
        raise


def generator_agent(state: BillboardState) -> dict:
    """Generator node in the pipeline. Max 2 assets."""
    spec = state["spec"]
    output_dir = state["output_dir"]
    asset_specs = spec.get("image_assets", [])

    if len(asset_specs) == 0:
        asset_specs = [{
            "role": "background",
            "prompt": "Diverse group of happy people using smartphones, modern setting, natural lighting, no text, 8k",
            "description": "People using phones, centered in frame",
        }]

    if len(asset_specs) > 2:
        background = [a for a in asset_specs if a["role"] == "background"]
        foreground = [a for a in asset_specs if a["role"] != "background"]
        asset_specs = background[:1] + foreground[:1]

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    generated_assets: list[ImageAsset] = []

    for i, asset_spec in enumerate(asset_specs):
        result = _generate_with_fallback(client, asset_spec["prompt"])
        save_path = os.path.join(output_dir, f"asset_{i}_{asset_spec['role']}.png")
        result.generated_images[0].image.save(save_path)
        generated_assets.append({
            "role": asset_spec["role"],
            "path": save_path,
            "description": asset_spec.get("description", ""),
        })

    return {"image_assets": generated_assets}