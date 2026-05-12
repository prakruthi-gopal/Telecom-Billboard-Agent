"""
Image editing tools for the Editor Agent.

Tools:
- create_canvas: blank billboard canvas
- place_asset_on_canvas: crop/resize/place an image on the canvas
- add_text_overlay: headline with auto-wrap and auto-sizing
- add_subtext: secondary text with auto-wrap
- place_logo: brand logo (hardcoded top-right)
- apply_brand_overlay: semi-transparent color overlay

Validation:
- regions_overlap / resolve_overlap: prevent overlapping assets
- validate_crop_ratio: warn on extreme crops
"""

import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from config import BILLBOARD_WIDTH, BILLBOARD_HEIGHT, BRAND_WHITE, BRAND_PRIMARY


def _next_path(output_dir: str, edit_history: list[str], label: str) -> str:
    step = len(edit_history) + 1
    return os.path.join(output_dir, f"edit_{step:03d}_{label}.png")


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

MIN_ASSET_DIMENSION = 50
MIN_GAP = 10


def regions_overlap(r1: dict, r2: dict) -> bool:
    """Check if two rectangular regions overlap."""
    if r1["x"] + r1["width"] <= r2["x"] or r2["x"] + r2["width"] <= r1["x"]:
        return False
    if r1["y"] + r1["height"] <= r2["y"] or r2["y"] + r2["height"] <= r1["y"]:
        return False
    return True


def resolve_overlap(new_region: dict, placed_regions: list[dict]) -> dict:
    """Shift new_region to avoid overlapping placed regions. Tries right, then down, then shrink."""
    if not placed_regions:
        return dict(new_region)

    # Strategy 1: Shift right
    h = dict(new_region)
    for _ in range(20):
        if not any(regions_overlap(h, p) for p in placed_regions):
            break
        for p in placed_regions:
            if regions_overlap(h, p):
                h["x"] = p["x"] + p["width"] + MIN_GAP
                break
    if h["x"] + h["width"] <= BILLBOARD_WIDTH:
        return h

    # Strategy 2: Shift down
    v = dict(new_region)
    for _ in range(20):
        if not any(regions_overlap(v, p) for p in placed_regions):
            break
        for p in placed_regions:
            if regions_overlap(v, p):
                v["y"] = p["y"] + p["height"] + MIN_GAP
                break
    if v["y"] + v["height"] <= BILLBOARD_HEIGHT:
        return v

    # Strategy 3: Shrink to fit
    remaining_w = BILLBOARD_WIDTH - h["x"]
    if remaining_w >= MIN_ASSET_DIMENSION:
        h["width"] = remaining_w
        return h

    return dict(new_region)


def validate_crop_ratio(source_width: int, source_height: int, target_width: int, target_height: int) -> dict:
    """Check if cropping would discard too much of the image (>60% loss)."""
    target_ratio = target_width / target_height
    source_ratio = source_width / source_height

    if source_ratio > target_ratio:
        kept = int(source_height * target_ratio)
        loss_pct = (1 - kept / source_width) * 100
    else:
        kept = int(source_width / target_ratio)
        loss_pct = (1 - kept / source_height) * 100

    safe = loss_pct < 60
    warning = None if safe else f"Crop discards {loss_pct:.0f}% of the image."
    return {"safe": safe, "crop_loss_pct": loss_pct, "warning": warning}


# ---------------------------------------------------------------------------
# COMPOSITION TOOLS
# ---------------------------------------------------------------------------

def create_canvas(output_dir: str, edit_history: list[str], bg_color: tuple = (0, 61, 165)) -> dict:
    """Create a blank billboard canvas."""
    canvas = Image.new("RGB", (BILLBOARD_WIDTH, BILLBOARD_HEIGHT), bg_color)
    save_path = _next_path(output_dir, edit_history, "canvas")
    canvas.save(save_path)
    return {
        "new_image_path": save_path,
        "edit_description": f"Created {BILLBOARD_WIDTH}x{BILLBOARD_HEIGHT} canvas",
    }


def place_asset_on_canvas(
    canvas_path: str, asset_path: str, output_dir: str, edit_history: list[str],
    x: int, y: int, width: int, height: int,
    crop_focus: str = "center", resize_mode: str = "crop",
) -> dict:
    """
    Place an image asset onto the canvas.
    resize_mode: "crop" (preserves aspect ratio, best for people) or "stretch" (fills region, best for backgrounds).
    Backgrounds are auto-blurred to not compete with the hero image.
    """
    canvas = Image.open(canvas_path).convert("RGBA")
    asset = Image.open(asset_path).convert("RGBA")

    # Clamp to canvas bounds
    canvas_w, canvas_h = canvas.size
    x = max(0, min(x, canvas_w - 1))
    y = max(0, min(y, canvas_h - 1))
    width = min(width, canvas_w - x)
    height = min(height, canvas_h - y)

    if width < MIN_ASSET_DIMENSION or height < MIN_ASSET_DIMENSION:
        return {"new_image_path": canvas_path, "edit_description": f"Asset skipped — too small ({width}x{height})"}

    aw, ah = asset.size
    crop_check = validate_crop_ratio(aw, ah, width, height)
    crop_warning = f" WARNING: {crop_check['warning']}" if not crop_check["safe"] else ""

    if resize_mode == "stretch":
        asset = asset.resize((width, height), Image.LANCZOS)
        asset = asset.filter(ImageFilter.GaussianBlur(radius=3))
    else:
        # Crop to match aspect ratio, then resize
        target_ratio = width / height
        current_ratio = aw / ah

        if current_ratio > target_ratio:
            new_w = int(ah * target_ratio)
            left = {"left": 0, "right": aw - new_w}.get(crop_focus, (aw - new_w) // 2)
            asset = asset.crop((left, 0, left + new_w, ah))
        elif current_ratio < target_ratio:
            new_h = int(aw / target_ratio)
            top = {"top": 0, "bottom": ah - new_h}.get(crop_focus, (ah - new_h) // 2)
            asset = asset.crop((0, top, aw, top + new_h))

        asset = asset.resize((width, height), Image.LANCZOS)

        # Feather the edges so lifestyle blends into the background
        feather = 20
        mask = Image.new("L", (width, height), 255)
        draw_mask = ImageDraw.Draw(mask)
        for i in range(feather):
            alpha = int(255 * i / feather)
            draw_mask.rectangle([i, i, width - 1 - i, height - 1 - i], outline=alpha)
        asset.putalpha(mask)

    canvas.paste(asset, (x, y), asset)

    save_path = _next_path(output_dir, edit_history, "composed")
    canvas.convert("RGB").save(save_path)

    return {
        "new_image_path": save_path,
        "edit_description": (
            f"Placed asset at ({x},{y}) size {width}x{height} "
            f"(mode: {resize_mode}, crop_loss: {crop_check['crop_loss_pct']:.0f}%){crop_warning}"
        ),
    }


# ---------------------------------------------------------------------------
# TEXT & LOGO TOOLS
# ---------------------------------------------------------------------------

def add_text_overlay(
    image_path: str, output_dir: str, edit_history: list[str],
    headline: str, x: int, y: int, font_size: int = 48, text_color: str = BRAND_WHITE,
) -> dict:
    """Add headline text with auto-wrap and auto font-size reduction."""
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)

    img_w, img_h = img.size
    x = max(0, min(x, img_w - 10))
    y = max(0, min(y, img_h - 10))

    max_height = img_h - y - 20
    min_font_size = 28
    lines = [headline]
    line_height = int(font_size * 1.25)
    font = ImageFont.load_default()

    while font_size >= min_font_size:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()
            max_width = img_w - x - 20
            lines = _wrap_text(draw, headline, font, max_width)
            line_height = int(font_size * 1.25)
            break

        max_width = img_w - x - 20
        lines = _wrap_text(draw, headline, font, max_width)
        line_height = int(font_size * 1.25)

        if len(lines) * line_height <= max_height and len(lines) <= 3:
            break
        font_size -= 4

    shadow = 2
    for i, line in enumerate(lines):
        ly = y + i * line_height
        draw.text((x + shadow, ly + shadow), line, fill="#00000088", font=font)
        draw.text((x, ly), line, fill=text_color, font=font)

    text_bottom_y = y + len(lines) * line_height
    save_path = _next_path(output_dir, edit_history, "text")
    img.convert("RGB").save(save_path)

    return {
        "new_image_path": save_path,
        "text_bottom_y": text_bottom_y,
        "edit_description": f"Added headline '{headline}' at ({x},{y}) size {font_size}px, {len(lines)} lines",
    }


def add_subtext(
    image_path: str, output_dir: str, edit_history: list[str],
    text: str, x: int, y: int, font_size: int = 24, text_color: str = BRAND_WHITE,
) -> dict:
    """Add secondary text with auto-wrap."""
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)

    img_w, img_h = img.size
    x = max(0, min(x, img_w - 10))
    y = max(0, min(y, img_h - 10))

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()

    max_width = img_w - x - 20
    lines = _wrap_text(draw, text, font, max_width)
    line_height = int(font_size * 1.3)

    for i, line in enumerate(lines):
        ly = y + i * line_height
        if ly + font_size > img_h:
            break
        draw.text((x + 1, ly + 1), line, fill="#00000066", font=font)
        draw.text((x, ly), line, fill=text_color, font=font)

    save_path = _next_path(output_dir, edit_history, "subtext")
    img.convert("RGB").save(save_path)

    return {
        "new_image_path": save_path,
        "edit_description": f"Added subtext '{text}' at ({x},{y}) size {font_size}px",
    }


def place_logo(image_path: str, output_dir: str, edit_history: list[str], **kwargs) -> dict:
    """Place brand logo. Always top-right, fixed size."""
    img = Image.open(image_path).convert("RGBA")

    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "brand_logo.png")

    if os.path.exists(logo_path):
        logo = Image.open(logo_path).convert("RGBA")
        logo_h = int(BILLBOARD_HEIGHT * 0.28)
        logo_w = int(logo.width * (logo_h / logo.height))
        logo = logo.resize((logo_w, logo_h), Image.LANCZOS)
    else:
        # Generate a simple placeholder logo if no asset file exists
        logo_h = int(BILLBOARD_HEIGHT * 0.28)
        logo_w = int(logo_h * 1.8)
        logo = Image.new("RGBA", (logo_w, logo_h), (0, 0, 0, 0))
        draw_logo = ImageDraw.Draw(logo)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", logo_h // 3)
        except (IOError, OSError):
            font = ImageFont.load_default()
        from config import BRAND_NAME
        bbox = draw_logo.textbbox((0, 0), BRAND_NAME, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw_logo.text(((logo_w - tw) // 2, (logo_h - th) // 2), BRAND_NAME, fill=(0, 61, 165, 255), font=font)

    margin_x = int(BILLBOARD_WIDTH * 0.02)
    margin_y = int(BILLBOARD_HEIGHT * 0.06)
    x = BILLBOARD_WIDTH - logo_w - margin_x
    y = margin_y

    img.paste(logo, (x, y), logo)

    save_path = _next_path(output_dir, edit_history, "logo")
    img.convert("RGB").save(save_path)

    return {
        "new_image_path": save_path,
        "edit_description": f"Placed brand logo at top-right ({x},{y}), size {logo_w}x{logo_h}",
    }


def apply_brand_overlay(
    image_path: str, output_dir: str, edit_history: list[str],
    region: str = "bottom-strip", opacity: float = 0.3,
) -> dict:
    """Apply a semi-transparent brand color overlay."""
    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    r, g, b = 0, 61, 165
    a = int(255 * opacity)
    w, h = img.size

    regions = {
        "bottom-strip": [(0, int(h * 0.65)), (w, h)],
        "left-third": [(0, 0), (int(w * 0.35), h)],
        "right-third": [(int(w * 0.65), 0), (w, h)],
        "full": [(0, 0), (w, h)],
    }
    draw.rectangle(regions.get(region, regions["bottom-strip"]), fill=(r, g, b, a))

    result = Image.alpha_composite(img, overlay)
    save_path = _next_path(output_dir, edit_history, "overlay")
    result.convert("RGB").save(save_path)

    return {
        "new_image_path": save_path,
        "edit_description": f"Applied brand overlay on {region} (opacity {opacity})",
    }


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _wrap_text(draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    return lines