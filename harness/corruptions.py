"""Deterministic image corruptions for the A1 attribution sweep.

Severities are fixed and documented so the sweep is reproducible. These approximate the
real failure modes blind users hit: motion/defocus blur, under/over-exposure, and bad
framing (off-center crop). Keep them simple and inspectable — reviewers will rerun this.
"""
from __future__ import annotations
from PIL import Image, ImageEnhance, ImageFilter

# gaussian-blur radius (px) by severity 1..5
_BLUR_RADIUS = {1: 1.0, 2: 2.0, 3: 3.5, 4: 5.0, 5: 7.0}
# exposure multiplier by signed severity (-3..3); negative = underexpose
_EXPOSURE = {-3: 0.35, -2: 0.5, -1: 0.7, 0: 1.0, 1: 1.4, 2: 1.8, 3: 2.4}
# center-crop keep-fraction by severity 1..5 (then resized back to original size)
_CROP_KEEP = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6, 5: 0.5}


def apply_corruption(img: Image.Image, kind: str, severity: int) -> Image.Image:
    img = img.convert("RGB")
    if kind == "gaussian_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=_BLUR_RADIUS[severity]))
    if kind == "exposure":
        return ImageEnhance.Brightness(img).enhance(_EXPOSURE[severity])
    if kind == "center_crop":
        keep = _CROP_KEEP[abs(severity)]
        w, h = img.size
        cw, ch = int(w * keep), int(h * keep)
        left, top = (w - cw) // 2, (h - ch) // 2
        cropped = img.crop((left, top, left + cw, top + ch))
        return cropped.resize((w, h), Image.BILINEAR)
    raise ValueError(f"unknown corruption {kind!r}")


def condition_transform(cond: dict):
    """Return a callable image->image for a condition dict, or None for clean/blind."""
    if not cond or "corruption" not in cond:
        return None
    kind = cond["corruption"]
    sev = cond["severity"]
    return lambda im: apply_corruption(im, kind, sev)
