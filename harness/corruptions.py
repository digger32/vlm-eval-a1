"""Deterministic image corruptions for the A1 attribution sweep.

Severities are fixed and documented so the sweep is reproducible. These approximate the
real failure modes blind users hit: motion/defocus blur, under/over-exposure, and bad
framing (off-center crop). Keep them simple and inspectable — reviewers will rerun this.
"""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

# gaussian-blur radius (px) by severity 1..5
_BLUR_RADIUS = {1: 1.0, 2: 2.0, 3: 3.5, 4: 5.0, 5: 7.0}
# exposure multiplier by signed severity (-3..3); negative = underexpose
_EXPOSURE = {-3: 0.35, -2: 0.5, -1: 0.7, 0: 1.0, 1: 1.4, 2: 1.8, 3: 2.4}
# center-crop keep-fraction by severity 1..5 (then resized back to original size)
_CROP_KEEP = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6, 5: 0.5}

# --- camera-ready additions (reviewer request: broader capture-failure suite) --------
# motion-blur kernel length (px) by severity 1..5
_MOTION_LEN = {1: 5, 2: 9, 3: 15, 4: 21, 5: 31}
# additive sensor-noise sigma (0-255 scale) by severity
_NOISE_SIGMA = {1: 5.0, 2: 10.0, 3: 18.0, 4: 28.0, 5: 40.0}
# glare blob radius as a fraction of the short side, by severity
_GLARE_FRAC = {1: 0.15, 2: 0.25, 3: 0.35, 4: 0.45, 5: 0.55}
# occlusion (finger/hand over the lens) coverage fraction by severity
_OCCL_FRAC = {1: 0.10, 2: 0.20, 3: 0.30, 4: 0.40, 5: 0.50}
# low-resolution downscale factor by severity (image is restored to original size)
_LOWRES_FACTOR = {1: 2, 2: 3, 3: 4, 4: 6, 5: 8}
# off-centre (severe misframing) shift as a fraction of the image, by severity
_OFFCENTRE = {1: 0.10, 2: 0.20, 3: 0.30, 4: 0.40, 5: 0.50}


def _motion_blur(img: Image.Image, severity: int) -> Image.Image:
    # horizontal camera pan during exposure: average n horizontally shifted copies
    # (PIL's Kernel filter only accepts 3x3/5x5, so do it with numpy).
    n = _MOTION_LEN[severity]
    a = np.asarray(img, dtype=np.float32)
    acc = np.zeros_like(a)
    off = n // 2
    for k in range(n):
        acc += np.roll(a, k - off, axis=1)
    return Image.fromarray(np.clip(acc / n, 0, 255).astype(np.uint8))


def _sensor_noise(img: Image.Image, severity: int) -> Image.Image:
    a = np.asarray(img, dtype=np.float32)
    rng = np.random.default_rng(0)  # fixed -> the corruption is reproducible
    a = a + rng.normal(0.0, _NOISE_SIGMA[severity], a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _glare(img: Image.Image, severity: int) -> Image.Image:
    w, h = img.size
    r = int(min(w, h) * _GLARE_FRAC[severity])
    # radial white blob near the centre, alpha-composited (specular reflection / flash)
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w // 2, h // 2
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    alpha = np.clip(1.0 - d / max(r, 1), 0.0, 1.0)[..., None]
    a = np.asarray(img, dtype=np.float32)
    a = a * (1 - alpha) + 255.0 * alpha
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _occlusion(img: Image.Image, severity: int) -> Image.Image:
    # opaque blob intruding from the bottom edge: a finger or hand over the lens,
    # the single most common capture failure reported by blind photographers.
    w, h = img.size
    out = img.copy()
    bh = int(h * _OCCL_FRAC[severity])
    ImageDraw.Draw(out).ellipse([-int(w * 0.1), h - bh, int(w * 0.9), h + bh],
                                fill=(60, 45, 40))
    return out


def _low_resolution(img: Image.Image, severity: int) -> Image.Image:
    w, h = img.size
    f = _LOWRES_FACTOR[severity]
    small = img.resize((max(w // f, 1), max(h // f, 1)), Image.BILINEAR)
    return small.resize((w, h), Image.NEAREST)


def _off_centre_crop(img: Image.Image, severity: int) -> Image.Image:
    # severe misframing: the subject is pushed towards a corner, not centred.
    w, h = img.size
    shift = _OFFCENTRE[severity]
    keep = 1.0 - shift
    cw, ch = int(w * keep), int(h * keep)
    left, top = w - cw, h - ch          # crop the bottom-right region
    return img.crop((left, top, w, h)).resize((w, h), Image.BILINEAR)


def _gray(img: Image.Image, severity: int) -> Image.Image:
    # uniform mid-grey: an image that carries no scene information at all. Unlike the
    # blind control the model still receives an image token stream, so this separates
    # "no image" from "no information in the image".
    return Image.new("RGB", img.size, (128, 128, 128))


def _patch_shuffle(img: Image.Image, severity: int) -> Image.Image:
    # shuffle a grid of patches: local texture and colour statistics survive, global
    # scene structure does not.
    grid = {1: 2, 2: 4, 3: 8, 4: 12, 5: 16}[severity]
    w, h = img.size
    pw, ph = w // grid, h // grid
    if pw < 1 or ph < 1:
        return img
    patches = [img.crop((c * pw, r * ph, (c + 1) * pw, (r + 1) * ph))
               for r in range(grid) for c in range(grid)]
    rng = np.random.default_rng(0)          # fixed -> reproducible
    order = rng.permutation(len(patches))
    out = img.copy()
    for i, (r, c) in enumerate([(r, c) for r in range(grid) for c in range(grid)]):
        out.paste(patches[order[i]], (c * pw, r * ph))
    return out


_EXTRA = {
    "gray": _gray,
    "patch_shuffle": _patch_shuffle,
    "motion_blur": _motion_blur,
    "sensor_noise": _sensor_noise,
    "glare": _glare,
    "occlusion": _occlusion,
    "low_resolution": _low_resolution,
    "off_centre_crop": _off_centre_crop,
}


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
    if kind in _EXTRA:
        return _EXTRA[kind](img, abs(severity))
    raise ValueError(f"unknown corruption {kind!r}")


def condition_transform(cond: dict):
    """Return a callable image->image for a condition dict, or None for clean/blind."""
    if not cond or "corruption" not in cond:
        return None
    kind = cond["corruption"]
    sev = cond["severity"]
    return lambda im: apply_corruption(im, kind, sev)
