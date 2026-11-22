"""Pure-Python subset of the Hendrycks & Dietterich (2019) ImageNet-C / CIFAR-C corruptions."""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Callable

import numpy as np
from scipy import ndimage

try:
    import PIL  # noqa: F401
    from PIL import Image

    _HAS_PIL = True
except (
    ImportError
):  # pragma: no cover
    _HAS_PIL = False


_CORE_CORRUPTIONS: tuple[str, ...] = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "speckle_noise",
    "gaussian_blur",
    "defocus_blur",
    "contrast",
    "brightness",
    "saturate",
    "fog",
    "elastic_transform",
    "spatter_mud",
)

_PIL_CORRUPTIONS: tuple[str, ...] = ("jpeg_compression", "pixelate")

CORRUPTIONS: tuple[str, ...] = _CORE_CORRUPTIONS + (
    _PIL_CORRUPTIONS if _HAS_PIL else ()
)

OMITTED: dict[str, str] = {
    "glass_blur": "requires ImageMagick/wand; out of scope for the pure-Python suite.",
    "motion_blur": "requires ImageMagick/wand; out of scope for the pure-Python suite.",
    "zoom_blur": "requires ImageMagick/wand; out of scope for the pure-Python suite.",
    "snow": "requires ImageMagick/wand; out of scope for the pure-Python suite.",
    "frost": "requires ImageMagick/wand; out of scope for the pure-Python suite.",
}

NON_MONOTONIC: tuple[str, ...] = ("saturate", "fog", "elastic_transform", "spatter_mud")


_GAUSSIAN_NOISE_C = (0.08, 0.12, 0.18, 0.26, 0.38)
_SHOT_NOISE_C = (60.0, 25.0, 12.0, 5.0, 3.0)
_IMPULSE_NOISE_C = (0.03, 0.06, 0.09, 0.17, 0.27)
_SPECKLE_NOISE_C = (0.15, 0.2, 0.35, 0.45, 0.6)
_GAUSSIAN_BLUR_C = (1.0, 2.0, 3.0, 4.0, 6.0)
_DEFOCUS_BLUR_C = ((3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5))
_CONTRAST_C = (0.4, 0.3, 0.2, 0.1, 0.05)
_BRIGHTNESS_C = (0.1, 0.2, 0.3, 0.4, 0.5)
_SATURATE_C = ((0.3, 0.0), (0.1, 0.0), (2.0, 0.0), (5.0, 0.1), (20.0, 0.2))
_JPEG_QUALITY = (25, 18, 15, 10, 7)
_PIXELATE_C = (0.6, 0.5, 0.4, 0.3, 0.25)
_FOG_C = ((1.5, 2.0), (2.0, 2.0), (2.5, 1.7), (2.5, 1.5), (3.0, 1.4))
_ELASTIC_C = (
    (2.0, 0.7, 0.1),
    (2.0, 0.08, 0.2),
    (0.05, 0.01, 0.02),
    (0.07, 0.01, 0.02),
    (0.12, 0.01, 0.02),
)

_SPATTER_MUD_C = (
    (0.06, 4.0, 0.5),
    (0.10, 3.0, 0.6),
    (0.16, 2.0, 0.8),
    (0.24, 1.5, 1.0),
    (0.34, 1.0, 1.0),
)
_SPATTER_MUD_COLOR = (
    np.array([63.0, 42.0, 20.0]) / 255.0
)


def _rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    maxc = img.max(axis=-1)
    minc = img.min(axis=-1)
    v = maxc
    delta = maxc - minc
    safe_max = np.where(maxc == 0, 1.0, maxc)
    s = np.where(maxc == 0, 0.0, delta / safe_max)

    delta_safe = np.where(delta == 0, 1.0, delta)
    rc = (maxc - r) / delta_safe
    gc = (maxc - g) / delta_safe
    bc = (maxc - b) / delta_safe

    h = np.zeros_like(maxc)
    h = np.where(maxc == b, 4.0 + gc - rc, h)
    h = np.where(maxc == g, 2.0 + rc - bc, h)
    h = np.where(maxc == r, bc - gc, h)
    h = (h / 6.0) % 1.0
    h = np.where(delta == 0, 0.0, h)
    return np.stack([h, s, v], axis=-1)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (np.floor(h * 6.0).astype(np.int64)) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    conditions = [i == k for k in range(6)]
    r = np.select(conditions, [v, q, p, p, t, v], default=v)
    g = np.select(conditions, [t, v, v, q, p, p], default=v)
    b = np.select(conditions, [p, p, t, v, v, q], default=v)
    return np.stack([r, g, b], axis=-1)


def _gaussian_noise(
    img: np.ndarray, severity: int, rng: np.random.Generator
) -> np.ndarray:
    c = _GAUSSIAN_NOISE_C[severity - 1]
    return img + rng.normal(0.0, c, img.shape)


def _shot_noise(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    c = _SHOT_NOISE_C[severity - 1]
    return rng.poisson(np.clip(img, 0.0, 1.0) * c) / c


def _impulse_noise(
    img: np.ndarray, severity: int, rng: np.random.Generator
) -> np.ndarray:
    c = _IMPULSE_NOISE_C[severity - 1]
    flip = rng.random(img.shape) < c
    salt = rng.random(img.shape) < 0.5
    out = np.where(flip & salt, 1.0, img)
    out = np.where(flip & ~salt, 0.0, out)
    return out


def _speckle_noise(
    img: np.ndarray, severity: int, rng: np.random.Generator
) -> np.ndarray:
    c = _SPECKLE_NOISE_C[severity - 1]
    return img + img * rng.normal(0.0, c, img.shape)


def _gaussian_blur(
    img: np.ndarray, severity: int, rng: np.random.Generator
) -> np.ndarray:
    c = _GAUSSIAN_BLUR_C[severity - 1]
    out = np.empty_like(img)
    for ch in range(3):
        out[..., ch] = ndimage.gaussian_filter(img[..., ch], sigma=c, mode="reflect")
    return out


def _disk_kernel(radius: int, alias_blur: float) -> np.ndarray:
    span = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(span, span)
    kernel = ((xx**2 + yy**2) <= radius**2).astype(np.float64)
    kernel /= kernel.sum()
    if alias_blur > 0:
        kernel = ndimage.gaussian_filter(kernel, sigma=alias_blur)
        kernel /= kernel.sum()
    return kernel


def _defocus_blur(
    img: np.ndarray, severity: int, rng: np.random.Generator
) -> np.ndarray:
    radius, alias_blur = _DEFOCUS_BLUR_C[severity - 1]
    kernel = _disk_kernel(radius, alias_blur)
    out = np.empty_like(img)
    for ch in range(3):
        out[..., ch] = ndimage.convolve(img[..., ch], kernel, mode="reflect")
    return out


def _contrast(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    c = _CONTRAST_C[severity - 1]
    mean = img.mean(axis=(0, 1), keepdims=True)
    return (img - mean) * c + mean


def _brightness(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    c = _BRIGHTNESS_C[severity - 1]
    hsv = _rgb_to_hsv(np.clip(img, 0.0, 1.0))
    hsv = hsv.copy()
    hsv[..., 2] = np.clip(hsv[..., 2] + c, 0.0, 1.0)
    return _hsv_to_rgb(hsv)


def _saturate(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    mult, add = _SATURATE_C[severity - 1]
    hsv = _rgb_to_hsv(np.clip(img, 0.0, 1.0))
    hsv = hsv.copy()
    hsv[..., 1] = np.clip(hsv[..., 1] * mult + add, 0.0, 1.0)
    return _hsv_to_rgb(hsv)


def _jpeg_compression(
    img: np.ndarray, severity: int, rng: np.random.Generator
) -> np.ndarray:
    if not _HAS_PIL:
        raise ImportError(
            "jpeg_compression requires PIL (pillow), which is not importable in this "
            "environment; it is excluded from CORRUPTIONS when PIL is absent."
        )
    quality = _JPEG_QUALITY[severity - 1]
    arr8 = np.round(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr8, mode="RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    decoded = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float64) / 255.0
    return decoded


def _pixelate(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    if not _HAS_PIL:
        raise ImportError(
            "pixelate requires PIL (pillow), which is not importable in this environment; it "
            "is excluded from CORRUPTIONS when PIL is absent."
        )
    c = _PIXELATE_C[severity - 1]
    h, w = img.shape[:2]
    small_h, small_w = max(1, int(round(h * c))), max(1, int(round(w * c)))
    arr8 = np.round(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    im = Image.fromarray(arr8, mode="RGB")
    im = im.resize((small_w, small_h), Image.BOX)
    im = im.resize((w, h), Image.BOX)
    return np.asarray(im, dtype=np.float64) / 255.0


def _plasma_fractal(
    mapsize: int, wibbledecay: float, rng: np.random.Generator
) -> np.ndarray:
    assert mapsize & (mapsize - 1) == 0, "mapsize must be a power of two"
    maparray = np.zeros((mapsize, mapsize), dtype=np.float64)
    stepsize = mapsize
    wibble = 100.0

    def wibbled_mean(arr: np.ndarray) -> np.ndarray:
        return arr / 4.0 + wibble * rng.uniform(-wibble, wibble, arr.shape)

    while stepsize >= 2:
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = (
            cornerref
            + np.roll(cornerref, 1, axis=0)
            + np.roll(cornerref, 1, axis=1)
            + np.roll(cornerref, 1, axis=(0, 1))
        )
        half = stepsize // 2
        maparray[half:mapsize:stepsize, half:mapsize:stepsize] = wibbled_mean(
            squareaccum
        )

        drgrid = maparray[half:mapsize:stepsize, half:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]

        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        maparray[0:mapsize:stepsize, half:mapsize:stepsize] = wibbled_mean(
            ldrsum + lulsum
        )

        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        maparray[half:mapsize:stepsize, 0:mapsize:stepsize] = wibbled_mean(
            tdrsum + tulsum
        )

        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / (maparray.max() + 1e-12)


def _fog(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    intensity, wibbledecay = _FOG_C[severity - 1]
    h, w = img.shape[:2]
    mapsize = 1 << int(np.ceil(np.log2(max(h, w, 2))))
    fractal = _plasma_fractal(mapsize, wibbledecay, rng)[:h, :w]
    max_val = img.max()
    out = img + intensity * fractal[..., None]
    return out * max_val / (max_val + intensity + 1e-12)


def _affine_matrix_and_offset(
    affine_frac: float, h: int, w: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    jitter = rng.uniform(-affine_frac, affine_frac, (2, 2))
    matrix = np.eye(2) + jitter
    center = np.array([h / 2.0, w / 2.0])
    offset = center - matrix @ center
    return matrix, offset


def _elastic_transform(
    img: np.ndarray, severity: int, rng: np.random.Generator
) -> np.ndarray:
    alpha_frac, sigma_frac, affine_frac = _ELASTIC_C[severity - 1]
    h, w = img.shape[:2]
    size = min(h, w)
    alpha = alpha_frac * size
    sigma = max(sigma_frac * size, 1e-3)

    matrix, offset = _affine_matrix_and_offset(affine_frac, h, w, rng)
    dx = (
        ndimage.gaussian_filter(rng.uniform(-1.0, 1.0, (h, w)), sigma, mode="reflect")
        * alpha
    )
    dy = (
        ndimage.gaussian_filter(rng.uniform(-1.0, 1.0, (h, w)), sigma, mode="reflect")
        * alpha
    )

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = [yy + dy, xx + dx]

    out = np.empty_like(img)
    for ch in range(3):
        warped = ndimage.affine_transform(
            img[..., ch], matrix, offset=offset, mode="reflect", order=1
        )
        out[..., ch] = ndimage.map_coordinates(warped, coords, order=1, mode="reflect")
    return out


def _spatter_mud(
    img: np.ndarray, severity: int, rng: np.random.Generator
) -> np.ndarray:
    coverage, sigma, strength = _SPATTER_MUD_C[severity - 1]
    h, w = img.shape[:2]
    liquid = ndimage.gaussian_filter(
        rng.normal(size=(h, w)), sigma=sigma, mode="reflect"
    )
    thresh = np.quantile(liquid, 1.0 - coverage)
    mask = (liquid > thresh).astype(np.float64)
    mask = (
        ndimage.gaussian_filter(mask, sigma=max(sigma * 0.5, 0.5), mode="reflect")
        * strength
    )
    mask = np.clip(mask, 0.0, 1.0)[..., None]
    return img * (1.0 - mask) + _SPATTER_MUD_COLOR * mask


_CORRUPTION_FNS: dict[
    str, Callable[[np.ndarray, int, np.random.Generator], np.ndarray]
] = {
    "gaussian_noise": _gaussian_noise,
    "shot_noise": _shot_noise,
    "impulse_noise": _impulse_noise,
    "speckle_noise": _speckle_noise,
    "gaussian_blur": _gaussian_blur,
    "defocus_blur": _defocus_blur,
    "contrast": _contrast,
    "brightness": _brightness,
    "saturate": _saturate,
    "jpeg_compression": _jpeg_compression,
    "pixelate": _pixelate,
    "fog": _fog,
    "elastic_transform": _elastic_transform,
    "spatter_mud": _spatter_mud,
}


def apply_corruption(
    images: np.ndarray, corruption: str, severity: int, *, seed: int = 0
) -> np.ndarray:
    """Apply a named Hendrycks-Dietterich corruption to a batch of images."""
    if corruption not in CORRUPTIONS:
        if corruption in OMITTED:
            raise ValueError(
                f"corruption {corruption!r} is intentionally omitted: {OMITTED[corruption]}"
            )
        if corruption in _PIL_CORRUPTIONS and not _HAS_PIL:
            raise ImportError(
                f"corruption {corruption!r} requires PIL (pillow), which is not importable in "
                f"this environment."
            )
        raise ValueError(
            f"unknown corruption {corruption!r}; choose from {CORRUPTIONS}"
        )

    severity = int(severity)
    if not 1 <= severity <= 5:
        raise ValueError(
            f"severity must be in 1..5 (Hendrycks-Dietterich scale), got {severity}"
        )

    images = np.asarray(images)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"expected images of shape (N, H, W, 3), got {images.shape}")

    in_dtype = images.dtype
    is_uint8 = in_dtype == np.uint8
    if is_uint8:
        x = images.astype(np.float64) / 255.0
    elif np.issubdtype(in_dtype, np.floating):
        x = images.astype(np.float64)
    else:
        raise ValueError(f"images dtype must be uint8 or floating, got {in_dtype}")
    x = np.clip(x, 0.0, 1.0)

    fn = _CORRUPTION_FNS[corruption]
    rng = np.random.default_rng(seed)
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = np.clip(fn(x[i], severity, rng), 0.0, 1.0)

    if is_uint8:
        return np.round(out * 255.0).astype(np.uint8)
    return out.astype(in_dtype)


def apply_corruption_to_images(
    images: Sequence["Image.Image"], corruption: str, severity: int, *, seed: int = 0
) -> list:
    """Apply ``apply_corruption`` to a list of PIL images of possibly *varying* size."""
    if not _HAS_PIL:
        raise ImportError(
            "apply_corruption_to_images requires PIL (pillow), which is not importable."
        )

    seeds = np.random.SeedSequence(seed).spawn(len(images))
    out = []
    for im, sub_seed in zip(images, seeds):
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)[None, ...]
        sub_seed_int = int(sub_seed.generate_state(1)[0])
        corrupted = apply_corruption(arr, corruption, severity, seed=sub_seed_int)
        out.append(Image.fromarray(corrupted[0]))
    return out
