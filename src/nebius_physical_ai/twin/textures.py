"""Procedural floor textures for visual domain randomisation: wood grain, woven fabric and checker, each drawn from
a seed in a family of colours. A texture is named `<kind>:<seed>`, so a scene's look stays a small JSON value.

Training draws seeds below `HELD_OUT_SEED`; the benchmark's appearance suites draw from at and above it, so they
score unseen surfaces. None of these is any external benchmark's own texture (so101-benchmark's wood photo is never
used), which keeps that benchmark an honest test of generalisation.
"""

import colorsys
from functools import lru_cache

import cv2
import numpy as np

KINDS = ("wood", "fabric", "checker")
HELD_OUT_SEED = 1000
SIZE = 256


def smooth_noise(rng: np.random.Generator, size: int, cells: int) -> np.ndarray:
    """Values in [0, 1] varying smoothly over about `cells` blobs per side."""
    coarse = rng.random((cells + 1, cells + 1)).astype(np.float32)
    return cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC).clip(0, 1)


def hsv(h: float, s: float, v: float) -> np.ndarray:
    return np.array(colorsys.hsv_to_rgb(h % 1.0, s, v), np.float32)


def wood(rng: np.random.Generator, size: int) -> np.ndarray:
    """Grain lines along one axis, bent by low-frequency noise, between a dark and a light tone of one brown."""
    y = np.linspace(0, 1, size, dtype=np.float32)[:, None]
    warp = smooth_noise(rng, size, 3) * rng.uniform(0.05, 0.25) + smooth_noise(rng, size, 12) * 0.02
    grain = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(15, 45) * (y + warp))
    grain = grain ** rng.uniform(1.0, 3.0) * 0.8 + smooth_noise(rng, size, 40) * 0.2
    h, s = rng.uniform(0.04, 0.12), rng.uniform(0.3, 0.75)
    dark, light = hsv(h, s, rng.uniform(0.2, 0.45)), hsv(h + rng.uniform(-0.02, 0.02), s * 0.8, rng.uniform(0.55, 0.9))
    texture = dark + grain[..., None] * (light - dark)
    return np.rot90(texture) if rng.random() < 0.5 else texture


def fabric(rng: np.random.Generator, size: int) -> np.ndarray:
    """A fine weave plus mottling in any hue, muted enough to read as cloth or a desk mat."""
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    period = rng.uniform(3, 8)
    weave = 0.5 + 0.25 * (np.sin(2 * np.pi * x / period) + np.sin(2 * np.pi * y / period))
    shade = 0.75 + 0.25 * weave * 0.4 + 0.25 * smooth_noise(rng, size, 6)
    base = hsv(rng.random(), rng.uniform(0.05, 0.5), rng.uniform(0.3, 0.8))
    return base * shade[..., None]


def checker(rng: np.random.Generator, size: int) -> np.ndarray:
    squares = int(rng.integers(4, 16))
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    on = ((x * squares // size + y * squares // size) % 2)[..., None]
    h = rng.random()
    a, b = hsv(h, rng.uniform(0, 0.4), rng.uniform(0.3, 0.6)), hsv(h, rng.uniform(0, 0.4), rng.uniform(0.6, 0.9))
    return np.where(on, a, b)


@lru_cache(maxsize=64)
def texture(name: str, size: int = SIZE) -> bytes:
    """RGB bytes (size x size) of the texture named `<kind>:<seed>`, as MuJoCo's texture `data` takes them."""
    kind, seed = name.split(":")
    if kind not in KINDS:
        raise ValueError(f"unknown texture kind {kind!r}; expected one of {KINDS}")
    image = {"wood": wood, "fabric": fabric, "checker": checker}[kind](np.random.default_rng(int(seed)), size)
    return (np.clip(image, 0, 1) * 255).astype(np.uint8).tobytes()
