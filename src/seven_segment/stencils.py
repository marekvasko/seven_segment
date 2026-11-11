"""Geometric per-segment stencils for 7-segment displays.

A display is described by a :class:`DisplayStencil`: ``n_digits`` equally
spaced, equally sized digit cells sharing one slant and stroke width, plus
optional decimal points. From it we rasterise a soft mask (weights in 0..1)
for every segment of every digit, whether or not that segment ever lit in the
recorded data.

Segment naming follows the usual convention::

     aaa
    f   b
     ggg
    e   c
     ddd   (dp)

Masks deliberately leave out the corners where segments meet, so each pixel
belongs to at most one segment.

Array shape notation used in comments (``[A x B] -> [C]``):
    S  sessions (frames) of one display      H, W  image height, width (px)
    D  digits of a display                   N     segment masks of a display
                                                   (7 * D + number of dps)
    K  supersampling factor (``SUPERSAMPLE``)  L   number of values
    Scalar  a single number
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SEGMENTS = ("a", "b", "c", "d", "e", "f", "g")
SUPERSAMPLE = 4

Polygon = list[tuple[float, float]]


@dataclass
class DisplayStencil:
    """Geometry of one display, in pixel coordinates of its crop.

    ``x0, y0`` is the top-left corner of the first digit's *unslanted* cell,
    ``w, h`` the cell size, ``pitch`` the distance between digit origins,
    ``slant`` the horizontal shift per pixel of height (positive leans right),
    ``stroke`` the segment thickness. ``dp`` maps a digit index to the centre
    of the decimal point that follows that digit. ``dx`` holds optional small
    per-digit horizontal corrections for unevenly spaced digits. ``threshold``
    is the calibrated lit/unlit decision level (see ``segments.py``).
    """

    width: int
    height: int
    n_digits: int
    x0: float
    y0: float
    w: float
    h: float
    pitch: float
    slant: float
    stroke: float
    dp: dict[int, tuple[float, float]] = field(default_factory=dict)
    dx: list[float] = field(default_factory=list)
    threshold: float | None = None

    # ---- serialisation -------------------------------------------------
    def to_json(self) -> str:
        d = asdict(self)
        d["dp"] = {str(k): list(v) for k, v in self.dp.items()}
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text: str) -> DisplayStencil:
        d = json.loads(text)
        d["dp"] = {int(k): tuple(v) for k, v in d.get("dp", {}).items()}
        return cls(**d)

    def save(self, path: Path) -> None:
        path.write_text(self.to_json() + "\n")

    @classmethod
    def load(cls, path: Path) -> DisplayStencil:
        return cls.from_json(path.read_text())

    # ---- geometry ------------------------------------------------------
    def _to_image(self, digit: int, x: float, y: float) -> tuple[float, float]:
        """Map digit-local (x, y) to image coordinates, applying slant."""
        return (
            self.x0
            + digit * self.pitch
            + (self.dx[digit] if digit < len(self.dx) else 0.0)
            + x
            + self.slant * (self.h - y),
            self.y0 + y,
        )

    def _rect(self, digit: int, x1: float, y1: float, x2: float, y2: float) -> Polygon:
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return [self._to_image(digit, x, y) for x, y in corners]

    def segment_polygons(self, digit: int) -> dict[str, Polygon]:
        """Polygons of segments a-g of one digit, in image coordinates."""
        w, h, s = self.w, self.h, self.stroke
        m = 0.3 * s  # gap left at segment ends, keeps masks out of corners
        h2 = h / 2
        return {
            "a": self._rect(digit, s + m, 0, w - s - m, s),
            "b": self._rect(digit, w - s, s + m, w, h2 - s / 2 - m),
            "c": self._rect(digit, w - s, h2 + s / 2 + m, w, h - s - m),
            "d": self._rect(digit, s + m, h - s, w - s - m, h),
            "e": self._rect(digit, 0, h2 + s / 2 + m, s, h - s - m),
            "f": self._rect(digit, 0, s + m, s, h2 - s / 2 - m),
            "g": self._rect(digit, s + m, h2 - s / 2, w - s - m, h2 + s / 2),
        }

    def hole_polygons(self, digit: int) -> list[Polygon]:
        """The two unlit 'holes' inside a digit, used as local background."""
        w, h, s = self.w, self.h, self.stroke
        h2 = h / 2
        p = s  # extra margin so blur from lit segments doesn't leak in
        return [
            self._rect(digit, s + p, s + p, w - s - p, h2 - s / 2 - p),
            self._rect(digit, s + p, h2 + s / 2 + p, w - s - p, h - s - p),
        ]

    def dp_polygon(self, digit: int) -> Polygon:
        cx, cy = self.dp[digit]
        # Square of side 1.4 strokes, but at least 2 px: on small displays a
        # stroke-based box would be a speck dominated by blur and noise.
        r = max(0.7 * self.stroke, 1.0)
        return [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]

    # ---- rasterisation ---------------------------------------------------
    def rasterize(self, polygon: Polygon) -> np.ndarray:
        """Soft mask (float32, 0..1) of ``polygon`` at the crop resolution."""
        k = SUPERSAMPLE
        # Draw the polygon K times larger than the crop: [H*K x W*K], 0 or 255.
        img = Image.new("L", (self.width * k, self.height * k), 0)
        ImageDraw.Draw(img).polygon([(x * k, y * k) for x, y in polygon], fill=255)
        # [H*K x W*K] -> [H x K x W x K]: group each K x K block of sub-pixels
        # with the output pixel it belongs to.
        a = np.asarray(img, np.float32).reshape(self.height, k, self.width, k)
        # [H x K x W x K] -> [H x W]: the fraction of each pixel the polygon
        # covers, in 0..1 (anti-aliased mask).
        return a.mean(axis=(1, 3)) / 255.0

    def segment_masks(self) -> dict[str, np.ndarray]:
        """All segment masks keyed ``"<digit>.<segment>"``, e.g. ``"0.a"``, ``"1.dp"``.

        Digits are numbered left to right from 0.
        """
        # N entries, each [H x W] float32 in 0..1.
        masks = {}
        for d in range(self.n_digits):
            for name, poly in self.segment_polygons(d).items():
                masks[f"{d}.{name}"] = self.rasterize(poly)
            if d in self.dp:
                masks[f"{d}.dp"] = self.rasterize(self.dp_polygon(d))
        return masks

    def hole_masks(self) -> list[np.ndarray]:
        """Background mask per digit (both holes combined)."""
        # Per digit: 2 x [H x W] -> sum -> [H x W]. Result: D x [H x W].
        return [
            np.sum([self.rasterize(p) for p in self.hole_polygons(d)], axis=0)
            for d in range(self.n_digits)
        ]


def overlay(
    gray: np.ndarray,
    stencil: DisplayStencil,
    scale: int = 6,
    lit: dict[str, bool] | None = None,
) -> Image.Image:
    """Draw the stencil on ``gray`` (upscaled by ``scale``).

    Without ``lit`` (geometry check): segment outlines green, dp red, holes
    blue. With ``lit`` (a reading, e.g. from ``SegmentReader.lit``): lit
    segments filled green, unlit ones outlined red, holes not drawn.
    """
    # [H x W] uint8 -> RGB image [H x W x 3] -> upscaled [H*scale x W*scale x 3]
    img = Image.fromarray(gray).convert("RGBA")
    img = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    def scaled(poly: Polygon) -> list[tuple[float, float]]:
        return [(x * scale, y * scale) for x, y in poly]

    for d in range(stencil.n_digits):
        polys = dict(stencil.segment_polygons(d))
        if d in stencil.dp:
            polys["dp"] = stencil.dp_polygon(d)
        for name, poly in polys.items():
            if lit is None:
                colour = (255, 0, 0, 255) if name == "dp" else (0, 255, 0, 255)
                draw.polygon(scaled(poly), outline=colour)
            elif lit[f"{d}.{name}"]:
                draw.polygon(
                    scaled(poly), fill=(40, 220, 90, 150), outline=(40, 220, 90, 255)
                )
            else:
                draw.polygon(scaled(poly), outline=(255, 70, 70, 230))
        if lit is None:
            for poly in stencil.hole_polygons(d):
                draw.polygon(scaled(poly), outline=(60, 120, 255, 255))
    return Image.alpha_composite(img, layer).convert("RGB")
