from dataclasses import replace
from pathlib import Path

import numpy as np

from seven_segment.fit_stencils import fit_display
from seven_segment.segments import (
    DIGIT_PATTERNS,
    SegmentReader,
    calibrate_threshold,
    save_readings_csv,
)
from seven_segment.stencils import DisplayStencil

STENCIL = DisplayStencil(
    width=60, height=40, n_digits=2, x0=12, y0=6, w=16, h=28,
    pitch=22, slant=0.1, stroke=3, dp={0: (32.0, 33.0)},
)  # fmt: skip
PATTERN_OF = {ch: p for p, ch in DIGIT_PATTERNS.items()}


def render(stencil: DisplayStencil, text: str, dp: bool = True) -> np.ndarray:
    """Synthetic frame: dark panel, bright lit segments."""
    img = np.full((stencil.height, stencil.width), 60.0)
    for d, ch in enumerate(text):
        for seg in PATTERN_OF[ch]:
            img += 180 * stencil.rasterize(stencil.segment_polygons(d)[seg])
    if dp:
        img += 180 * stencil.rasterize(stencil.dp_polygon(0))
    return np.clip(img, 0, 255).astype(np.uint8)


def test_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    STENCIL.save(path)
    assert DisplayStencil.load(path) == STENCIL


def test_masks_are_disjoint_and_complete() -> None:
    masks = STENCIL.segment_masks()
    assert len(masks) == 2 * 7 + 1
    assert all(m.sum() > 0 for m in masks.values())
    assert np.sum(list(masks.values()), axis=0).max() <= 1.0 + 1e-6
    assert all(h.sum() > 0 for h in STENCIL.hole_masks())


def test_reads_every_digit() -> None:
    reader = SegmentReader(replace(STENCIL))
    frames = [render(STENCIL, f"{n:02d}"[-2:].replace("0", " ", 1)) for n in range(10)]
    frames += [render(STENCIL, f"{n}{n}") for n in range(10)]
    feats = np.stack([reader.features(f) for f in frames])
    reader.stencil.threshold = calibrate_threshold(feats)
    assert reader.read(render(STENCIL, "47")) == "4.7"
    assert reader.read(render(STENCIL, " 8", dp=False)) == " 8"


def test_calibrate_threshold_between_clusters() -> None:
    values = np.array([-5, 0, 2, 4, 60, 90, 100, 120], float)
    t = calibrate_threshold(values)
    assert 4 < t < 60


def test_fit_recovers_geometry() -> None:
    rng = np.random.default_rng(0)
    # The dp is lit in some frames only, as in the real data: the fit
    # locates it by its activity.
    frames = [render(STENCIL, f"{n}{(3 * n) % 10}", dp=n % 3 > 0) for n in range(10)]
    stack = np.stack(frames).astype(float) + rng.normal(0, 5, (10, 40, 60))
    fitted = fit_display(np.clip(stack, 0, 255).astype(np.uint8), 2, (0,))
    # What matters for reading: each fitted mask lies on the true segment.
    true, got = STENCIL.segment_masks(), fitted.segment_masks()
    for name, mask in got.items():
        on_target = (mask * (true[name] > 0)).sum() / mask.sum()
        assert on_target > 0.8, name


def test_save_readings_csv(tmp_path: Path) -> None:
    path = tmp_path / "out" / "readings.csv"
    save_readings_csv(path, [["  0", "1.4"], [" 12", "?.0"]])
    assert path.read_text().splitlines() == [
        "session,display_00,display_01",
        "1,0,1.4",
        "2,12,?.0",
    ]
