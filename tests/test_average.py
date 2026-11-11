from pathlib import Path

import numpy as np
from PIL import Image

from seven_segment.average import mean_per_display, stretch
from seven_segment.sessions import SessionLoader


def test_mean_per_display(tmp_path: Path) -> None:
    for s, value in enumerate([0, 100], start=1):
        d = tmp_path / f"session_{s:03d}"
        d.mkdir()
        Image.new("L", (4, 2), value).save(d / "display_00.png")
        Image.new("L", (3, 3), 2 * value).save(d / "display_01.png")

    means = mean_per_display(SessionLoader(tmp_path))

    assert [m.shape for m in means] == [(2, 4), (3, 3)]
    assert np.allclose(means[0], 50) and np.allclose(means[1], 100)


def test_stretch_full_range() -> None:
    out = stretch(np.array([[10.0, 20.0, 30.0]]))
    assert out.tolist() == [[0, 128, 255]]
    assert stretch(np.full((2, 2), 7.0)).max() == 0
