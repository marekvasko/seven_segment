from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from seven_segment.sessions import SessionLoader, group_into_sessions

SIZES = [(10, 5), (20, 8), (6, 6)]


def _make_flat(src: Path, sessions: int, sizes=SIZES) -> None:
    src.mkdir()
    n = 0
    for s in range(sessions):
        for w, h in sizes:
            n += 1
            Image.new("L", (w, h), color=s).save(src / f"{n:04d}.png")


def test_group_and_load_roundtrip(tmp_path: Path) -> None:
    _make_flat(tmp_path / "src", sessions=2)

    n = group_into_sessions(tmp_path / "src", tmp_path / "out", displays_per_session=3)
    loader = SessionLoader(tmp_path / "out")

    assert n == 2
    assert len(loader) == 2
    session = loader[1]
    assert [a.shape for a in session] == [(h, w) for w, h in SIZES]
    assert all(a.dtype == np.uint8 and (a == 1).all() for a in session)
    assert len(list(loader)) == 2


def test_rejects_incomplete_session(tmp_path: Path) -> None:
    _make_flat(tmp_path / "src", sessions=1)
    with pytest.raises(ValueError, match="multiple"):
        group_into_sessions(tmp_path / "src", tmp_path / "out", displays_per_session=2)


def test_rejects_inconsistent_display_sizes(tmp_path: Path) -> None:
    _make_flat(tmp_path / "src", sessions=2)
    Image.new("L", (99, 99)).save(tmp_path / "src" / "0004.png")
    with pytest.raises(ValueError, match="display 0"):
        group_into_sessions(tmp_path / "src", tmp_path / "out", displays_per_session=3)
