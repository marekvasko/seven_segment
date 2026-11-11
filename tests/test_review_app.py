import csv
from pathlib import Path

from PIL import Image

from seven_segment.review_app import Review
from seven_segment.sessions import display_file_name, session_dir_name
from tests.test_stencils import STENCIL, render


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    """Two sessions of one synthetic display, plus its stencil."""
    sessions, stencils = tmp_path / "sessions", tmp_path / "stencils"
    stencils.mkdir()
    STENCIL.save(stencils / "display_00.json")
    for k, text in enumerate(["12", " 7"], start=1):
        d = sessions / session_dir_name(k)
        d.mkdir(parents=True)
        Image.fromarray(render(STENCIL, text)).save(d / display_file_name(0))
    return sessions, stencils


def test_review_state_labels_and_images(tmp_path: Path) -> None:
    sessions, stencils = _setup(tmp_path)
    labels = tmp_path / "labels" / "readings.csv"
    review = Review(sessions, stencils, labels)

    items = review.state()["items"]
    assert [(it["session"], it["predicted"]) for it in items] == [(1, "1.2"), (2, ".7")]

    review.set_label(1, 0, "1.2")
    review.set_label(2, 0, "7.0")
    review.set_label(2, 0, "7.1")  # overwrite
    with labels.open() as f:
        rows = list(csv.DictReader(f))
    assert [(r["session"], r["label"], r["predicted"]) for r in rows] == [
        ("1", "1.2", "1.2"),
        ("2", "7.1", ".7"),
    ]

    # Labels persist across restarts, and clearing removes the row.
    review2 = Review(sessions, stencils, labels)
    assert review2.labels == {(1, 0): "1.2", (2, 0): "7.1"}
    review2.set_label(1, 0, None)
    assert "1.2" not in labels.read_text().splitlines()[1]

    for kind in ("raw", "overlay", "segments"):
        assert review.image_png(1, 0, kind, 3).startswith(b"\x89PNG")
