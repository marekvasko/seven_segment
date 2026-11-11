"""Group display crops into capture sessions and load them back.

The raw dataset is a flat, sequentially numbered list of crops. Each capture
session records every display once, in a fixed order, so consecutive blocks
of ``displays_per_session`` files form one session::

    0001.png .. 0010.png -> session_001  (display 0 .. 9)
    0011.png .. 0020.png -> session_002
    ...

On disk a session is ``<root>/session_NNN/display_DD.png``. Display ``DD`` is
always the same physical display across sessions (verified by image size).
"""

import argparse
import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image

DISPLAYS_PER_SESSION = 10
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def session_dir_name(index: int) -> str:
    """Directory name for the 1-based session ``index``."""
    return f"session_{index:03d}"


def display_file_name(index: int) -> str:
    """File name for the 0-based display ``index`` within a session."""
    return f"display_{index:02d}.png"


def group_into_sessions(
    src: Path, dst: Path, *, displays_per_session: int = DISPLAYS_PER_SESSION
) -> int:
    """Copy a flat, numbered image set from ``src`` into session directories.

    Raises ``ValueError`` if the image count is not a multiple of
    ``displays_per_session`` or if a display's image size differs between
    sessions (which would mean the assumed ordering is wrong).
    Returns the number of sessions written.
    """
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files or len(files) % displays_per_session:
        raise ValueError(
            f"{len(files)} images is not a multiple of {displays_per_session}"
        )

    sizes: list[tuple[int, int]] = []
    for i, path in enumerate(files):
        with Image.open(path) as img:
            size = img.size
        display = i % displays_per_session
        if i < displays_per_session:
            sizes.append(size)
        elif size != sizes[display]:
            raise ValueError(
                f"{path.name}: size {size} != {sizes[display]} for display {display}"
            )

    for i, path in enumerate(files):
        session, display = divmod(i, displays_per_session)
        out = dst / session_dir_name(session + 1)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out / display_file_name(display))
    return len(files) // displays_per_session


class SessionLoader:
    """Load sessions from ``<root>/session_NNN/display_DD.png``.

    Indexing is 0-based: ``loader[0]`` is ``session_001``. Each session is a
    list of 2-D ``uint8`` arrays, one per display, ordered by display index.
    Displays have different sizes, so they are not stacked into one array.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.session_dirs = sorted(
            p
            for p in self.root.iterdir()
            if p.is_dir() and p.name.startswith("session_")
        )

    def __len__(self) -> int:
        return len(self.session_dirs)

    def __getitem__(self, index: int) -> list[np.ndarray]:
        return self.load(self.session_dirs[index])

    def __iter__(self) -> Iterator[list[np.ndarray]]:
        for session_dir in self.session_dirs:
            yield self.load(session_dir)

    @staticmethod
    def load(session_dir: Path) -> list[np.ndarray]:
        """Load all displays of one session directory as grayscale arrays."""
        # PNG files -> 10 x [H_i x W_i] uint8 (each display has its own size,
        # so they can't be stacked into one array).
        return [
            np.asarray(Image.open(p).convert("L"))
            for p in sorted(session_dir.glob("display_*.png"))
        ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Group a flat image set into sessions."
    )
    parser.add_argument("src", type=Path, help="directory with the flat image set")
    parser.add_argument("dst", type=Path, help="output root directory")
    parser.add_argument(
        "-n", "--displays-per-session", type=int, default=DISPLAYS_PER_SESSION
    )
    args = parser.parse_args(argv)
    n = group_into_sessions(
        args.src, args.dst, displays_per_session=args.displays_per_session
    )
    print(f"wrote {n} sessions to {args.dst}")


if __name__ == "__main__":
    main()
