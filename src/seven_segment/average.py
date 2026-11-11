"""Per-display average image over all sessions.

Averaging every capture of a display blurs out noise and the changing digits,
leaving the bezel plus a ghost of every segment that was ever lit, which shows
where the segments are.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from seven_segment.sessions import SessionLoader, display_file_name


def mean_per_display(loader: SessionLoader) -> list[np.ndarray]:
    """Return the float32 mean image of each display across all sessions."""
    # Running sum per display, so only one session is in memory at a time.
    # sums: 10 x [H_i x W_i] float64.
    sums: list[np.ndarray] | None = None
    for displays in loader:  # displays: 10 x [H_i x W_i] uint8
        if sums is None:
            sums = [d.astype(np.float64) for d in displays]
        else:
            for acc, d in zip(sums, displays, strict=True):
                acc += d  # [H x W] += [H x W]
    if sums is None:
        raise ValueError(f"no sessions found in {loader.root}")
    # [H x W] / S (Scalar) -> [H x W] float32, per display.
    return [(s / len(loader)).astype(np.float32) for s in sums]


def stretch(img: np.ndarray) -> np.ndarray:
    """Linearly rescale ``img`` to the full 0-255 ``uint8`` range."""
    lo, hi = float(img.min()), float(img.max())  # [H x W] -> Scalar, Scalar
    if hi == lo:
        return np.zeros(img.shape, np.uint8)
    # [H x W] float -> [H x W] uint8, mapping lo -> 0 and hi -> 255.
    return ((img - lo) * (255.0 / (hi - lo))).round().astype(np.uint8)


def montage(images: list[np.ndarray], gap: int = 4, scale: int = 3) -> Image.Image:
    """Lay images side by side on a white background, upscaled for viewing."""
    # L x [H_i x W_i] -> canvas [max(H_i) x (sum(W_i) + gaps)] -> upscaled.
    h = max(i.shape[0] for i in images)
    w = sum(i.shape[1] for i in images) + gap * (len(images) - 1)
    canvas = Image.new("L", (w, h), 255)
    x = 0
    for img in images:
        canvas.paste(Image.fromarray(img), (x, 0))
        x += img.shape[1] + gap
    return canvas.resize((w * scale, h * scale), Image.Resampling.NEAREST)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", type=Path, help="root with session_NNN dirs")
    parser.add_argument("out", type=Path, help="output directory")
    args = parser.parse_args(argv)

    means = [stretch(m) for m in mean_per_display(SessionLoader(args.sessions))]
    args.out.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(means):
        Image.fromarray(img).save(args.out / display_file_name(i))
    montage(means).save(args.out / "montage.png")
    print(f"wrote {len(means)} average images + montage.png to {args.out}")


if __name__ == "__main__":
    main()
