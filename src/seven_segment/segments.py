"""Decide which segments are lit, and decode them into digits.

Images are normalised per frame (each spans 0..255), so absolute brightness
is not comparable between frames. The feature for a segment is therefore its
mean brightness *minus* the mean of that digit's unlit holes, so unlit
segments sit near 0. A per-display threshold is calibrated on all sessions
pooled: Otsu splits the features into an unlit and a lit cluster, and the
threshold is placed at ``LIT_FRACTION`` of the way from the unlit median to
the lit median. (Otsu's own split lands too high because lit brightness
varies a lot between segments.) Segments that never change in the data are
still classified, as long as other segments of the display do change.

Array shape notation used in comments (``[A x B] -> [C]``):
    S  sessions (frames) of one display      H, W  image height, width (px)
    D  digits of a display                   N     segment masks of a display
                                                   (7 * D + number of dps)
    K  supersampling factor (``SUPERSAMPLE``)  L   number of values
    Scalar  a single number
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from seven_segment.sessions import SessionLoader
from seven_segment.stencils import SEGMENTS, DisplayStencil

# Lit-segment sets for each digit; some displays draw 6, 7, 9 in two styles.
DIGIT_PATTERNS: dict[frozenset[str], str] = {
    frozenset(p): ch
    for ch, pats in {
        " ": [""],
        "0": ["abcdef"],
        "1": ["bc"],
        "2": ["abdeg"],
        "3": ["abcdg"],
        "4": ["bcfg"],
        "5": ["acdfg"],
        "6": ["acdefg", "cdefg"],
        "7": ["abc", "abcf"],
        "8": ["abcdefg"],
        "9": ["abcdfg", "abcfg"],
        "-": ["g"],
    }.items()
    for p in pats
}
INVALID = "?"
LIT_FRACTION = 0.35


class SegmentReader:
    """Precomputed masks for one display; turns images into segment features."""

    def __init__(self, stencil: DisplayStencil) -> None:
        self.stencil = stencil
        masks = stencil.segment_masks()  # N x [H x W]
        self.names = list(masks)  # N names, e.g. "0.a"; fixes the feature order
        # N x [H x W] -> [N x H x W]
        self._masks = np.stack([masks[n] for n in self.names])
        # Normalise each mask to sum 1, so that a dot product with an image
        # gives the mean brightness under the mask:
        # [N x H x W] / [N x 1 x 1] -> [N x H x W]
        self._masks /= self._masks.sum(axis=(1, 2), keepdims=True)
        holes = stencil.hole_masks()  # D x [H x W]
        # D x [H x W], each normalised to sum 1 -> [D x H x W]
        self._holes = np.stack([h / h.sum() for h in holes])
        # [N] int: index of the digit each segment belongs to, used to pick
        # the right hole background for every segment.
        self._digit_of = np.array([int(n.split(".")[0]) for n in self.names])

    def features(self, img: np.ndarray) -> np.ndarray:
        """Segment brightness minus its digit's hole brightness, one per name."""
        img = img.astype(np.float32)  # [H x W] uint8 -> [H x W] float32
        # Contract over both pixel axes, i.e. a weighted mean per mask:
        # [N x H x W] . [H x W] -> [N]
        seg = np.tensordot(self._masks, img, axes=2)
        # [D x H x W] . [H x W] -> [D]
        bg = np.tensordot(self._holes, img, axes=2)
        # [D] indexed by [N] -> [N]: each segment's own digit background.
        # [N] - [N] -> [N]
        return seg - bg[self._digit_of]

    def lit(self, img: np.ndarray, threshold: float | None = None) -> dict[str, bool]:
        """Which segments are lit; uses the stencil's calibrated threshold."""
        t = self.stencil.threshold if threshold is None else threshold
        if t is None:
            raise ValueError("stencil has no calibrated threshold")
        # [H x W] -> features [N] -> compared with Scalar -> [N] bool -> dict
        return dict(zip(self.names, self.features(img) > t, strict=True))

    def read(self, img: np.ndarray) -> str:
        """Decode an image of this display straight to text."""
        return self.decode(self.lit(img))

    def decode(self, lit: dict[str, bool]) -> str:
        """Render lit segments as text, e.g. ``"15.4"``; unknown patterns -> ``?``."""
        out = []
        for d in range(self.stencil.n_digits):
            on = frozenset(s for s in SEGMENTS if lit[f"{d}.{s}"])
            out.append(DIGIT_PATTERNS.get(on, INVALID))
            if lit.get(f"{d}.dp"):
                out.append(".")
        return "".join(out)


def otsu_threshold(values: np.ndarray) -> float:
    """Threshold maximising between-class variance of 1-D ``values``."""
    # Any shape (e.g. [S x N]) -> flattened and sorted [L].
    v = np.sort(values.ravel())
    best_t, best = float(v[0]), -1.0
    # Try every split point i: [L] -> lower [i] and upper [L - i].
    for i in range(1, len(v)):
        if v[i] == v[i - 1]:
            continue
        lo, hi = v[:i], v[i:]
        # [i], [L - i] -> Scalar (proportional to between-class variance)
        between = len(lo) * len(hi) * (lo.mean() - hi.mean()) ** 2
        if between > best:
            best, best_t = between, float((v[i - 1] + v[i]) / 2)
    return best_t


def calibrate_threshold(features: np.ndarray, fraction: float = LIT_FRACTION) -> float:
    """Lit/unlit threshold from pooled segment features of many frames."""
    split = otsu_threshold(features)  # [S x N] -> Scalar
    # Boolean indexing flattens: [S x N] -> [L_off] -> median -> Scalar
    off = float(np.median(features[features <= split]))
    # [S x N] -> [L_on] -> median -> Scalar
    on = float(np.median(features[features > split]))
    return off + fraction * (on - off)  # Scalar


def read_all(
    loader: SessionLoader, stencils: list[DisplayStencil]
) -> tuple[list[list[str]], list[float]]:
    """Decode every display of every session. Returns (readings, thresholds)."""
    readers = [SegmentReader(s) for s in stencils]
    sessions = list(loader)  # S x (10 x [H_i x W_i])
    # Per display i: S x [H_i x W_i] -> S x [N_i] -> stack -> [S x N_i]
    feats = [
        np.stack([r.features(s[i]) for s in sessions]) for i, r in enumerate(readers)
    ]
    # Per display: [S x N_i] -> Scalar
    thresholds = [calibrate_threshold(f) for f in feats]
    # Per session k and display: [N_i] row > Scalar -> [N_i] bool -> text
    readings = [
        [
            r.decode(dict(zip(r.names, f[k] > t, strict=True)))
            for r, f, t in zip(readers, feats, thresholds, strict=True)
        ]
        for k in range(len(sessions))
    ]
    return readings, thresholds


def load_labels(path: Path) -> dict[tuple[int, int], str]:
    """Hand-checked labels from the review app: (session, display) -> text."""
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {
            (int(row["session"]), int(row["display"])): row["label"]
            for row in csv.DictReader(f)
        }


def evaluate(
    readings: list[list[str]], labels: dict[tuple[int, int], str]
) -> dict[int, tuple[int, int]]:
    """Per display: (correct, checked) readings compared with the labels.

    ``readings[k][i]`` is session ``k + 1``, display ``i``. Leading blanks are
    ignored, as in the labels.
    """
    result: dict[int, tuple[int, int]] = {}
    for (session, display), label in labels.items():
        ok, n = result.get(display, (0, 0))
        hit = readings[session - 1][display].strip() == label
        result[display] = (ok + hit, n + 1)
    return dict(sorted(result.items()))


def save_readings_csv(path: Path, readings: list[list[str]]) -> None:
    """Write one row per session, one column per display.

    Leading blanks (unlit leading digits) are stripped, so ``"  0"`` -> ``"0"``.
    Values stay text: ``"?"`` marks an unknown segment pattern.
    """
    n_displays = len(readings[0]) if readings else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", *(f"display_{i:02d}" for i in range(n_displays))])
        for k, row in enumerate(readings, start=1):
            w.writerow([k, *(r.strip() for r in row)])


def save_segments_csv(
    path: Path,
    loader: SessionLoader,
    stencils: list[DisplayStencil],
    thresholds: list[float],
) -> None:
    """Write every segment decision, one row per (session, display, segment).

    Columns: session, display, segment (e.g. ``"1.dp"``), feature (segment
    minus hole brightness), threshold, lit (0/1).
    """
    readers = [SegmentReader(s) for s in stencils]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "display", "segment", "feature", "threshold", "lit"])
        for k, displays in enumerate(loader, start=1):  # 10 x [H_i x W_i]
            for i, (r, t) in enumerate(zip(readers, thresholds, strict=True)):
                feats = r.features(displays[i])  # [H_i x W_i] -> [N_i]
                for name, v in zip(r.names, feats, strict=True):
                    w.writerow([k, i, name, f"{v:.2f}", f"{t:.2f}", int(v > t)])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", type=Path)
    parser.add_argument("--stencils", type=Path, default=Path("stencils"))
    parser.add_argument(
        "--save", action="store_true", help="store thresholds in the stencil JSON"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("labels/readings.csv"),
        help="hand-checked labels to evaluate against (if the file exists)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/readings"),
        help="directory for readings.csv and segments.csv",
    )
    args = parser.parse_args(argv)

    paths = sorted(args.stencils.glob("display_*.json"))
    stencils = [DisplayStencil.load(p) for p in paths]
    loader = SessionLoader(args.sessions)
    readings, thresholds = read_all(loader, stencils)
    if args.save:
        for path, stencil, t in zip(paths, stencils, thresholds, strict=True):
            stencil.threshold = round(t, 2)
            stencil.save(path)
    print("thresholds:", " ".join(f"{t:.0f}" for t in thresholds))
    for k, row in enumerate(readings, start=1):
        print(f"session_{k:03d}  " + "  ".join(f"{r:>5}" for r in row))
    n_bad = sum(INVALID in r for row in readings for r in row)
    print(f"invalid readings: {n_bad} / {len(readings) * len(stencils)}")
    labels = load_labels(args.labels)
    if labels:
        scores = evaluate(readings, labels)
        per = "  ".join(f"{d:02d}:{ok}/{n}" for d, (ok, n) in scores.items())
        ok = sum(o for o, _ in scores.values())
        n = sum(c for _, c in scores.values())
        print(f"accuracy vs {args.labels}: {ok}/{n} = {100 * ok / n:.1f} %")
        print(f"  per display  {per}")
    save_readings_csv(args.out / "readings.csv", readings)
    save_segments_csv(args.out / "segments.csv", loader, stencils, thresholds)
    print(f"saved readings.csv and segments.csv to {args.out}")


if __name__ == "__main__":
    main()
