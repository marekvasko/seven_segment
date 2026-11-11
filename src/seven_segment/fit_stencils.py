"""Fit a :class:`DisplayStencil` to each display from session data.

The fit uses the per-pixel standard deviation over all sessions ("activity"):
segments that switch on and off are bright, static background is dark. The
template is moved and resized by coordinate descent to maximise activity under
the segment masks relative to the digit holes. Shared geometry across digits
lets fully lit digits pin down rarely lit ones.

Results are written to ``stencils/display_DD.json``; hand-edit those files if
an overlay shows a misfit and re-run with ``--overlay-only``.

Array shape notation used in comments (``[A x B] -> [C]``):
    S  sessions (frames) of one display      H, W  image height, width (px)
    D  digits of a display                   N     segment masks of a display
                                                   (7 * D + number of dps)
    K  supersampling factor (``SUPERSAMPLE``)  L   number of values
    Scalar  a single number
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from seven_segment.average import stretch
from seven_segment.sessions import SessionLoader
from seven_segment.stencils import DisplayStencil, overlay

# Digit count and decimal-point positions (dp after digit index), read off raw
# frames. E.g. display 01 shows "15.4": 3 digits, dp after digit 1.
LAYOUT: dict[int, tuple[int, tuple[int, ...]]] = {
    0: (3, ()),
    1: (3, (1,)),
    2: (2, ()),
    3: (1, ()),
    4: (3, ()),
    5: (3, ()),
    6: (3, (1,)),
    7: (2, ()),
    8: (1, ()),
    9: (3, ()),
}

PARAMS = ("x0", "y0", "w", "h", "pitch", "slant", "stroke")


def activity(stack: np.ndarray) -> np.ndarray:
    """Per-pixel standard deviation over sessions, as float32."""
    # [S x H x W] uint8 -> float32 -> std over sessions -> [H x W]
    return stack.astype(np.float32).std(axis=0)


def min_max_activity(stack: np.ndarray) -> np.ndarray:
    """Per-pixel min-max normalised activity over sessions, as float32."""
    _stack = stack.astype(np.float32)
    return _stack.max(axis=0) - _stack.min(axis=0)


def _weighted_mean(img: np.ndarray, mask: np.ndarray) -> float:
    total = mask.sum()  # [H x W] -> Scalar
    # [H x W] * [H x W] -> [H x W] -> sum -> Scalar, divided by Scalar
    return float((img * mask).sum() / total) if total > 0 else 0.0


def score(
    stencil: DisplayStencil, act: np.ndarray, digits: range | None = None
) -> float:
    """Contrast of activity under segments versus inside digit holes."""
    digits = digits if digits is not None else range(stencil.n_digits)
    # Union of all segment masks and of all hole masks: [H x W] each.
    seg = np.zeros_like(act)
    holes = np.zeros_like(act)
    for d in digits:
        for poly in stencil.segment_polygons(d).values():
            seg += stencil.rasterize(poly)  # [H x W] += [H x W]
        for poly in stencil.hole_polygons(d):
            holes += stencil.rasterize(poly)  # [H x W] += [H x W]
    # ([H x W], [H x W]) -> Scalar, twice -> Scalar - Scalar -> Scalar
    return _weighted_mean(act, seg) - _weighted_mean(act, holes)


def refine_offsets(stencil: DisplayStencil, act: np.ndarray) -> DisplayStencil:
    """Shift each digit horizontally (within +-1.5 strokes) to fit it alone."""
    dx = [0.0] * stencil.n_digits
    limit = 1.5 * stencil.stroke
    for d in range(stencil.n_digits):
        candidates = np.arange(-limit, limit + 1e-9, 0.25)  # [L] shifts in px
        scores = []  # L x Scalar
        for c in candidates:
            trial = replace(stencil, dx=[*dx[:d], float(c), *dx[d + 1 :]])
            scores.append(score(trial, act, range(d, d + 1)))  # [H x W] -> Scalar
        dx[d] = float(candidates[int(np.argmax(scores))])  # [L] -> Scalar
    return replace(stencil, dx=dx)


def initial_guess(act: np.ndarray, n_digits: int) -> DisplayStencil:
    """Rough stencil from the bounding box of high-activity pixels.

    The goal is only to start ``refine`` close enough to the right answer
    that its local search can't settle on a wrong optimum (e.g. the template
    shifted by a whole digit). The steps are:

    1. Find "hot" pixels: those that switch on and off a lot over the
       sessions. These are lit segments, almost never the background.
    2. Take their bounding box. Its height gives the digit height directly.
       Its width is less reliable, because a rarely used leading digit may be
       missing, so only its right edge is trusted.
    3. Fill in the remaining geometry from typical 7-segment proportions
       (digit width, spacing, slant, stroke) scaled by the height.
    4. Place the digits right-to-left from that right edge.
    """
    hgt, wid = act.shape  # act: [H x W] float32

    # 1. Ignore a band along the image border: the bezel and crop edges can
    #    wobble slightly between frames and show up as spurious activity.
    margin_y, margin_x = max(2, hgt // 10), max(2, wid // 20)
    inner = np.zeros_like(act, dtype=bool)  # [H x W] bool
    inner[margin_y:-margin_y, margin_x:-margin_x] = True
    # "Hot" = at least half the activity of the strongest pixels. The 99th
    # percentile (rather than the max) keeps one noisy pixel from setting
    # the scale.
    # act[inner]: [H x W] -> [L] inner pixels -> percentile -> Scalar.
    # [H x W] > Scalar -> [H x W] bool, & [H x W] bool -> [H x W] bool.
    hot = inner & (act > 0.5 * np.percentile(act[inner], 99))

    # 2. Bounding box of hot pixels. Top/bottom are reliable: every digit
    #    spans the full height and segments a/d light up often. Of the
    #    horizontal extent only the right edge x1 is used (see step 4).
    ys, xs = np.nonzero(hot)  # [H x W] bool -> [L] row, [L] column indices
    y0, y1, x1 = ys.min(), ys.max() + 1, xs.max() + 1  # [L] -> Scalar each
    h = float(y1 - y0)

    # 3. Typical proportions, all relative to digit height h:
    #    - slant: ~0.08 px of rightward lean per px of height (~5 degrees);
    #    - pitch: spacing of digit origins ~0.62 h (digit width + gap);
    #    - w: the digit cell takes ~75 % of the pitch, the rest is the gap;
    #    - stroke: segment thickness ~h / 9.
    #    They only need to be roughly right; refine() adjusts all of them.
    slant = 0.08
    pitch = h * 0.62
    w = pitch * 0.75

    # 4. Anchor at the right edge: the last digit changes in every recording,
    #    so its right side is always in the hot box. A stencil's x0 is the
    #    bottom-left of the first, unslanted cell. The top-right corner of
    #    the last digit is x0 + (n-1)*pitch + w + slant*h, so solve for x0.
    x_first = x1 - slant * h - w - (n_digits - 1) * pitch
    return DisplayStencil(
        width=wid,
        height=hgt,
        n_digits=n_digits,
        x0=float(x_first),
        y0=float(y0),
        w=w,
        h=h,
        pitch=pitch,
        slant=slant,
        stroke=h / 9,
    )


def refine(stencil: DisplayStencil, act: np.ndarray) -> DisplayStencil:
    """Coordinate descent over the geometric parameters.

    The template is improved by repeatedly nudging one parameter at a time:

    * For every parameter, try ``value + step`` and ``value - step``. Keep
      the change if it raises :func:`score` (activity under the segments
      minus activity in the digit holes).
    * Repeat passes over all parameters until no single nudge helps any
      more: a local optimum at the current step size.
    * Halve all step sizes and repeat, until pixel steps drop below
      0.2 px. Starting coarse lets the template travel a few pixels quickly;
      the finer steps then settle it to sub-pixel accuracy (the masks are
      supersampled, so sub-pixel moves do change the score).

    No gradients are needed and every intermediate state is a valid stencil,
    which keeps this simple and robust for 7 parameters on small images.
    Being a local search, it relies on :func:`initial_guess` starting within
    a couple of pixels of the answer.
    """
    # Initial step per parameter: 2 px for positions and sizes, 1 px for the
    # thin stroke, 0.04 for slant (a dimensionless shear: 0.04 * h is about
    # 2 px of shift at the top of a digit 50 px tall).
    steps = {
        "x0": 2.0,
        "y0": 2.0,
        "w": 2.0,
        "h": 2.0,
        "pitch": 2.0,
        "slant": 0.04,
        "stroke": 1.0,
    }
    # With a single digit the spacing has no effect on the masks, so tuning
    # it would only waste evaluations.
    if stencil.n_digits == 1:
        steps.pop("pitch")

    best, best_score = stencil, score(stencil, act)
    # Outer loop: one round per step size. Stop once the pixel-sized steps
    # are below 0.2 px. Slant is excluded from that test because its steps
    # are in different units; it is halved together with the rest.
    while max(v for k, v in steps.items() if k != "slant") > 0.2:
        # Inner loop: keep sweeping all parameters while any nudge helps.
        improved = True
        while improved:
            improved = False
            for name, step in steps.items():
                for sign in (1, -1):
                    cand = replace(best, **{name: getattr(best, name) + sign * step})
                    # Reject degenerate shapes. A stroke under 1 px would
                    # just sample the brightest centre line of a segment.
                    # A cell narrower than 3 strokes leaves no room for the
                    # hole between the left and right segments.
                    if cand.stroke < 1.0 or cand.w < 3 * cand.stroke:
                        continue
                    s = score(cand, act)  # [H x W] -> Scalar
                    # Greedy: accept any strict improvement at once, so the
                    # following parameters are tried from the updated state.
                    # The small epsilon keeps float noise from causing loops.
                    if s > best_score + 1e-6:
                        best, best_score, improved = cand, s, True
        # Local optimum at this resolution: refine with half-size steps.
        steps = {k: v / 2 for k, v in steps.items()}
    return best


def fit_dp(stencil: DisplayStencil, act: np.ndarray, after: int) -> tuple[float, float]:
    """Centre of the decimal point that follows digit ``after``.

    Searches a window around the bottom of the gap between digit ``after``
    and the next one, and picks the spot with the highest *activity*: the dp
    switches on and off between sessions, the background doesn't.

    Candidates overlapping a segment mask are rejected. Otherwise the search
    latches onto the end of segment c/d, which is also active and usually
    brighter than a small dot. (That's what went wrong on display 06: its dp
    sits ~3 px below the digits, almost on the bezel.)

    Limitation: a dp that is lit (or dark) in *every* frame has no activity
    and can't be located this way. Set its position by hand in the JSON.
    """
    s = stencil
    # Union of all segment masks: [H x W], > 0 where any segment is.
    segments = np.sum(list(s.segment_masks().values()), axis=0)
    occupied = segments > 0.05  # [H x W] bool

    base = s.x0 + after * s.pitch + (s.dx[after] if s.dx else 0.0)
    left, right = base + s.w - s.stroke, base + s.pitch + s.stroke
    # Vertically from just above the digit bottom to well below it (a quarter
    # of the digit height), clipped to the image.
    top = s.y0 + s.h - 2 * s.stroke
    bottom = min(s.y0 + s.h + 0.25 * s.h, s.height - 1.0)

    best, best_val = ((left + right) / 2, s.y0 + s.h), -np.inf
    for cx in np.arange(left, right, 0.25):
        for cy in np.arange(top, bottom, 0.25):
            trial = replace(s, dp={after: (float(cx), float(cy))})
            mask = trial.rasterize(trial.dp_polygon(after))  # [H x W]
            if (mask * occupied).sum() > 0.05 * mask.sum():  # [H x W] -> Scalar
                continue
            val = _weighted_mean(act, mask)  # ([H x W], [H x W]) -> Scalar
            if val > best_val:
                best, best_val = (float(cx), float(cy)), val
    return best


def fit_display(
    stack: np.ndarray, n_digits: int, dps: tuple[int, ...]
) -> DisplayStencil:
    act = activity(stack)  # [S x H x W] -> [H x W]
    stencil = refine_offsets(refine(initial_guess(act, n_digits), act), act)
    dp = {d: fit_dp(stencil, act, d) for d in dps}
    return replace(stencil, dp=dp)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", type=Path)
    parser.add_argument("--stencils", type=Path, default=Path("stencils"))
    parser.add_argument("--overlays", type=Path, default=Path("outputs/stencils"))
    parser.add_argument(
        "--overlay-only",
        action="store_true",
        help="don't fit, just redraw overlays from existing JSON",
    )
    parser.add_argument(
        "--display",
        type=int,
        action="append",
        help="only process this display (repeatable)",
    )
    args = parser.parse_args(argv)

    # S sessions x 10 displays -> transpose (zip) -> 10 displays x S frames.
    # Per display: S x [H_i x W_i] -> stack -> [S x H_i x W_i].
    stacks = [np.stack(d) for d in zip(*SessionLoader(args.sessions), strict=True)]
    args.stencils.mkdir(parents=True, exist_ok=True)
    args.overlays.mkdir(parents=True, exist_ok=True)
    for i in args.display or range(len(stacks)):
        path = args.stencils / f"display_{i:02d}.json"
        if args.overlay_only:
            stencil = DisplayStencil.load(path)
        else:
            stencil = fit_display(stacks[i], *LAYOUT[i])
            if path.exists():  # keep a calibrated threshold across refits
                stencil.threshold = DisplayStencil.load(path).threshold
            stencil.save(path)
            print(f"display {i:02d}: score {score(stencil, activity(stacks[i])):.1f}")
        # [S x H x W] -> [H x W] float -> stretched [H x W] uint8, for display.
        act = stretch(activity(stacks[i]))
        mean = stretch(stacks[i].astype(np.float32).mean(axis=0))
        a, m = overlay(act, stencil), overlay(mean, stencil)
        both = Image.new("RGB", (a.width, a.height * 2 + 6), (255, 255, 255))
        both.paste(a, (0, 0))
        both.paste(m, (0, a.height + 6))
        both.save(args.overlays / f"display_{i:02d}.png")


if __name__ == "__main__":
    main()
