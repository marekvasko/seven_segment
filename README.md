# seven-segment

Recognition of 7-segment display readings (bachelor's thesis).

## Setup

```sh
uv sync
```

## Dataset

Grayscale crops of 10 displays captured in 64 sessions, stored as
`data/sessions/session_NNN/display_DD.png`. To recreate from the original archive:

```sh
unzip grayscale.zip -d /tmp/raw
uv run python -m seven_segment.sessions /tmp/raw/grayscale data/sessions
```

Loading:

```python
from seven_segment.sessions import SessionLoader

loader = SessionLoader("data/sessions")
displays = loader[0]  # session_001 -> list of 10 uint8 arrays (H, W)
for displays in loader:  # iterate over all sessions
    ...
```

## Reading segments

Each display has a geometric stencil in `stencils/display_DD.json`: a mask per
segment (`"<digit>.<a-g|dp>"`) for every digit, including segments that never
lit in the recordings. A segment is lit if its brightness exceeds the digit's
background by the display's calibrated threshold.

```python
from pathlib import Path
from seven_segment.sessions import SessionLoader
from seven_segment.stencils import DisplayStencil
from seven_segment.segments import SegmentReader

reader = SegmentReader(DisplayStencil.load(Path("stencils/display_01.json")))
img = SessionLoader("data/sessions")[8][1]
reader.lit(img)  # {"0.a": False, "0.b": True, ..., "1.dp": True, ...}
reader.read(img)  # "15.4"
```

Pipeline to rebuild stencils and print all readings:

```sh
uv run python -m seven_segment.fit_stencils data/sessions   # -> stencils/, outputs/stencils/
uv run python -m seven_segment.segments data/sessions --save
```

`segments` also saves the results to `outputs/readings/` (change with `--out`):

- `readings.csv`: one row per session, one column per display (`?` marks an
  unknown segment pattern);
- `segments.csv`: every segment decision: session, display, segment, feature
  (segment minus hole brightness), threshold, lit (0/1).

## Review app

A small local web app to check the reader's output and record the correct
value where it's wrong:

```sh
uv run python -m seven_segment.review_app      # opens http://127.0.0.1:8765/
```

It shows one display at a time as a grid of tiles, one per session. Each tile
has the original image, below it a render of only the segments the reader
found lit, and a box pre-filled with the reading. Press **Enter** (or ✓) to
submit the value in the box and jump to the next tile; type the correct value
first if it's wrong. **Shift+Enter** goes back, **Esc** resets the box, **×**
removes a label. Filter by display or by status (unchecked, invalid,
corrected). The URL keeps your place (e.g. `#s11-d1`).

Labels are saved immediately to `labels/readings.csv`
(`session,display,label,predicted`). This file is hand-checked ground truth
and **is committed**. The header shows the reader's accuracy on the
checked items.

## Demo notebook

`notebooks/stencil_demo.ipynb` walks through stencil fitting step by step and
shows how to use `SegmentReader`. Run it with the notebook dependencies:

```sh
uv sync --group notebook
uv run --group notebook jupyter lab notebooks/stencil_demo.ipynb   # or open it in VS Code
```

See [AGENTS.md](AGENTS.md) for development and contribution guidelines.
