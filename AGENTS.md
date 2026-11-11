# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository.

## Project

Project on recognising readings from **7-segment displays**.

- `src/seven_segment/` – Python package (src layout).
- `tests/` – pytest tests, mirroring module names (`test_<module>.py`).
- `data/sessions/session_NNN/display_DD.png` – grayscale display crops grouped
  by capture session (64 sessions × 10 displays). `display_DD` is always the
  same physical display; displays differ in size. Built from `grayscale.zip`
  with `uv run python -m seven_segment.sessions <flat_dir> data/sessions`.
  Load with `seven_segment.sessions.SessionLoader`.
- `stencils/display_DD.json` – fitted per-segment stencil geometry + calibrated
  lit threshold for each display. These are small hand-checkable annotations
  and **are committed**. Regenerate with
  `uv run python -m seven_segment.fit_stencils data/sessions` (writes overlays
  to `outputs/stencils/` – always look at them), then calibrate with
  `uv run python -m seven_segment.segments data/sessions --save`. If a fit is
  off, hand-edit the JSON and redraw with `fit_stencils ... --overlay-only`.
- `labels/readings.csv` – hand-checked readings recorded with the review app
  (`uv run python -m seven_segment.review_app`). It's ground truth: commit it,
  never regenerate or overwrite it from code, and use it to measure accuracy.
- `notebooks/` – demo/exploration notebooks (dependency group `notebook`:
  `uv sync --group notebook`). Commit them executed, with outputs, and re-run
  top to bottom (`uv run jupyter nbconvert --to notebook --execute --inplace
  <nb>`) before committing so outputs match the code. Ruff lints them too.
- `outputs/` – generated images/reports, git-ignored.
- `data/` and `*.zip` are git-ignored; never commit datasets or model weights.

## Tooling: always use `uv`

Never call `pip`, `python -m venv` or a bare `python` directly.

| Task | Command |
| --- | --- |
| Install / sync env | `uv sync` |
| Run a module/script | `uv run python -m seven_segment.<module>` |
| Add a dependency | `uv add <pkg>` |
| Add a dev dependency | `uv add --dev <pkg>` |
| Remove a dependency | `uv remove <pkg>` |
| Run tests | `uv run pytest` |
| Lint | `uv run ruff check` (auto-fix: `--fix`) |
| Format | `uv run ruff format` |
| Type check | `uv run ty check` |

`pyproject.toml` and `uv.lock` are the source of truth for dependencies; commit
both together whenever dependencies change. Do not edit `uv.lock` by hand.

## Code style

- Python ≥ 3.14 (see `.python-version`); use modern syntax (`list[int]`,
  `X | None`, `pathlib.Path` instead of `os.path`).
- Type hints on all public functions; short docstrings explaining *why*.
- Keep functions small and pure where possible; put I/O and CLI parsing
  (`argparse`, `main()`) at the edges so logic is testable.
- Annotate array operations with their shapes in comments, using the
  notation in the module docstrings (`S` sessions, `H`/`W` image size, `D`
  digits, `N` segments, `L` generic length), e.g.
  `# [S x H x W] -> std over sessions -> [H x W]` or `# [H x W] -> Scalar`.
- No hard-coded absolute paths – take paths as arguments.
- Code must pass `ruff check`, `ruff format` and `ty check` with no changes
  or errors. Avoid builtin `sum()` over NumPy arrays (it types as `int | ndarray`);
  use `np.sum(..., axis=0)`.

## Contributing workflow

1. Work on a branch (`feature/<topic>`, `fix/<topic>`); don't commit straight
   to `main` for non-trivial changes.
2. Keep changes focused: one logical change per commit. Don't reformat or
   refactor unrelated code in the same commit.
3. Add or update tests for any behaviour change. Tests must not depend on the
   real dataset – generate small synthetic images in `tmp_path`.
4. Before committing, all of these must pass:
   ```sh
   uv run ruff format && uv run ruff check && uv run ty check && uv run pytest
   ```
5. Commit messages: imperative mood, short summary line (≤ 72 chars), optional
   body explaining *why*. E.g. `Add per-segment threshold classifier`.
6. Never modify or delete raw data in place without keeping the original
   (`grayscale.zip` is the pristine copy). Data-transforming scripts must be
   reproducible from the command line.
7. If a task is ambiguous (e.g. which display types to include, how to label
   data), ask rather than guess.
