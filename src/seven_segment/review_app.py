"""Tiny local web app to review readings and record the correct value.

Run ``uv run python -m seven_segment.review_app`` and open the printed URL.
For every (session, display) it shows the original image, a render of only
the segments the reader found lit, and the decoded reading. Confirm the
reading or type the correct one. Labels are stored in
``labels/readings.csv`` (one row per checked image) and can be used as
ground truth to measure the reader's accuracy.

Uses only the standard library plus Pillow; the UI is ``review_app.html``.

Array shape notation: H, W = image height, width; N = segments of a display.
"""

import argparse
import csv
import io
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw

from seven_segment.segments import SegmentReader, load_labels, read_all
from seven_segment.sessions import SessionLoader
from seven_segment.stencils import DisplayStencil

HTML = Path(__file__).with_name("review_app.html")
LABEL_FIELDS = ["session", "display", "label", "predicted"]


class Review:
    """Data behind the app: images, predictions, and the labels file."""

    def __init__(self, sessions: Path, stencils: Path, labels: Path) -> None:
        self.loader = SessionLoader(sessions)
        self.sessions = list(self.loader)  # S x (10 x [H_i x W_i])
        paths = sorted(stencils.glob("display_*.json"))
        self.stencils = [DisplayStencil.load(p) for p in paths]
        self.readers = [SegmentReader(s) for s in self.stencils]
        readings, self.thresholds = read_all(self.loader, self.stencils)
        # predicted[(session, display)] -> text, e.g. "15.4" (1-based session)
        self.predicted = {
            (k, i): r.strip()
            for k, row in enumerate(readings, start=1)
            for i, r in enumerate(row)
        }
        self.labels_path = labels
        self.labels: dict[tuple[int, int], str] = {}
        self._lock = threading.Lock()
        self._png_cache: dict[tuple[int, int, str, int], bytes] = {}
        self.labels.update(load_labels(labels))

    def state(self) -> dict:
        items = [
            {
                "session": k,
                "display": i,
                "predicted": p,
                "label": self.labels.get((k, i)),
            }
            for (k, i), p in sorted(self.predicted.items())
        ]
        return {
            "n_sessions": len(self.sessions),
            "n_displays": len(self.stencils),
            # [width, height] of each display's crop, for layout in the page.
            "sizes": [[s.width, s.height] for s in self.stencils],
            "items": items,
        }

    def set_label(self, session: int, display: int, label: str | None) -> None:
        """Store (or with ``None``/empty, remove) a label and rewrite the CSV."""
        key = (session, display)
        if key not in self.predicted:
            raise KeyError(key)
        with self._lock:
            if label:
                self.labels[key] = label
            else:
                self.labels.pop(key, None)
            self.labels_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.labels_path.with_suffix(".tmp")
            with tmp.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(LABEL_FIELDS)
                for (k, i), lab in sorted(self.labels.items()):
                    w.writerow([k, i, lab, self.predicted[(k, i)]])
            tmp.replace(self.labels_path)  # atomic: never a half-written file

    def image_png(self, session: int, display: int, kind: str, scale: int) -> bytes:
        """PNG of the raw crop, or of the crop with the segment overlay (cached)."""
        cache_key = (session, display, kind, scale)
        if cache_key not in self._png_cache:
            self._png_cache[cache_key] = self._render_png(session, display, kind, scale)
        return self._png_cache[cache_key]

    def _render_png(self, session: int, display: int, kind: str, scale: int) -> bytes:
        """Render one of three views of a crop, upscaled by ``scale``:

        * ``raw``: the original grayscale image;
        * ``overlay``: the original with lit segments filled green and unlit
          ones outlined red;
        * ``segments``: only the segments the reader found lit, drawn on a
          dark panel, i.e. what the reader "saw", without the photo.
        """
        if session < 1:
            raise IndexError(session)
        if kind not in ("raw", "overlay", "segments"):
            raise ValueError(kind)
        gray = self.sessions[session - 1][display]  # [H x W] uint8
        size = (gray.shape[1] * scale, gray.shape[0] * scale)
        if kind == "segments":
            img = Image.new("RGB", size, (24, 26, 24))
        else:
            # [H x W] -> RGB, nearest-neighbour upscale -> [H*scale x W*scale x 3]
            img = (
                Image.fromarray(gray)
                .convert("RGB")
                .resize(size, Image.Resampling.NEAREST)
            )
        if kind != "raw":
            img = self._draw_segments(img, session, display, scale, kind)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _segment_polygons(self, display: int) -> dict[str, list[tuple[float, float]]]:
        """All segment polygons of a display keyed like the reader, e.g. "0.a"."""
        stencil = self.stencils[display]
        polys = {}
        for d in range(stencil.n_digits):
            for name, poly in stencil.segment_polygons(d).items():
                polys[f"{d}.{name}"] = poly
            if d in stencil.dp:
                polys[f"{d}.dp"] = stencil.dp_polygon(d)
        return polys

    def _draw_segments(
        self, img: Image.Image, session: int, display: int, scale: int, kind: str
    ) -> Image.Image:
        lit = self.readers[display].lit(
            self.sessions[session - 1][display], self.thresholds[display]
        )  # [H x W] -> N bools
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for name, poly in self._segment_polygons(display).items():
            pts = [(x * scale, y * scale) for x, y in poly]
            if lit[name]:
                if kind == "segments":
                    draw.polygon(pts, fill=(90, 235, 120, 255))
                else:
                    draw.polygon(
                        pts, fill=(40, 220, 90, 140), outline=(40, 220, 90, 255)
                    )
            elif kind == "overlay":
                draw.polygon(pts, outline=(255, 90, 90, 200))
        return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def make_handler(review: Review) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # quiet
            pass

        def _send(self, body: bytes, ctype: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: object, status: int = 200) -> None:
            self._send(json.dumps(obj).encode(), "application/json", status)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path == "/":
                # Embed the initial state so the page renders without waiting
                # for a second request. "</" is escaped so the JSON can't end
                # the <script> element early.
                state = json.dumps(review.state()).replace("</", "<\\/")
                page = HTML.read_text().replace("/*__STATE__*/null", state)
                self._send(page.encode(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                self._json(review.state())
            elif url.path == "/img":
                q = {k: v[0] for k, v in parse_qs(url.query).items()}
                try:
                    png = review.image_png(
                        int(q["session"]),
                        int(q["display"]),
                        q.get("kind", "raw"),
                        min(int(q.get("scale", 6)), 12),
                    )
                except KeyError, ValueError, IndexError:
                    self._json({"error": "bad image request"}, HTTPStatus.BAD_REQUEST)
                    return
                self._send(png, "image/png")
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/label":
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                label = (body.get("label") or "").strip() or None
                review.set_label(int(body["session"]), int(body["display"]), label)
            except KeyError, ValueError, TypeError, json.JSONDecodeError:
                self._json({"error": "bad label request"}, HTTPStatus.BAD_REQUEST)
                return
            self._json({"ok": True})

    return Handler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, default=Path("data/sessions"))
    parser.add_argument("--stencils", type=Path, default=Path("stencils"))
    parser.add_argument("--labels", type=Path, default=Path("labels/readings.csv"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    print("loading sessions and decoding readings ...", flush=True)
    review = Review(args.sessions, args.stencils, args.labels)
    # Bind to localhost only: this is a personal tool, not a public server.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(review))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"review app running at {url}  (Ctrl+C to stop)")
    print(f"labels are saved to {args.labels}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
