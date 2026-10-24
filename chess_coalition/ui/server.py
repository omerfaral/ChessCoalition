from __future__ import annotations

import json


def serve_ui(events: list[dict], port: int):
    """ui.html'i ve /events?since=N ile olay listesini sunan küçük yerel sunucu."""
    import threading
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from pathlib import Path
    from urllib.parse import parse_qs, urlparse

    page = Path(__file__).with_name("ui.html")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/events":
                since = int(parse_qs(url.query).get("since", ["0"])[0])
                body, ctype = json.dumps(events[since:], ensure_ascii=False).encode(), "application/json"
            else:
                body, ctype = page.read_bytes(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    print(f"UI: {url}")
    webbrowser.open(url)
