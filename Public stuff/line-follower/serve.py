"""Serve the Line Follower page on localhost with caching off.

    python3 serve.py          # http://localhost:8000/
    python3 serve.py 8001     # another port

PyScript fetches its own .py files, and a plain `http.server` lets the browser keep old
copies. This one never does. Open the page in Chrome (Web Bluetooth needs it;
http://localhost counts as a secure context).
"""
import functools
import http.server
import os
import sys


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    here = os.path.dirname(os.path.abspath(__file__))
    handler = functools.partial(NoCacheHandler, directory=here)
    print(f"Line Follower: http://localhost:{port}/   (Ctrl-C to stop)")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()
