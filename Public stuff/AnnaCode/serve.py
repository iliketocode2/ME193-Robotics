"""Serve AnnaCode on localhost with caching switched off.

    python3 serve.py          # http://localhost:8000/AnnaCode/
    python3 serve.py 8001     # pick another port

This serves the whole "Public stuff" folder (the one above AnnaCode), because AnnaCode's
pages load the shared Bluetooth helper ../QLearn/webhub.py from the folder next door.

Plain `python3 -m http.server` works too, but it lets the browser cache files, and
PyScript fetches its .py files itself, so after you switch branches or edit a file the
page can keep running the old code (a hard reload does not always fix it). This server
tells the browser never to keep a copy. It only listens on this computer.
"""
import http.server
import os
import sys


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):      # the bare address: go to AnnaCode
            self.send_response(302)
            self.send_header("Location", "/AnnaCode/")
            self.end_headers()
            return
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # "Public stuff"
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), NoCacheHandler) as server:
        print(f"AnnaCode on http://localhost:{port}/AnnaCode/   (builder)")
        print(f"                http://localhost:{port}/AnnaCode/trainer.html   (trainer)")
        print("Ctrl-C to stop.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
