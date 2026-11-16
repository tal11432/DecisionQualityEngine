"""Local web app for the Decision Score: python app/server.py  ->  http://127.0.0.1:8000  (no outside access; nothing is uploaded: the screenshot stays in your browser)."""
import json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api import score_situation
HERE = os.path.dirname(os.path.abspath(__file__))

class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode(); self.send_response(code); self.send_header("Content-Type", ctype + "; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path in ("/", "/index.html"): self._send(200, open(os.path.join(HERE, "static", "index.html"), "rb").read(), "text/html")
        else: self._send(404, json.dumps({"error": "not found"}))
    def do_POST(self):
        if self.path != "/api/score": return self._send(404, json.dumps({"error": "not found"}))
        try:
            n = int(self.headers.get("Content-Length", 0)); self._send(200, json.dumps(score_situation(json.loads(self.rfile.read(n) or b"{}"))))
        except ValueError as e: self._send(400, json.dumps({"error": str(e)}))
        except Exception as e: self._send(500, json.dumps({"error": f"internal error: {e}"}))
    def log_message(self, *a): pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000)); print(f"Decision Score app: http://127.0.0.1:{port}", flush=True); ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
