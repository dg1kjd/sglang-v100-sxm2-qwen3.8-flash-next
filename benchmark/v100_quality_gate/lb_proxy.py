"""Least-in-flight HTTP fan-out across the two identical serving instances.

sgl-eval takes a single --base-url, so to use both halves of the box we put
this in front of them. Dispatch is least-in-flight rather than round-robin:
GSM8K completion lengths are very skewed (the 24-example smoke spent 60 s of
78 s on one straggler), and strict alternation would keep queueing work behind
a busy instance while the other sits idle.

Usage:
    python lb_proxy.py --port 11400 \
        --upstream http://SERVER_HOST:30000 \
        --upstream http://SECOND_HOST:30001
"""

import argparse
import itertools
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_lock = threading.Lock()
_inflight: dict[str, int] = {}
_served: dict[str, int] = {}
_rr = itertools.count()


def pick_upstream(upstreams: list[str]) -> str:
    """Least in-flight, ties broken round-robin so a cold start does not stick."""
    with _lock:
        fewest = min(_inflight[u] for u in upstreams)
        candidates = [u for u in upstreams if _inflight[u] == fewest]
        chosen = candidates[next(_rr) % len(candidates)]
        _inflight[chosen] += 1
        _served[chosen] += 1
        return chosen


def release(upstream: str) -> None:
    with _lock:
        _inflight[upstream] -= 1


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstreams: list[str] = []

    def log_message(self, fmt, *args):  # quieter than the default stderr spam
        pass

    def _proxy(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None

        # /v1/models and other metadata are identical on both; pin them to the
        # first upstream so a model-list probe never counts as load.
        if method == "GET" or self.path.rstrip("/").endswith("/models"):
            upstream, counted = self.upstreams[0], False
        else:
            upstream, counted = pick_upstream(self.upstreams), True

        req = urllib.request.Request(
            upstream + self.path,
            data=body,
            method=method,
            headers={
                k: v
                for k, v in self.headers.items()
                if k.lower() not in ("host", "content-length", "connection")
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                payload, status = r.read(), r.status
        except urllib.error.HTTPError as e:
            payload, status = e.read(), e.code
        except Exception as e:  # upstream down / reset: report as 502
            payload = json.dumps({"error": {"message": f"{upstream}: {e}"}}).encode()
            status = 502
        finally:
            if counted:
                release(upstream)

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/_lb_stats":
            payload = json.dumps({"inflight": _inflight, "served": _served}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self._proxy("GET")

    def do_POST(self):
        self._proxy("POST")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11400)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--upstream", action="append", required=True)
    args = ap.parse_args()

    Handler.upstreams = [u.rstrip("/") for u in args.upstream]
    for u in Handler.upstreams:
        _inflight[u] = 0
        _served[u] = 0

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"lb on {args.host}:{args.port} -> {Handler.upstreams}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
