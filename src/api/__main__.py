"""Run the phone server and manage the phone's access tokens.

    python -m src.api serve [--host 0.0.0.0] [--port 8765]
    python -m src.api token create "<rider>" [--label "Pixel 8"]
    python -m src.api token list "<rider>"
    python -m src.api token revoke "<rider>" <id>

The database is the same one the desktop app uses (BIKETOOLS_DB)."""
import argparse
import datetime
import logging
import socketserver
import sys
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

from src.data import db
from .app import application


class ThreadingServer(socketserver.ThreadingMixIn, WSGIServer):
    daemon_threads = True


class QuietHandler(WSGIRequestHandler):
    def log_message(self, fmt, *args):          # never log the query string or headers: they can carry tokens
        line = str(args[0]).split("?")[0] if args else ""
        logging.getLogger("biketools.api").info("%s %s %s", self.address_string(), line, " ".join(str(a) for a in args[1:]))


def _when(ts):
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "never"


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.api", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="start the phone server")
    s.add_argument("--host", default="127.0.0.1", help="address to listen on (0.0.0.0 = reachable from other devices)")
    s.add_argument("--port", type=int, default=8765)
    tk = sub.add_parser("token", help="manage phone access tokens").add_subparsers(dest="action", required=True)
    c = tk.add_parser("create")
    c.add_argument("rider")
    c.add_argument("--label", default="")
    tk.add_parser("list").add_argument("rider")
    r = tk.add_parser("revoke")
    r.add_argument("rider")
    r.add_argument("id", type=int)
    a = p.parse_args(argv)

    db.init_db()
    if a.cmd == "serve":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        with make_server(a.host, a.port, application, server_class=ThreadingServer, handler_class=QuietHandler) as httpd:
            print(f"BikeTools phone server on http://{a.host}:{a.port}  (Ctrl+C to stop)")
            if a.host not in ("127.0.0.1", "localhost"):
                print("Reachable from other devices. Plain HTTP is only safe on a network you trust; "
                      "see mobile/README.md for Tailscale or HTTPS.")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("stopped")
        return 0
    if a.action == "create":
        token, tid = db.create_api_token(a.rider, a.label)
        if not token:
            print(f"No rider called {a.rider!r}.", file=sys.stderr)
            return 1
        print(f"Token #{tid} for {a.rider}. Copy it now; it is not shown again:\n\n{token}\n")
        return 0
    if a.action == "list":
        rows = db.list_api_tokens(a.rider)
        for t in rows:
            print(f"#{t['id']}  {t['label'] or '-':<20} created {_when(t['created_at'])}  last used {_when(t['last_used_at'])}")
        if not rows:
            print("No tokens.")
        return 0
    ok = db.revoke_api_token(a.rider, a.id)
    print("Revoked." if ok else "No such token for that rider.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
