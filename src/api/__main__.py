"""Run the phone server and manage the phone's access tokens.

    python -m src.api serve [--host 0.0.0.0] [--port 8765]
    python -m src.api token create "<rider>" [--label "Pixel 8"]
    python -m src.api token list "<rider>"
    python -m src.api token revoke "<rider>" <id>
    python -m src.api password set "<rider>"        (asks for the password)
    python -m src.api password remove "<rider>"
    python -m src.api invite create [--uses 1] [--days 7] [--label "for Marta"]
    python -m src.api invite list | revoke <id>

The database is the same one the desktop app uses (BIKETOOLS_DB)."""
import argparse
import datetime
import logging
import sys

from src.data import db
from . import runner


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
    pw = sub.add_parser("password", help="set or remove a rider's password").add_subparsers(dest="action", required=True)
    pw.add_parser("set").add_argument("rider")
    pw.add_parser("remove").add_argument("rider")
    inv = sub.add_parser("invite", help="invitation codes for new riders").add_subparsers(dest="action", required=True)
    ic = inv.add_parser("create")
    ic.add_argument("--uses", type=int, default=1, help="how many riders can sign up with it")
    ic.add_argument("--days", type=float, default=7, help="valid for this many days (0 = no expiry)")
    ic.add_argument("--label", default="")
    inv.add_parser("list")
    inv.add_parser("revoke").add_argument("id", type=int)
    a = p.parse_args(argv)

    db.init_db()
    if a.cmd == "serve":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        with runner.make(a.host, a.port) as httpd:
            print(f"BikeTools phone server on http://{a.host}:{a.port}  (Ctrl+C to stop)")
            if a.host not in ("127.0.0.1", "localhost"):
                print("Reachable from other devices. Plain HTTP is only safe on a network you trust; "
                      "see mobile/README.md for Tailscale or HTTPS.")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("stopped")
        return 0
    if a.cmd == "invite":
        if a.action == "create":
            code, iid = db.create_invite(a.uses, a.days or None, a.label)
            print(f"Invitation #{iid}, {a.uses} use(s), {'no expiry' if not a.days else f'{a.days:g} day(s)'}. "
                  f"Give it to the person now; it is not shown again:\n\n{code}\n")
            return 0
        if a.action == "list":
            rows = db.list_invites()
            for r in rows:
                print(f"#{r['id']}  {r['label'] or '-':<20} {r['uses_left']} use(s) left  expires {_when(r['expires_at']) if r['expires_at'] else 'never'}")
            if not rows:
                print("No valid invitations.")
            return 0
        ok = db.revoke_invite(a.id)
        print("Revoked." if ok else "No such invitation.")
        return 0 if ok else 1
    if a.cmd == "password":
        if a.action == "remove":
            ok, msg = db.clear_user_password(a.rider, force=True)
        else:
            import getpass
            first = getpass.getpass("New password: ")
            if first != getpass.getpass("Repeat it: "):
                print("The two passwords differ.", file=sys.stderr)
                return 1
            ok, msg = db.set_user_password(a.rider, first, force=True)
        print(msg or "Done.")
        return 0 if ok else 1
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
