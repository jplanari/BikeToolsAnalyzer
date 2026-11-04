"""
Administration view of a BikeTools server, in the terminal. Read-only: it changes nothing.

Run it on the server, with the same settings the app uses (BIKETOOLS_DATABASE_URL for PostgreSQL, or BIKETOOLS_DB for a
SQLite file):

    python tools/admin.py              overview: riders, rides, stored data, accounts, activity
    python tools/admin.py riders       one line per rider
    python tools/admin.py tables       rows and size of every table
    python tools/admin.py --json       the same, as JSON (any command)

Rider names and e-mail addresses are shown, so treat the output as private.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.db import admin                                   # noqa: E402


def size(n):
    if n is None:
        return "-"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def ago(ts, now=None):
    if not ts:
        return "-"
    days = int(((now or time.time()) - ts) // 86400)
    return "today" if days <= 0 else f"{days} d ago"


def table(rows, header, align=None):
    """Plain aligned text. align: string of 'l'/'r' per column."""
    cells = [header] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    align = align or "l" * len(header)
    lines = []
    for k, r in enumerate(cells):
        lines.append("  ".join(c.rjust(w) if a == "r" else c.ljust(w) for c, w, a in zip(r, widths, align)).rstrip())
        if k == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def show_overview(ov):
    avg = ov["per_rider_average"]
    rows = [
        ("Database", f"{size(ov['database_bytes'])}  ({ov['database']})"),
        ("Original ride files", f"{ov['ride_files']} files, {size(ov['ride_files_bytes'])}"
                                f"  (average {size(avg['bytes_per_ride_file'])} per file)"),
        ("Riders", ov["riders"]),
    ]
    if "riders_with_account" in ov:
        rows += [("  with a sign-in account", f"{ov['riders_with_account']} ({ov['accounts_confirmed']} confirmed, "
                                               f"{ov['accounts_waiting_for_confirmation']} waiting for e-mail confirmation)"),
                 ("  new accounts", f"{ov['new_accounts_last_7_days']} in 7 days, {ov['new_accounts_last_30_days']} in 30 days")]
    if "riders_seen_last_7_days" in ov:
        rows += [("  signed in during the last 7 days", ov["riders_seen_last_7_days"]),
                 ("  sessions open now", ov["sessions_active"])]
    rows += [("Activities (rides)", ov["rides"]),
             ("  dated in the last 7 days", ov["rides_last_7_days"]),
             ("  dated in the last 30 days", ov["rides_last_30_days"]),
             ("  average per rider", avg["rides"]),
             ("  average stored data per rider", size(avg["stored_bytes"])),
             ("Deleted synced rides remembered", ov["deleted_ride_tombstones"])]
    links = [f"{p} {ov[p + '_connections']}" for p in ("strava", "wahoo", "polar") if p + "_connections" in ov]
    if links:
        rows.append(("Provider connections", ", ".join(links)))
    if "weekly_email_enabled" in ov:
        rows.append(("Weekly e-mail switched on", ov["weekly_email_enabled"]))
    stale = ov.get("sessions_expired_not_purged", 0) + ov.get("tokens_expired_not_purged", 0)
    if stale:
        rows.append(("Expired sessions/tokens not cleaned", stale))
    width = max(len(str(r[0])) for r in rows)
    print("\n".join(f"{str(k).ljust(width)}  {v}" for k, v in rows))
    print("\nLargest tables:")
    print(show_tables(ov["tables"][:5]))


def show_tables(rows):
    return table([(t, f"{n:,}", size(b)) for t, n, b in rows], ["table", "rows", "size"], "lrr")


def show_riders(rows, now=None):
    return table([(r["name"], r["email"] or "-", r["state"], r["rides"], r["last_ride"] or "-", size(r["stored_bytes"]),
                   ago(r["last_seen"], now), ", ".join(r["providers"]) or "-") for r in rows],
                 ["rider", "e-mail", "account", "rides", "last ride", "stored", "last seen", "connected"], "lllrlrll")


def main(argv=None):
    p = argparse.ArgumentParser(description="Read-only administration view of a BikeTools database.")
    p.add_argument("command", nargs="?", default="overview", choices=["overview", "riders", "tables"])
    p.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = p.parse_args(argv)
    try:
        if args.command == "overview":
            data = admin.overview()
        elif args.command == "riders":
            data = admin.riders_report()
        else:
            data = admin.table_report()
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(data, indent=2, default=str))
    elif args.command == "overview":
        show_overview(data)
    elif args.command == "riders":
        print(show_riders(data) if data else "No riders yet.")
    else:
        print(show_tables(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
