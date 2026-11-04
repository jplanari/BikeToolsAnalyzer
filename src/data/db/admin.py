"""
Server-wide figures for whoever runs the server: how many riders and rides, how much is stored, who is active.

READ-ONLY and not tied to one rider on purpose, so it is NOT re-exported from `src.data.db` and nothing in the app's pages
may import it: riders must never see these numbers (see docs/DATABASE.md, "One rider never sees another's data").
It is used by `tools/admin.py`, a terminal tool run on the server by the person who administers it.
"""
import os
import time
from datetime import datetime, timedelta, timezone

from . import core, dialect

REQUIRED = ("users", "rides")


def _all(conn, sql, params=()):
    return [tuple(r) for r in conn.execute(sql, params).fetchall()]


def _one(conn, sql, params=()):
    row = conn.execute(sql, params).fetchone()
    return row[0] if row else None


def _tables(conn):
    if dialect._on_pg(conn):
        rows = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema() "
                            "AND table_type = 'BASE TABLE' ORDER BY table_name").fetchall()
    else:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                            "ORDER BY name").fetchall()
    return [r[0] for r in rows]


def missing_tables(conn):
    have = set(_tables(conn))
    return [t for t in REQUIRED if t not in have]


def _count(conn, table, where=""):
    return int(_one(conn, f'SELECT COUNT(*) FROM "{table}" {where}') or 0)


def _day(days_ago):
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%d")


def _database_bytes(conn):
    """(total bytes, description of what was measured) for the whole database."""
    if dialect._on_pg(conn):
        return int(_one(conn, "SELECT pg_database_size(current_database())") or 0), "PostgreSQL database"
    total = 0
    for suffix in ("", "-wal", "-shm"):
        try:
            total += os.path.getsize(core.DB_FILE + suffix)
        except OSError:
            pass
    return total, f"SQLite file {core.DB_FILE}"


def table_report(conn=None):
    """[(table, rows, bytes_or_None)] largest first. Bytes need PostgreSQL, or SQLite built with dbstat."""
    def build(c):
        sizes = {}
        if dialect._on_pg(c):
            for name, size in _all(c, "SELECT c.relname, pg_total_relation_size(c.oid) FROM pg_class c "
                                      "JOIN pg_namespace n ON n.oid = c.relnamespace "
                                      "WHERE c.relkind = 'r' AND n.nspname = current_schema()"):
                sizes[name] = int(size)
        else:
            try:
                for name, size in _all(c, "SELECT name, SUM(pgsize) FROM dbstat GROUP BY name"):
                    sizes[name] = int(size)
            except dialect.operational_errors():
                pass
        out = [(t, _count(c, t), sizes.get(t)) for t in _tables(c)]
        return sorted(out, key=lambda r: (-(r[2] or 0), -r[1], r[0]))
    if conn is not None:
        return build(conn)
    with core.connect() as c:
        return build(c)


def overview():
    """One dict with the headline numbers. Raises RuntimeError when the database has not been created yet."""
    now = int(time.time())
    with core.connect() as c:
        absent = missing_tables(c)
        if absent:
            raise RuntimeError("The database has no table " + ", ".join(absent) + ": start the app once to create it.")
        have = set(_tables(c))
        has = have.__contains__
        db_bytes, where = _database_bytes(c)
        stored = int(_one(c, "SELECT COALESCE(SUM(length(content)), 0) FROM ride_files") or 0) if has("ride_files") else 0
        ov = {
            "database": where, "database_bytes": db_bytes, "ride_files_bytes": stored,
            "ride_files": _count(c, "ride_files") if has("ride_files") else 0,
            "riders": _count(c, "users"),
            "rides": _count(c, "rides"),
            "rides_last_7_days": _count(c, "rides", f"WHERE date_time >= '{_day(7)}'"),
            "rides_last_30_days": _count(c, "rides", f"WHERE date_time >= '{_day(30)}'"),
            "deleted_ride_tombstones": _count(c, "deleted_rides") if has("deleted_rides") else 0,
            "power_records": _count(c, "power_records") if has("power_records") else 0,
        }
        if "email" in dialect.columns(c, "users"):
            ov["riders_with_account"] = _count(c, "users", "WHERE password_hash IS NOT NULL")
            ov["accounts_confirmed"] = _count(c, "users", "WHERE email_verified_at IS NOT NULL")
            ov["accounts_waiting_for_confirmation"] = _count(c, "users", "WHERE password_hash IS NOT NULL "
                                                              "AND email_verified_at IS NULL")
            ov["new_accounts_last_7_days"] = _count(c, "users", f"WHERE created_at >= {now - 7 * 86400}")
            ov["new_accounts_last_30_days"] = _count(c, "users", f"WHERE created_at >= {now - 30 * 86400}")
        if has("sessions"):
            ov["sessions_active"] = _count(c, "sessions", f"WHERE expires_at > {now}")
            ov["riders_seen_last_7_days"] = int(_one(c, f"SELECT COUNT(DISTINCT user_id) FROM sessions "
                                                        f"WHERE last_seen >= {now - 7 * 86400}") or 0)
            ov["sessions_expired_not_purged"] = _count(c, "sessions", f"WHERE expires_at <= {now}")
        if has("account_tokens"):
            ov["tokens_expired_not_purged"] = _count(c, "account_tokens", f"WHERE expires_at <= {now}")
        for provider in ("strava", "wahoo", "polar"):
            if has(f"{provider}_auth"):
                ov[f"{provider}_connections"] = _count(c, f"{provider}_auth")
        if has("weekly_email"):
            ov["weekly_email_enabled"] = _count(c, "weekly_email", "WHERE enabled = 1")
        ov["tables"] = table_report(c)
    ov["per_rider_average"] = {
        "rides": round(ov["rides"] / ov["riders"], 1) if ov["riders"] else 0,
        "stored_bytes": int(ov["ride_files_bytes"] / ov["riders"]) if ov["riders"] else 0,
        "bytes_per_ride_file": int(ov["ride_files_bytes"] / ov["ride_files"]) if ov["ride_files"] else 0,
    }
    return ov


def riders_report():
    """One row per rider, biggest stored data first: name, e-mail, account state, rides, last ride, stored bytes,
    last seen (unix seconds or None), created (unix seconds or None), connected providers."""
    with core.connect() as c:
        if missing_tables(c):
            raise RuntimeError("The database has not been created yet: start the app once.")
        have = set(_tables(c))
        account = "email" in dialect.columns(c, "users")
        cols = "u.email, u.email_verified_at, u.password_hash IS NOT NULL, u.created_at" if account else \
               "NULL, NULL, FALSE, NULL" if dialect._on_pg(c) else "NULL, NULL, 0, NULL"
        seen = "(SELECT MAX(s.last_seen) FROM sessions s WHERE s.user_id = u.id)" if "sessions" in have else "NULL"
        files = ("(SELECT COALESCE(SUM(length(f.content)), 0) FROM ride_files f JOIN rides r2 ON r2.id = f.ride_id "
                 "WHERE r2.user_id = u.id)") if "ride_files" in have else "0"
        rows = _all(c, f"""SELECT u.id, u.name, {cols},
                                  (SELECT COUNT(*) FROM rides r WHERE r.user_id = u.id),
                                  (SELECT MAX(r.date_time) FROM rides r WHERE r.user_id = u.id),
                                  {files}, {seen}
                           FROM users u""")
        providers = {}
        for provider in ("strava", "wahoo", "polar"):
            if f"{provider}_auth" in have:
                for (uid,) in _all(c, f"SELECT user_id FROM {provider}_auth"):
                    providers.setdefault(uid, []).append(provider)
    out = []
    for uid, name, email, verified, has_pw, created, n, last_ride, stored, seen_at in rows:
        state = "no account (desktop-style)" if not has_pw else "confirmed" if verified else "waiting for confirmation"
        out.append({"name": name, "email": email, "state": state, "rides": int(n or 0),
                    "last_ride": str(last_ride)[:10] if last_ride else None, "stored_bytes": int(stored or 0),
                    "last_seen": int(seen_at) if seen_at else None, "created": int(created) if created else None,
                    "providers": providers.get(uid, [])})
    return sorted(out, key=lambda r: (-r["stored_bytes"], r["name"].lower()))
