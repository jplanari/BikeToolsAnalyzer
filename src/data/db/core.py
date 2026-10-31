"""Connections, backup and the date filters every query uses."""
from contextlib import contextmanager
from src.analysis.dates import local_day_bounds_utc
from src.analysis.dates import to_local_naive
import os
import sqlite3
import time

# BIKETOOLS_DB lets you keep the database somewhere else, e.g. outside a OneDrive/Dropbox folder:
# those sync tools (and Windows drives mounted in WSL) lock files unpredictably, which SQLite hates.
DB_FILE = os.environ.get("BIKETOOLS_DB", "bikethools.db")
BUSY_TIMEOUT_S = 60          # wait this long for another connection's lock instead of failing at once


def _connect():
    """
    Every connection goes through here. SQLite only enforces foreign keys (and so
    ON DELETE CASCADE) when asked to, per connection; without this, deleting a user
    left their rides, records and histograms behind. synchronous=NORMAL is the safe, faster
    setting for WAL (a crash can lose the last commit, never corrupt the file).
    """
    conn = sqlite3.connect(DB_FILE, timeout=BUSY_TIMEOUT_S)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


@contextmanager
def connect(row_factory=None):
    """
    `with connect() as conn:` commits when the block ends, rolls back when it raises, and always closes,
    so an early return or an exception cannot leak a connection or leave a half-written change.
    """
    conn = _connect()
    if row_factory is not None:
        conn.row_factory = row_factory
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _add_column(c, table, column, definition):
    """ALTER TABLE ... ADD COLUMN when the column is missing. Any other error is a real one and propagates."""
    existing = {row[1] for row in c.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _enable_wal():
    """
    Write-ahead logging: readers no longer block the writer (the background weekly e-mail writes while the
    page reads). It needs shared memory, which network and some synced/mounted drives lack: when SQLite
    stays on its default journal there, that is fine, the app just works as before.
    """
    conn = sqlite3.connect(DB_FILE, timeout=BUSY_TIMEOUT_S)
    try:
        return conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()
        

BACKUP_KEEP = 3

def backup_db(reason="manual", keep=None):
    """
    Consistent copy of the database to backups/ next to it (SQLite's backup API, safe
    while the app is running). Keeps the newest `keep` backups. Returns the path, or
    None if there is no database yet or the copy failed; callers go on either way,
    since a failed backup should not block the work it was protecting.
    """
    import os, glob
    if not os.path.exists(DB_FILE):
        return None
    folder = os.path.join(os.path.dirname(os.path.abspath(DB_FILE)), "backups")
    stem = os.path.splitext(os.path.basename(DB_FILE))[0]
    dest = os.path.join(folder, f"{stem}-{time.strftime('%Y%m%d-%H%M%S')}-{reason}.db")
    try:
        os.makedirs(folder, exist_ok=True)
        src, dst = sqlite3.connect(DB_FILE, timeout=BUSY_TIMEOUT_S), sqlite3.connect(dest)
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
    except (OSError, sqlite3.Error):
        if os.path.exists(dest):
            os.remove(dest)
        return None
    old = sorted(glob.glob(os.path.join(folder, f"{stem}-*.db")))[:-(keep or BACKUP_KEEP)]
    for path in old:
        try:
            os.remove(path)
        except OSError:
            pass
    return dest

def data_stamp():
    """
    Changes whenever the database file is written. UI caches key on it, so cached reads
    stay valid between interactions and refresh themselves after any save or update.
    """
    import os
    stamp = [DB_FILE]
    for path in (DB_FILE, DB_FILE + "-wal"):
        try:
            st = os.stat(path)
            stamp.append((st.st_mtime_ns, st.st_size))
        except OSError:
            stamp.append(None)
    return tuple(stamp)


# --- Deeper analytics queries ------------------------------------------------

def _parse_dates(series, user=None):
    """Stored UTC instants -> naive datetimes on the rider's own clock (their local day is what they mean)."""
    return to_local_naive(series, (user or {}).get('timezone'))

def _date_filter(start, end, user=None):
    """SQL for rides in [start, end] (inclusive LOCAL dates of the rider): their day boundaries in UTC."""
    first, after = local_day_bounds_utc(start, end, (user or {}).get('timezone'))
    sql, params = "", []
    if first is not None:
        sql += " AND r.date_time >= ?"
        params.append(first)
    if after is not None:
        sql += " AND r.date_time < ?"
        params.append(after)
    return sql, params
