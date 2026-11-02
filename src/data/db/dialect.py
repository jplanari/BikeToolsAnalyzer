"""
SQLite or PostgreSQL, behind the same small interface.

The desktop app uses a SQLite file and needs nothing installed. A server shared by many riders sets
BIKETOOLS_DATABASE_URL=postgresql://user:password@host/dbname and uses PostgreSQL. All the SQL in this
package is written once, in SQLite's style with `?` placeholders, and runs on both:

* with SQLite `connect()` hands out a plain sqlite3 connection, untouched;
* with PostgreSQL it hands out `PgConnection`, which has the same methods (execute, cursor, executemany,
  commit, rollback, row_factory, cursor.lastrowid, rowcount) and rewrites what differs: placeholders, column
  types in CREATE / ALTER, `RETURNING id` for inserts (so lastrowid works), booleans and numpy numbers.

Differences the SQL itself must avoid are listed in docs/DATABASE.md (no INSERT OR REPLACE, no PRAGMA, no
COLLATE NOCASE, no last_insert_rowid()). Use the helpers here (`table_exists`, `columns`) instead.
"""
import os
import re
import sqlite3

try:                                              # only needed (and only installed) for a server
    import psycopg
    from psycopg.adapt import Loader
except ImportError:                               # pragma: no cover - the desktop app has no PostgreSQL driver
    psycopg = None
    Loader = object

URL_ENV = "BIKETOOLS_DATABASE_URL"

Row = sqlite3.Row                                 # `conn.row_factory = Row` works on both databases


def database_url():
    return os.environ.get(URL_ENV, "").strip()


def is_pg():
    return database_url().lower().startswith(("postgres://", "postgresql://"))


def integrity_errors():
    return (sqlite3.IntegrityError,) + ((psycopg.errors.IntegrityError,) if psycopg else ())


def operational_errors():
    return (sqlite3.OperationalError,) + ((psycopg.OperationalError,) if psycopg else ())


# --- SQL rewriting (PostgreSQL only) ------------------------------------------------------------------------------

_DDL = re.compile(r"^\s*(CREATE|ALTER)\b", re.I)
_INSERT = re.compile(r"^\s*INSERT\s+INTO\s+(\w+)", re.I)
_READ = re.compile(r"^\s*(SELECT|WITH|SHOW|EXPLAIN)\b", re.I)
SERIAL_TABLES = frozenset({"users", "rides", "power_records", "bikes", "planned_sessions"})   # tables with `id`


def to_pg_ddl(sql):
    sql = re.sub(r"INTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT", "SERIAL PRIMARY KEY", sql, flags=re.I)
    sql = re.sub(r"\bBLOB\b", "BYTEA", sql)
    return re.sub(r"\bREAL\b", "DOUBLE PRECISION", sql)


def to_pg(sql, has_params):
    """`?` -> `%s`. With parameters a literal % (LIKE 'a_%') must be doubled; without them psycopg leaves it alone."""
    if _DDL.match(sql):
        sql = to_pg_ddl(sql)
    if not has_params:
        return sql
    return sql.replace("%", "%%").replace("?", "%s")


def _param(v):
    if isinstance(v, bool):
        return int(v)
    if type(v).__module__ == "numpy" and hasattr(v, "item"):
        return v.item()
    if isinstance(v, memoryview):
        return bytes(v)
    return v


class PgRow(tuple):
    """A row that can be read by position or by column name, like sqlite3.Row."""
    def __new__(cls, values, index):
        row = super().__new__(cls, values)
        row._index = index
        return row

    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(self._index[key])
        return super().__getitem__(key)

    def keys(self):
        return list(self._index)


class _NumericAsFloat(Loader):
    """AVG() and SUM() over integers come back as NUMERIC; SQLite (and our maths) want floats."""
    def load(self, data):
        return float(bytes(data))


class PgCursor:
    def __init__(self, conn):
        self._conn = conn
        self._cur = conn._raw.cursor()
        self.lastrowid = None

    def _run(self, sql, params, many=False):
        conn = self._conn
        if not _READ.match(sql):
            conn._wrote = True
        returning = False
        m = _INSERT.match(sql)
        if m and not many and m.group(1).lower() in SERIAL_TABLES and "returning" not in sql.lower():
            sql, returning = sql.rstrip().rstrip(";") + " RETURNING id", True
        text = to_pg(sql, bool(params))
        if many:
            self._cur.executemany(text, [[_param(v) for v in row] for row in params])
        else:
            self._cur.execute(text, [_param(v) for v in params] if params else None)
        if returning:
            self.lastrowid = self._cur.fetchone()[0]
        return self

    def execute(self, sql, params=()):
        return self._run(sql, params)

    def executemany(self, sql, rows):
        return self._run(sql, list(rows), many=True)

    @property
    def rowcount(self):
        return self._cur.rowcount

    @property
    def description(self):
        return self._cur.description

    def _wrap(self, values):
        if values is None or self._conn.row_factory is None:
            return values
        index = {d.name: i for i, d in enumerate(self._cur.description)}
        return PgRow(values, index)

    def fetchone(self):
        return self._wrap(self._cur.fetchone())

    def fetchall(self):
        rows = self._cur.fetchall()
        if self._conn.row_factory is None or not rows:
            return rows
        index = {d.name: i for i, d in enumerate(self._cur.description)}
        return [PgRow(r, index) for r in rows]

    def fetchmany(self, n=1):
        return [self._wrap(r) for r in self._cur.fetchmany(n)]

    def __iter__(self):
        return iter(self.fetchall())

    def close(self):
        self._cur.close()


class PgConnection:
    """What `connect()` returns on PostgreSQL: the part of the sqlite3.Connection interface this code uses."""
    def __init__(self, raw):
        self._raw = raw
        self.row_factory = None
        self._wrote = False

    def cursor(self):
        return PgCursor(self)

    def execute(self, sql, params=()):
        return self.cursor().execute(sql, params)

    def executemany(self, sql, rows):
        return self.cursor().executemany(sql, rows)

    def commit(self):
        if self._wrote:
            self._raw.execute("SELECT nextval('biketools_data_version')")      # see data_stamp()
            self._wrote = False
        self._raw.commit()

    def rollback(self):
        self._wrote = False
        self._raw.rollback()

    def close(self):
        self._raw.close()


def connect_pg():
    if psycopg is None:
        raise RuntimeError("BIKETOOLS_DATABASE_URL is set, but the PostgreSQL driver is missing: pip install \"psycopg[binary]\"")
    raw = psycopg.connect(database_url())
    raw.adapters.register_loader("numeric", _NumericAsFloat)
    return PgConnection(raw)


# --- things SQLite and PostgreSQL ask differently -----------------------------------------------------------------

def _on_pg(x):
    return isinstance(x, (PgConnection, PgCursor))


def table_exists(conn, name):
    """`conn` may be a connection or a cursor."""
    if _on_pg(conn):
        return conn.execute("SELECT 1 FROM information_schema.tables WHERE table_schema = current_schema() "
                            "AND table_name = ?", (name,)).fetchone() is not None
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)).fetchone() is not None


def columns(conn, table):
    if _on_pg(conn):
        return {r[0] for r in conn.execute("SELECT column_name FROM information_schema.columns "
                                           "WHERE table_schema = current_schema() AND table_name = ?", (table,))}
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
