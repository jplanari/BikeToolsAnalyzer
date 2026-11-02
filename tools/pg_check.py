"""
Checks, against a real PostgreSQL server, every SQL statement the test suite makes.

The tests run on SQLite, recording each statement. The CREATE / ALTER ones are then translated the way
src/data/db/dialect.py does and applied to an empty database, and every other statement is PREPAREd there
(PostgreSQL parses it and resolves its tables, columns and functions, without running it). It needs only `psql`.

    python tools/pg_check.py "host=/tmp/pgd port=54329 user=postgres dbname=bt"
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data.db.dialect import to_pg_ddl                     # noqa: E402

PLUGIN = '''
import os, re, sqlite3
LOG = os.environ["SQL_LOG"]
def _log(sql):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(" ".join(re.sub(r"--[^\\n]*", "", sql).split()) + "\\n@@\\n")
class Cur(sqlite3.Cursor):
    def execute(self, sql, *a):
        _log(sql); return super().execute(sql, *a)
    def executemany(self, sql, *a):
        _log(sql); return super().executemany(sql, *a)
class Conn(sqlite3.Connection):
    def cursor(self, factory=Cur): return super().cursor(factory)
    def execute(self, sql, *a):
        _log(sql); return super().execute(sql, *a)
    def executemany(self, sql, *a):
        _log(sql); return super().executemany(sql, *a)
_connect = sqlite3.connect
def connect(*a, **k):
    k.setdefault("factory", Conn); return _connect(*a, **k)
sqlite3.connect = connect
'''
SKIP = re.compile(r"^\s*(PRAGMA|BEGIN|COMMIT|ROLLBACK|VACUUM|SAVEPOINT|RELEASE)\b|sqlite_|last_insert_rowid|pragma_|no_such_table|x.00.|^\s*DROP\b", re.I)


def psql(dsn, sql, **kw):
    return subprocess.run(["psql", dsn, "-v", "ON_ERROR_STOP=1", "-q", "-At", "-c", sql], capture_output=True, text=True, **kw)


def main(dsn, pytest_args):
    tmp = tempfile.mkdtemp()
    open(os.path.join(tmp, "sqllog_plugin.py"), "w").write(PLUGIN)
    log = os.path.join(tmp, "sql.log")
    env = dict(os.environ, SQL_LOG=log, PYTHONPATH=tmp + os.pathsep + os.environ.get("PYTHONPATH", ""))
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "sqllog_plugin", "-p", "no:cacheprovider", *pytest_args],
                   cwd=ROOT, env=env, check=False)
    statements, seen = [], set()
    for chunk in open(log, encoding="utf-8").read().split("\n@@\n"):
        chunk = chunk.strip()
        if chunk and chunk not in seen:
            seen.add(chunk)
            statements.append(chunk)
    ddl = [s for s in statements if re.match(r"^(CREATE|ALTER)\b", s, re.I) and not SKIP.search(s)]
    psql(dsn, "DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    psql(dsn, "CREATE TABLE biketools_schema (version INTEGER NOT NULL); CREATE SEQUENCE biketools_data_version;")
    failures = 0
    # CREATE TABLE IF NOT EXISTS statements first (they are idempotent), then ALTERs and indexes, until stable
    pending = [to_pg_ddl(s) for s in ddl]
    for _ in range(4):
        left = []
        for s in pending:
            r = psql(dsn, s)
            if r.returncode and "already exists" not in r.stderr:
                left.append(s)
                last = (s, r.stderr)
        if not left or len(left) == len(pending):
            pending = left
            break
        pending = left
    for s in pending:
        failures += 1
        print("DDL FAILED:", s[:200], "\n   ", psql(dsn, s).stderr.strip())
    checked = 0
    for s in statements:
        if re.match(r"^(CREATE|ALTER)\b", s, re.I) or SKIP.search(s):
            continue
        n = [0]

        def num(_):
            n[0] += 1
            return f"${n[0]}"
        q = re.sub(r"\?", num, s)
        if re.match(r"^INSERT\s+INTO\s+(users|rides|power_records|bikes|planned_sessions)\b", q, re.I) and "returning" not in q.lower():
            q += " RETURNING id"
        r = psql(dsn, f"PREPARE s AS {q}; DEALLOCATE s;")
        checked += 1
        if r.returncode:
            failures += 1
            print("FAILED:", q[:240], "\n   ", r.stderr.strip().splitlines()[0])
    print(f"{len(ddl)} DDL statements, {checked} queries checked, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2:]))
