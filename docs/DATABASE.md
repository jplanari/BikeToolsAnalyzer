# Database

BikeTools keeps its data in **SQLite** (a single file, nothing to install) or, for a server shared by many
riders, in **PostgreSQL**. The same code runs on both.

## Choosing

| | SQLite (default) | PostgreSQL |
|---|---|---|
| Setting | `BIKETOOLS_DB=/path/to/file.db` (optional) | `BIKETOOLS_DATABASE_URL=postgresql://user:password@host:5432/dbname` |
| Extra install | none | `pip install -r requirements-server.txt` |
| For | the desktop app, one computer | a server with several riders at once |
| Backups | automatic copies in `backups/` before risky changes | use `pg_dump` (the app does not back up PostgreSQL itself) |

If `BIKETOOLS_DATABASE_URL` is set, it wins and `BIKETOOLS_DB` is ignored.

The tables are created and migrated on start-up in both cases (`init_db()`); the schema version is stored in
`PRAGMA user_version` (SQLite) or in the table `biketools_schema` (PostgreSQL, guarded by an advisory lock so two
server processes never migrate at once). A PostgreSQL database starts with the current schema, so the old SQLite-only
migrations (rebuilding `rides`) do nothing there.

**Encryption key.** OAuth tokens and passwords are encrypted before they are stored. On SQLite the key file sits next
to the database; on a server set `BIKETOOLS_SECRET_KEY` (a Fernet key) so the key is not in the database's host
folder. Keep it out of database backups, or the encryption protects nothing.

## Writing SQL

All SQL lives in `src/data/db/` and is written once, in SQLite's style with `?` placeholders.
`src/data/db/dialect.py` makes it run on PostgreSQL: it changes the placeholders, the column types in
`CREATE`/`ALTER` (`INTEGER PRIMARY KEY AUTOINCREMENT` → `SERIAL`, `BLOB` → `BYTEA`, `REAL` → `DOUBLE PRECISION`), adds
`RETURNING id` to inserts so `cursor.lastrowid` works, and turns booleans and numpy numbers into plain values.

Keep to what both understand:

- Upserts: `INSERT ... ON CONFLICT (key) DO UPDATE SET ... = excluded....` (not `INSERT OR REPLACE`).
- Case-insensitive ordering: `ORDER BY lower(name)` (not `COLLATE NOCASE`).
- The id of a new row: `cursor.lastrowid` (not `last_insert_rowid()`).
- No `PRAGMA`, no `sqlite_master`: use `dialect.table_exists(conn, name)` and `dialect.columns(conn, table)`.
- Catch constraint errors with `except integrity_errors():` (from `dialect`), not `sqlite3.IntegrityError`.
- Every selected column in a `GROUP BY` query must be grouped or aggregated; PostgreSQL rejects the rest.
- A new table with an integer `id` primary key must be added to `SERIAL_TABLES` in `dialect.py`.
- Row access by name: `conn.row_factory = Row` (from `dialect`).

## Checking

- `pytest` runs everything on SQLite. CI runs it a second time on PostgreSQL (the `postgres` job); a test about the
  SQLite file itself (backups, PRAGMAs, file locking) is marked `@pytest.mark.sqlite_only` and skipped there.
- To try it locally: `BIKETOOLS_DATABASE_URL=postgresql://... pytest` (the database is emptied for each test: use a
  throwaway one).
- `python tools/pg_check.py "<psql connection>" tests/` replays every statement the tests make through
  PostgreSQL's parser (it needs only `psql`), which catches syntax and column mistakes without a Python driver.

## Not done yet (needed for a public server)

- Connection pooling (each call opens a connection).
- Per-user isolation tests at the data-layer level, accounts with e-mail verification, retention and deletion tooling.
- Moving an existing SQLite database into PostgreSQL (a one-off import script).
