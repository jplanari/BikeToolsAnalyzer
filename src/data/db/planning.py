"""Planned training sessions."""
import pandas as pd
from .core import connect
from .users import get_user_data

def get_planned_sessions(user_name, after=None, structured=None):
    """
    Planned sessions as DataFrame(id, date, tss, note, blocks), optionally only after a date.
    structured: None = all, False = quick TSS entries only, True = interval-built sessions only.
    blocks is the JSON text of the interval blocks (None for quick entries).
    """
    cols = ['id', 'date', 'tss', 'note', 'blocks']
    user = get_user_data(user_name)
    if not user:
        return pd.DataFrame(columns=cols)
    sql = "SELECT id, date, tss, note, blocks FROM planned_sessions WHERE user_id = ?"
    params = [user['id']]
    if after is not None:
        sql += " AND date > ?"
        params.append(pd.Timestamp(after).strftime("%Y-%m-%d"))
    if structured is not None:
        sql += " AND blocks IS NOT NULL" if structured else " AND blocks IS NULL"
    with connect() as conn:
        df = pd.read_sql_query(sql + " ORDER BY date, id", conn, params=params)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df[cols]

def add_structured_session(user_name, date, blocks, tss, note=None):
    """Stores one interval-built session (blocks: list of dicts, scored to `tss`). Returns its id."""
    import json
    user = get_user_data(user_name)
    if not user:
        return None
    with connect() as conn:
        cur = conn.execute("INSERT INTO planned_sessions (user_id, date, tss, note, blocks) VALUES (?, ?, ?, ?, ?)",
                           (user['id'], pd.Timestamp(date).strftime("%Y-%m-%d"), float(tss),
                            note or None, json.dumps(blocks)))
        new_id = cur.lastrowid
    return new_id

def delete_planned_session(user_name, session_id):
    user = get_user_data(user_name)
    if not user:
        return False
    with connect() as conn:
        cur = conn.execute("DELETE FROM planned_sessions WHERE id = ? AND user_id = ?", (int(session_id), user['id']))
    return cur.rowcount > 0

def replace_planned_sessions(user_name, sessions, after):
    """
    Replaces the user's quick-entry planned sessions dated after `after` (e.g. today)
    with `sessions` (rows with date, tss, note). Past plans and interval-built sessions
    (see add_structured_session) are kept.
    Rows without a date or with a negative/missing TSS are ignored.
    Returns the number of sessions stored.
    """
    user = get_user_data(user_name)
    if not user:
        return 0
    cutoff = pd.Timestamp(after).normalize()
    rows = []
    for r in (sessions.to_dict('records') if hasattr(sessions, 'to_dict') else sessions):
        d = pd.to_datetime(r.get('date'), errors='coerce')
        t = pd.to_numeric(r.get('tss'), errors='coerce')
        if pd.isna(d) or pd.isna(t) or t < 0 or d.normalize() <= cutoff:
            continue
        note = r.get('note')
        rows.append((user['id'], d.strftime("%Y-%m-%d"), float(t),
                     None if note is None or (isinstance(note, float) and pd.isna(note)) else str(note)))
    with connect() as conn:
        conn.execute("DELETE FROM planned_sessions WHERE user_id = ? AND date > ? AND blocks IS NULL",
                     (user['id'], cutoff.strftime("%Y-%m-%d")))
        conn.executemany("INSERT INTO planned_sessions (user_id, date, tss, note) VALUES (?, ?, ?, ?)", rows)
    return len(rows)
