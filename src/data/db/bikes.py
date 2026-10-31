"""Bikes by components."""
import hashlib
import sqlite3
from .core import connect
from .users import get_user_data

# --- Bikes -------------------------------------------------------------------------------------------------------
# A bike is four parts (frame, wheelset, groupset, tyres) plus whatever they leave out; the parts add up to the bike's
# weight and the tyres give the rolling resistance. CdA is not a bike property: it is computed from each ride.

_BIKE_PARTS = ('frame', 'wheelset', 'groupset', 'tyres')
_BIKE_FIELDS = ([f"{p}_name" for p in _BIKE_PARTS] + [f"{p}_kg" for p in _BIKE_PARTS] + ['extras_kg', 'crr'])


def _bike_values(parts):
    return [(parts.get(f) or None) if f.endswith('_name') else (None if parts.get(f) is None else float(parts[f]))
            for f in _BIKE_FIELDS]


def save_bike(user_name, name, parts, make_default=False):
    """parts: {'frame_name', 'frame_kg', ..., 'extras_kg', 'crr'}. Returns (ok, message)."""
    user = get_user_data(user_name)
    if not user:
        return False, "User not found."
    if not (name or "").strip():
        return False, "Give the bike a name."
    try:
        with connect() as conn:
            first = conn.execute("SELECT COUNT(*) FROM bikes WHERE user_id = ?", (user['id'],)).fetchone()[0] == 0
            cols = ['user_id', 'name'] + _BIKE_FIELDS + ['is_default']
            conn.execute(f"INSERT INTO bikes ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                         [user['id'], name.strip()] + _bike_values(parts) + [int(first)])
            bike_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    except sqlite3.IntegrityError:
        return False, "You already have a bike with that name."
    if make_default:
        set_default_bike(user_name, bike_id)
    return True, "Bike saved."


def update_bike(bike_id, name, parts):
    if not (name or "").strip():
        return False, "Give the bike a name."
    try:
        with connect() as conn:
            sets = ', '.join(f"{c} = ?" for c in ['name'] + _BIKE_FIELDS)
            conn.execute(f"UPDATE bikes SET {sets} WHERE id = ?", [name.strip()] + _bike_values(parts) + [bike_id])
        return True, "Bike updated."
    except sqlite3.IntegrityError:
        return False, "You already have a bike with that name."


def get_bikes(user_name):
    """The rider's bikes, default first, then by name."""
    user = get_user_data(user_name)
    if not user:
        return []
    with connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM bikes WHERE user_id = ? ORDER BY is_default DESC, name COLLATE NOCASE",
                            (user['id'],)).fetchall()
    return [dict(r) for r in rows]


def delete_bike(bike_id):
    """Removes the bike; rides done on it go back to the generic bike. The default moves to another bike if any."""
    with connect() as conn:
        row = conn.execute("SELECT user_id, is_default FROM bikes WHERE id = ?", (bike_id,)).fetchone()
        conn.execute("UPDATE rides SET bike_id = NULL WHERE bike_id = ?", (bike_id,))
        conn.execute("DELETE FROM bikes WHERE id = ?", (bike_id,))
        if row and row[1]:
            nxt = conn.execute("SELECT id FROM bikes WHERE user_id = ? ORDER BY name COLLATE NOCASE LIMIT 1",
                               (row[0],)).fetchone()
            if nxt:
                conn.execute("UPDATE bikes SET is_default = 1 WHERE id = ?", (nxt[0],))


def set_default_bike(user_name, bike_id):
    """The bike new rides start with."""
    user = get_user_data(user_name)
    if not user:
        return
    with connect() as conn:
        conn.execute("UPDATE bikes SET is_default = 0 WHERE user_id = ?", (user['id'],))
        conn.execute("UPDATE bikes SET is_default = 1 WHERE id = ? AND user_id = ?", (bike_id, user['id']))


def get_ride_bike(user_name, file_content):
    """(ride_id, bike_id) of an already saved ride, found by its file; None when it has not been saved."""
    user = get_user_data(user_name)
    if not user:
        return None
    with connect() as conn:
        row = conn.execute("SELECT id, bike_id FROM rides WHERE user_id = ? AND file_hash = ?",
                           (user['id'], hashlib.md5(file_content).hexdigest())).fetchone()
    return (row[0], row[1]) if row else None


def set_ride_bike(ride_id, bike_id):
    """Records the bike a saved ride was done on: a bike id, 0 for the generic bike, None for "not chosen"."""
    with connect() as conn:
        conn.execute("UPDATE rides SET bike_id = ? WHERE id = ?", (bike_id, ride_id))
