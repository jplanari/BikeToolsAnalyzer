"""Riders and their thresholds."""
import sqlite3
from .core import backup_db, connect

DEFAULT_FTP, DEFAULT_LTHR = 200, 170
DEFAULT_W_PRIME = 20000          # joules, when the rider has not given theirs

def save_user(name, ftp=None, lthr=None, weight=70.0, height=175.0, w_prime=None):
    """
    ftp / lthr None: stored as the defaults but flagged to be estimated from the rider's rides.
    w_prime (joules) None: DEFAULT_W_PRIME; a ride that uses more than it raises the figure (see the Power section).
    """
    try:
        with connect() as conn:
            conn.execute("""INSERT INTO users (name, ftp, lthr, weight, height, ftp_auto, lthr_auto, w_prime)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                         (name, DEFAULT_FTP if ftp is None else ftp, DEFAULT_LTHR if lthr is None else lthr,
                          weight, height, int(ftp is None), int(lthr is None),
                          DEFAULT_W_PRIME if not w_prime else float(w_prime)))
        return True, "User saved successfully!"
    except sqlite3.IntegrityError:
        return False, "User already exists."
    except Exception as e:
        return False, f"Database Error: {e}"

def set_user_thresholds(user_name, ftp=None, lthr=None, auto_ftp=None, auto_lthr=None):
    """
    Sets FTP and/or LTHR by hand (which stops them following the data), or switches them back to
    "estimate from my rides" with auto_ftp / auto_lthr=True (the value stays until the next estimate).
    """
    with connect() as conn:
        if ftp is not None:
            conn.execute("UPDATE users SET ftp = ?, ftp_auto = 0 WHERE name = ?", (int(round(ftp)), user_name))
        if lthr is not None:
            conn.execute("UPDATE users SET lthr = ?, lthr_auto = 0 WHERE name = ?", (int(round(lthr)), user_name))
        if auto_ftp:
            conn.execute("UPDATE users SET ftp_auto = 1 WHERE name = ?", (user_name,))
        if auto_lthr:
            conn.execute("UPDATE users SET lthr_auto = 1 WHERE name = ?", (user_name,))

def set_user_timezone(user_name, timezone):
    """The rider's IANA time zone ('Europe/Madrid'); None or an unknown name goes back to this computer's zone."""
    from src.analysis.dates import valid_timezone
    with connect() as conn:
        conn.execute("UPDATE users SET timezone = ? WHERE name = ?", (valid_timezone(timezone), user_name))

def set_user_w_prime(user_name, w_prime):
    """Sets the rider's W' (joules)."""
    with connect() as conn:
        conn.execute("UPDATE users SET w_prime = ? WHERE name = ?", (float(w_prime), user_name))

def set_auto_thresholds(user_name, ftp=None, lthr=None):
    """Stores an estimate without turning the 'estimated' flag off."""
    with connect() as conn:
        if ftp is not None:
            conn.execute("UPDATE users SET ftp = ? WHERE name = ?", (int(round(ftp)), user_name))
        if lthr is not None:
            conn.execute("UPDATE users SET lthr = ? WHERE name = ?", (int(round(lthr)), user_name))

def update_user_ftp(user_name, ftp):
    """Sets the FTP only. Rides already saved keep the FTP they were scored with."""
    with connect() as conn:
        conn.execute("UPDATE users SET ftp = ?, ftp_auto = 0 WHERE name = ?", (int(round(ftp)), user_name))

def update_user_physics(user_name, cp, w_prime, update_ftp=False):
    """
    Updates the user's physiological metrics.
    Optionally updates FTP to match CP.
    """
    with connect() as conn:
        c = conn.cursor()
    
        if update_ftp:
            c.execute('''
                UPDATE users 
                SET cp = ?, w_prime = ?, ftp = ?, ftp_auto = 0 
                WHERE name = ?
            ''', (cp, w_prime, cp, user_name))
        else:
            c.execute('''
                UPDATE users 
                SET cp = ?, w_prime = ? 
                WHERE name = ?
            ''', (cp, w_prime, user_name))

def get_all_users():
    with connect() as conn:
        c = conn.cursor()
        c.execute("SELECT name FROM users")
        users = [row[0] for row in c.fetchall()]
    return users

def get_user_data(name):
    with connect() as conn:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE name = ?", (name,))
        row = c.fetchone()
    if row:
        return dict(row)
    return None

def delete_user(name):
    """Deletes the rider and, by cascade, everything of theirs. A backup is taken first: this cannot be undone."""
    if get_user_data(name):
        backup_db("before-delete-user")
    with connect() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM users WHERE name = ?", (name,))

def update_user_crank(user_name, crank_mm):
    with connect() as conn:
        conn.execute("UPDATE users SET crank_mm = ? WHERE name = ?", (float(crank_mm), user_name))
