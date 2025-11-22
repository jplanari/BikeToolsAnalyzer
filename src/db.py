import sqlite3
import pandas as pd

DB_FILE = "bikethools.db"

def init_db():
    """Initializes the database with a users table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            ftp INTEGER DEFAULT 200,
            lthr INTEGER DEFAULT 170,
            weight REAL DEFAULT 70.0
        )
    ''')
    conn.commit()
    conn.close()

def save_user(name, ftp, lthr, weight):
    """Saves a new user or updates an existing one."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO users (name, ftp, lthr, weight) 
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
            ftp=excluded.ftp,
            lthr=excluded.lthr,
            weight=excluded.weight
        ''', (name, ftp, lthr, weight))
        conn.commit()
        return True, "User saved successfully."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def get_all_users():
    """Returns a list of all user names."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT name FROM users")
    users = [row[0] for row in c.fetchall()]
    conn.close()
    return users

def get_user_data(name):
    """Returns a dictionary of user data."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # Access columns by name
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE name = ?", (name,))
    row = c.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def delete_user(name):
    """Deletes a user from the database."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE name = ?", (name,))
    conn.commit()
    conn.close()
