import sqlite3
import pandas as pd
import hashlib

DB_FILE = "bikethools.db"

def init_db():
    """Initializes the database with users and rides tables."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Users Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            ftp INTEGER DEFAULT 200,
            lthr INTEGER DEFAULT 170,
            weight REAL DEFAULT 70.0
        )
    ''')

    # Rides Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS rides (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT,
            file_hash TEXT UNIQUE,
            file_content BLOB,
            date_time TEXT,
            distance_km REAL,
            elevation_m REAL,
            avg_speed_kph REAL,
            norm_power INTEGER,
            avg_power INTEGER,
            avg_hr INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()

# --- USER FUNCTIONS ---

def save_user(name, ftp, lthr, weight):
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
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT name FROM users")
    users = [row[0] for row in c.fetchall()]
    conn.close()
    return users

def get_user_data(name):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE name = ?", (name,))
    row = c.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def delete_user(name):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE name = ?", (name,))
    conn.commit()
    conn.close()

# --- RIDE FUNCTIONS ---

def save_ride(user_name, filename, file_bytes, stats):
    user = get_user_data(user_name)
    if not user:
        return False, "User not found."
   
    file_hash = hashlib.md5(file_bytes).hexdigest()
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO rides (
                user_id, filename, file_hash, file_content, date_time, 
                distance_km, elevation_m, avg_speed_kph, 
                norm_power, avg_power, avg_hr
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user['id'], filename, file_hash, file_bytes, str(stats['date_time']),
            stats['dist_km'], stats['ele_m'], stats['speed'],
            stats['np'], stats['avg_p'], stats['avg_hr']
        ))
        conn.commit()
        return True, "Ride saved to history!"
    except sqlite3.IntegrityError:
        return False, "Ride already exists (duplicate file)."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def get_user_rides(user_name):
    user = get_user_data(user_name)
    if not user:
        return []
        
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('''
        SELECT * FROM rides 
        WHERE user_id = ? 
        ORDER BY date_time DESC
    ''', (user['id'],))
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_ride_file(ride_id):
    """Retrieves the file content for a specific ride."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # FIXED: Select BOTH filename and file_content
    c.execute("SELECT filename, file_content FROM rides WHERE id = ?", (ride_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return row[0], row[1] # Now this works (filename, blob)
    return None, None
