import sqlite3
import pandas as pd
import hashlib

DB_FILE = "bikethools.db"

def init_db():
    """Initializes the database with users, rides, and power_records tables."""
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
            tss INTEGER,
            if_factor REAL,
            avg_hr INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # NEW: Power Records Table
    # Stores specific duration bests (e.g., 5s, 60s, 1200s) for each ride
    c.execute('''
        CREATE TABLE IF NOT EXISTS power_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            ride_id INTEGER,
            duration_sec INTEGER,
            watts REAL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(ride_id) REFERENCES rides(id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()

# --- USER FUNCTIONS ---
# (Keep save_user, get_all_users, get_user_data, delete_user as they are)
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

# UPDATED: Now accepts `power_curve_dict`
def save_ride(user_name, filename, file_bytes, stats, power_curve_dict=None):
    user = get_user_data(user_name)
    if not user:
        return False, "User not found."
   
    file_hash = hashlib.md5(file_bytes).hexdigest()
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        # 1. Insert Ride
        c.execute('''
            INSERT INTO rides (
                user_id, filename, file_hash, file_content, date_time, 
                distance_km, elevation_m, avg_speed_kph, 
                norm_power, avg_power, avg_hr, tss, if_factor
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user['id'], filename, file_hash, file_bytes, str(stats['date_time']),
            stats['dist_km'], stats['ele_m'], stats['speed'],
            stats['np'], stats['avg_p'], stats['avg_hr'], stats.get('tss',0), stats.get('if_factor', stats.get('if',0.0))
        ))
        
        ride_id = c.lastrowid

        # 2. Insert Power Records (if power data existed)
        if power_curve_dict:
            records_data = []
            for duration, watts in power_curve_dict.items():
                # We filter out NaNs or 0s to keep DB clean
                if pd.notna(watts) and watts > 0:
                    records_data.append((user['id'], ride_id, int(duration), float(watts)))
            
            if records_data:
                c.executemany('''
                    INSERT INTO power_records (user_id, ride_id, duration_sec, watts)
                    VALUES (?, ?, ?, ?)
                ''', records_data)

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
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT filename, file_content FROM rides WHERE id = ?", (ride_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return row[0], row[1]
    return None, None

# NEW FUNCTION: Get User's All-Time Best Power Curve
def get_user_best_power(user_name):
    user = get_user_data(user_name)
    if not user:
        return {}
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Query: Find the maximum watts for each duration across ALL rides for this user
    c.execute('''
        SELECT duration_sec, MAX(watts) 
        FROM power_records 
        WHERE user_id = ? 
        GROUP BY duration_sec
        ORDER BY duration_sec ASC
    ''', (user['id'],))
    
    rows = c.fetchall()
    conn.close()
    
    # Convert to dict {duration: watts}
    return {row[0]: row[1] for row in rows}

def get_user_tss_history(user_name):
    user = get_user_data(user_name)
    if not user:
        return []

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute('''
        SELECT date_time, tss
        FROM rides
        WHERE user_id = ?
        ORDER BY date_time ASC
    ''', (user['id'],))

    rows = c.fetchall()
    conn.close()

    return [{'date': row[0], 'tss': row[1]} for row in rows]
