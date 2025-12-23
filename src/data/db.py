import sqlite3
import pandas as pd
import hashlib

DB_FILE = "bikethools.db"

def migrate_db():
    """
    Adds new columns (cp, w_prime) to the users table if they don't exist.
    Run this once safely; it catches errors if columns already exist.
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    try:
        c.execute("ALTER TABLE users ADD COLUMN cp INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass # Column likely exists already

    try:
        c.execute("ALTER TABLE users ADD COLUMN w_prime REAL DEFAULT 20000")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()

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
            weight REAL DEFAULT 70.0,
            height REAL DEFAULT 175.0,
            cp INTEGER DEFAULT 0,
            w_prime REAL DEFAULT 20000
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

    # Power Records Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS power_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            ride_id INTEGER,
            duration_sec INTEGER,
            watts REAL,
            date_time TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(ride_id) REFERENCES rides(id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()
    
    # Run migration to ensure existing databases get the new columns
    migrate_db()

def save_user(name, ftp, lthr, weight, height):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (name, ftp, lthr, weight, height) VALUES (?, ?, ?, ?, ?)", 
                  (name, ftp, lthr, weight, height))
        conn.commit()
        return True, "User saved successfully!"
    except sqlite3.IntegrityError:
        return False, "User already exists."
    except Exception as e:
        return False, f"Database Error: {e}"
    finally:
        conn.close()

def update_user_physics(user_name, cp, w_prime, update_ftp=False):
    """
    Updates the user's physiological metrics.
    Optionally updates FTP to match CP.
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    if update_ftp:
        c.execute('''
            UPDATE users 
            SET cp = ?, w_prime = ?, ftp = ? 
            WHERE name = ?
        ''', (cp, w_prime, cp, user_name))
    else:
        c.execute('''
            UPDATE users 
            SET cp = ?, w_prime = ? 
            WHERE name = ?
        ''', (cp, w_prime, user_name))
        
    conn.commit()
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

def save_ride(user_name, filename, file_content, metadata, power_curve_dict):
    user = get_user_data(user_name)
    if not user:
        return False, "User not found."

    file_hash = hashlib.md5(file_content).hexdigest()
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    try:
        # 1. Insert Ride Metadata
        c.execute('''
            INSERT INTO rides (
                user_id, filename, file_hash, file_content, date_time, 
                distance_km, elevation_m, avg_speed_kph, 
                norm_power, avg_power, tss, if_factor, avg_hr
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user['id'], filename, file_hash, file_content, metadata.get('date'),
            metadata.get('dist', 0), metadata.get('ele', 0), metadata.get('speed', 0),
            metadata.get('np', 0), metadata.get('avg_p', 0), metadata.get('tss', 0),
            metadata.get('if', 0), metadata.get('avg_hr', 0)
        ))
        
        ride_id = c.lastrowid
        
        # 2. Insert Power Records (Curve)
        if power_curve_dict:
            records = []
            for duration, watts in power_curve_dict.items():
                records.append((user['id'], ride_id, duration, watts, metadata.get('date')))
            
            c.executemany('''
                INSERT INTO power_records (user_id, ride_id, duration_sec, watts, date_time)
                VALUES (?, ?, ?, ?, ?)
            ''', records)
            
        conn.commit()
        return True, "Ride saved successfully!"
        
    except sqlite3.IntegrityError:
        return False, "Ride already exists (duplicate file)."
        
    except Exception as e:
        return False, f"Database Error: {e}"
        
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
        SELECT id, filename, date_time, distance_km, elevation_m, norm_power, tss 
        FROM rides 
        WHERE user_id = ? 
        ORDER BY date_time DESC
    ''', (user['id'],))
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_ride_file(ride_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT filename, file_content FROM rides WHERE id = ?", (ride_id, ())) # Fixed tuple
    row = c.fetchone()
    conn.close()
    if row:
        return row[0], row[1]
    return None, None

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
    """
    Fetches date and TSS for all rides, mimicking get_user_rides logic.
    Does NOT filter by TSS value in SQL to ensure we get raw data first.
    """
    user = get_user_data(user_name)
    if not user:
        return []
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Just select the columns we need
    c.execute('''
        SELECT date_time, tss 
        FROM rides 
        WHERE user_id = ? 
        ORDER BY date_time ASC
    ''', (user['id'],))
    
    rows = c.fetchall()
    conn.close()
    
    # Return raw list of dicts
    return [{'date': row[0], 'tss': row[1]} for row in rows]
