import sqlite3
import pytest
from unittest.mock import patch
from src.data import db

class UnclosableConnection:
    """
    A wrapper around the sqlite3 connection that ignores .close() calls.
    It delegates all attribute access AND assignment to the real connection.
    """
    def __init__(self, conn):
        # Use object.__setattr__ to avoid triggering our own __setattr__ logic
        object.__setattr__(self, 'real_conn', conn)
    
    def close(self):
        # Do nothing!
        pass
    
    def __getattr__(self, name):
        # Delegate reading attributes (cursor, commit, etc.) to the real connection
        return getattr(self.real_conn, name)

    def __setattr__(self, name, value):
        # Delegate setting attributes (like row_factory) to the real connection
        if name == 'real_conn':
            object.__setattr__(self, name, value)
        else:
            setattr(self.real_conn, name, value)

@pytest.fixture
def mock_db_cursor():
    # 1. Create the real in-memory connection
    real_conn = sqlite3.connect(":memory:")
    
    # 2. Initialize the schema
    c = real_conn.cursor()
    c.execute('''CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT UNIQUE, ftp INTEGER, lthr INTEGER, weight REAL)''')
    
    c.execute('''CREATE TABLE rides (
        id INTEGER PRIMARY KEY, user_id INTEGER, filename TEXT, file_hash TEXT UNIQUE, 
        file_content BLOB, date_time TEXT, distance_km REAL, elevation_m REAL, 
        avg_speed_kph REAL, norm_power INTEGER, avg_power INTEGER, 
        tss INTEGER, if_factor REAL, avg_hr INTEGER
    )''')
    
    c.execute('''CREATE TABLE power_records (id INTEGER PRIMARY KEY, user_id INTEGER, ride_id INTEGER, duration_sec INTEGER, watts REAL)''')
    
    # 3. Create our wrapper
    wrapped_conn = UnclosableConnection(real_conn)
    
    # 4. Patch sqlite3.connect to return our WRAPPER
    with patch('sqlite3.connect', return_value=wrapped_conn):
        yield c
    
    # 5. Cleanup
    real_conn.close()

def test_save_and_get_user(mock_db_cursor):
    # save_user calls wrapped_conn.close(), which does nothing
    success, msg = db.save_user("TestUser", 250, 170, 75.0)
    assert success is True
    
    # get_user_data sets row_factory on wrapped_conn, which now correctly forwards to real_conn
    user = db.get_user_data("TestUser")
    assert user is not None
    assert user['ftp'] == 250
    assert user['weight'] == 75.0
