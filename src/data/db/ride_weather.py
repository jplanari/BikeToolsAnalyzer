"""Weather kept with a ride (see src/data/weather.py for what the payload holds)."""
import json
from datetime import datetime, timezone
from .core import connect


def save_ride_weather(user_name, ride_id, payload):
    """Stores (or replaces) the weather payload of one of the rider's saved rides (any other ride: nothing happens)."""
    with connect() as conn:
        if conn.execute("SELECT 1 FROM rides WHERE id = ? AND user_id = (SELECT id FROM users WHERE name = ?)",
                        (ride_id, user_name)).fetchone() is None:
            return
        conn.execute("""INSERT INTO ride_weather (ride_id, payload, fetched_at) VALUES (?, ?, ?)
                        ON CONFLICT (ride_id) DO UPDATE SET payload = excluded.payload, fetched_at = excluded.fetched_at""",
                     (ride_id, json.dumps(payload), datetime.now(timezone.utc).isoformat(timespec='seconds')))


def get_ride_weather(user_name, ride_id):
    """The stored weather payload of one of the rider's rides, or None."""
    with connect() as conn:
        row = conn.execute("""SELECT w.payload FROM ride_weather w JOIN rides r ON r.id = w.ride_id
                              WHERE w.ride_id = ? AND r.user_id = (SELECT id FROM users WHERE name = ?)""",
                           (ride_id, user_name)).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except ValueError:
        return None
