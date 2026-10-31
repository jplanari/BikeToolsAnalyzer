"""Weather kept with a ride (see src/data/weather.py for what the payload holds)."""
import json
from datetime import datetime, timezone
from .core import connect


def save_ride_weather(ride_id, payload):
    """Stores (or replaces) the weather payload of a saved ride."""
    with connect() as conn:
        conn.execute("INSERT OR REPLACE INTO ride_weather (ride_id, payload, fetched_at) VALUES (?, ?, ?)",
                     (ride_id, json.dumps(payload), datetime.now(timezone.utc).isoformat(timespec='seconds')))


def get_ride_weather(ride_id):
    """The stored weather payload of a ride, or None."""
    with connect() as conn:
        row = conn.execute("SELECT payload FROM ride_weather WHERE ride_id = ?", (ride_id,)).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except ValueError:
        return None
