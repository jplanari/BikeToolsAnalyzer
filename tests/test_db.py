import sqlite3
import pandas as pd
import pytest

from src.data import db


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """A real, empty database file built with the app's own schema."""
    path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_FILE", str(path))
    db.init_db()
    return path


def _ride_df(start="2026-10-01T08:00:00Z", n=3600):
    return pd.DataFrame({
        'time': pd.date_range(start, periods=n, freq='1s'),
        'power': [200.0] * n,
        'hr': [140.0] * n,
    })


def test_save_and_get_user(temp_db):
    success, msg = db.save_user("TestUser", 250, 170, 75.0)
    assert success is True

    user = db.get_user_data("TestUser")
    assert user is not None
    assert user['ftp'] == 250
    assert user['weight'] == 75.0
    assert user['height'] == 175.0  # default


def test_save_ride_stores_date_distance_elevation(temp_db):
    """Regression: metadata keys used to mismatch, storing NULL date and 0 km / 0 m."""
    db.save_user("Rider", 250, 170, 70.0, 180.0)
    meta = db.build_ride_metadata(_ride_df(), dist_m=42_195, ele_m=812,
                                  speed_kph=30.0, np_val=210.4, tss=71.2, if_val=0.84)
    ok, msg = db.save_ride("Rider", "ride.gpx", b"<gpx/>", meta, {60: 300.0, 300: 260.0})
    assert ok, msg

    conn = sqlite3.connect(temp_db)
    row = conn.execute("SELECT date_time, distance_km, elevation_m, norm_power, tss, avg_power, avg_hr "
                       "FROM rides").fetchone()
    pr_dates = {r[0] for r in conn.execute("SELECT date_time FROM power_records")}
    conn.close()

    assert row[0] == "2026-10-01T08:00:00+00:00"
    assert row[1] == pytest.approx(42.195)
    assert row[2] == pytest.approx(812)
    assert row[3:] == (210, 71, 200, 140)
    assert pr_dates == {"2026-10-01T08:00:00+00:00"}

    history = db.get_user_tss_history("Rider")
    assert history == [{'date': "2026-10-01T08:00:00", 'tss': 71}]


def test_get_ride_file(temp_db):
    db.save_user("Rider", 250, 170, 70.0)
    meta = db.build_ride_metadata(_ride_df(), 1000, 10, 25, 200, 50, 0.8)
    db.save_ride("Rider", "a.gpx", b"payload", meta, {})
    ride_id = db.get_user_rides("Rider")[0]['id']

    assert db.get_ride_file(ride_id) == ("a.gpx", b"payload")
    assert db.get_ride_file(9999) == (None, None)


def test_w_prime_is_20_kj_unless_the_rider_gives_theirs(temp_db):
    db.save_user("Default")
    db.save_user("Given", w_prime=27500)
    assert db.get_user_data("Default")['w_prime'] == 20000
    assert db.get_user_data("Given")['w_prime'] == 27500
    db.set_user_w_prime("Default", 23400)
    assert db.get_user_data("Default")['w_prime'] == 23400
