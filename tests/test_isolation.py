"""
One rider can never read or change another rider's data, whatever id they pass.

This is the guarantee a shared server stands on. The first test fails the day someone adds a database function that
takes a ride / bike / session id without also taking the rider, so the check cannot be forgotten.
"""
import inspect

import pytest

from src.data import db
from tests.test_bikes import PARTS
from tests.test_db import temp_db  # noqa: F401 (fixture)

ID_PARAMS = {"ride_id", "bike_id", "session_id"}
# Functions that take an id and are not tied to a rider on purpose (children of a row the caller already owns).
UNSCOPED_ON_PURPOSE = {"_insert_children", "_bike_values"}


def test_every_function_that_takes_an_id_also_takes_the_rider():
    loose = []
    for mod in (db.rides, db.bikes, db.planning, db.ride_weather, db.auth, db.weekly, db.stats, db.users):
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if fn.__module__ != mod.__name__ or name in UNSCOPED_ON_PURPOSE:
                continue
            params = set(inspect.signature(fn).parameters)
            if params & ID_PARAMS and "user_name" not in params:
                loose.append(f"{mod.__name__.split('.')[-1]}.{name}{inspect.signature(fn)}")
    assert not loose, "These take an id but not the rider: " + "; ".join(loose)


@pytest.fixture
def two(temp_db):
    """Ana and Beni, each with a ride, a bike, a planned session, fuel, weather and a power curve."""
    for name in ("Ana", "Beni"):
        db.save_user(name, 250, 170, 70.0, 175.0)
        assert db.save_bike(name, "Road", PARTS)[0]
        ok, msg = db.save_ride(name, "same.fit", b"identical bytes", {"date": "2026-10-01T08:00:00Z", "dist": 30}, {60: 300.0})
        assert ok, msg
    ids = {}
    for name in ("Ana", "Beni"):
        (ride,) = db.get_user_rides(name)
        (bike,) = db.get_bikes(name)
        db.set_ride_fuel(name, ride["id"], 100, 500)
        db.save_ride_weather(name, ride["id"], {"source": name})
        sid = db.add_structured_session(name, "2030-01-01", [{"kind": "x"}], 50.0)
        ids[name] = {"ride": ride["id"], "bike": bike["id"], "session": sid}
    return ids


def test_the_same_file_can_be_saved_by_two_riders(two):
    assert two["Ana"]["ride"] != two["Beni"]["ride"]


def test_lists_only_show_the_riders_own_things(two):
    for name, other in (("Ana", "Beni"), ("Beni", "Ana")):
        assert [r["id"] for r in db.get_user_rides(name)] == [two[name]["ride"]]
        assert [b["id"] for b in db.get_bikes(name)] == [two[name]["bike"]]
        assert [s["id"] for s in db.get_planned_sessions(name).to_dict("records")] == [two[name]["session"]]
        assert two[other]["ride"] not in [r["id"] for r in db.get_user_rides(name)]


def test_reading_another_riders_ride_by_id_gives_nothing(two):
    a, b = two["Ana"], two["Beni"]
    assert db.get_ride_file("Ana", b["ride"]) == (None, None)
    assert db.get_ride_fuel("Ana", b["ride"]) == (None, None)
    assert db.get_ride_weather("Ana", b["ride"]) is None
    assert db.get_ride_file("Ana", a["ride"])[1] == b"identical bytes"          # and their own still works
    assert db.get_ride_weather("Ana", a["ride"]) == {"source": "Ana"}


def test_changing_another_riders_ride_by_id_does_nothing(two):
    a, b = two["Ana"], two["Beni"]
    ok, _ = db.set_ride_fuel("Ana", b["ride"], 999, 999)
    assert not ok and db.get_ride_fuel("Beni", b["ride"]) == (100.0, 500.0)
    db.save_ride_weather("Ana", b["ride"], {"source": "hacked"})
    assert db.get_ride_weather("Beni", b["ride"]) == {"source": "Beni"}
    db.set_ride_hr20("Ana", b["ride"], 199)
    assert [r[0] for r in db.get_rides_without_hr20("Beni", 10)] == [b["ride"]]       # still not computed: nothing was set
    with pytest.raises(ValueError):
        db.update_ride_metrics("Ana", b["ride"], {"date": "2026-10-01T08:00:00Z", "dist": 1}, {60: 1.0})
    assert [r["distance_km"] for r in db.get_user_rides("Beni")] == [30.0]
    assert not db.delete_ride("Ana", b["ride"])[0]
    assert [r["id"] for r in db.get_user_rides("Beni")] == [b["ride"]]


def test_bikes_belong_to_one_rider(two):
    a, b = two["Ana"], two["Beni"]
    assert db.update_bike("Ana", b["bike"], "Stolen", PARTS)[0] is False
    db.delete_bike("Ana", b["bike"])
    assert [x["name"] for x in db.get_bikes("Beni")] == ["Road"]
    assert db.set_ride_bike("Ana", b["ride"], a["bike"]) is False              # not their ride
    assert db.set_ride_bike("Ana", a["ride"], b["bike"]) is False              # not their bike
    assert db.set_ride_bike("Ana", a["ride"], a["bike"]) is True
    assert db.get_ride_bike("Beni", b"identical bytes")[1] != a["bike"]


def test_planned_sessions_and_wahoo_links_belong_to_one_rider(two):
    a, b = two["Ana"], two["Beni"]
    assert db.delete_planned_session("Ana", b["session"]) is False
    assert len(db.get_planned_sessions("Beni")) == 1
    db.save_wahoo_sent("Beni", b["session"], 1, 2, "2030-01-01T07:00:00Z", "now")
    db.save_wahoo_sent("Ana", b["session"], 9, 9, "x", "x")                    # Ana cannot attach to Beni's session
    db.delete_wahoo_sent("Ana", b["session"])                                  # nor remove Beni's link
    assert db.get_wahoo_sent("Beni")[b["session"]]["workout_id"] == 2
    assert db.get_wahoo_sent("Ana") == {}


def test_records_history_and_connections_are_per_rider(two):
    db.save_strava_auth("Ana", {"athlete_id": 1, "athlete_name": "A", "access_token": "tok-a", "refresh_token": "r", "expires_at": 4_000_000_000})
    assert db.get_strava_auth("Beni") is None
    assert db.get_strava_auth("Ana")["access_token"] == "tok-a"
    assert len(db.get_power_records("Ana")) == len(db.get_power_records("Beni")) == 1
    assert db.get_imported_strava_ids("Beni") == set()


def test_deleting_a_rider_leaves_the_other_untouched(two):
    b = two["Beni"]
    db.delete_user("Ana")
    assert db.get_user_data("Ana") is None
    assert [r["id"] for r in db.get_user_rides("Beni")] == [b["ride"]]
    assert db.get_ride_file("Beni", b["ride"])[1] == b"identical bytes"
    assert db.get_ride_weather("Beni", b["ride"]) == {"source": "Beni"}
    assert [x["id"] for x in db.get_bikes("Beni")] == [b["bike"]]
