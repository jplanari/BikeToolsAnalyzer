"""Bikes: parts add up to a weight, tyres give Crr, rides remember their bike; CdA is never a bike property."""
import pytest

from src.analysis import bikes as bk
from src.data import db
from tests.test_db import temp_db, _ride_df  # noqa: F401 (fixtures/helpers)

PARTS = {'frame_name': "Aero frame", 'frame_kg': 1.1, 'wheelset_name': "50 mm carbon", 'wheelset_kg': 1.5,
         'groupset_name': "12-speed", 'groupset_kg': 2.4, 'tyres_name': "28 mm", 'tyres_kg': 0.65,
         'extras_kg': 1.2, 'crr': 0.0035}


def test_weight_is_the_sum_of_the_parts_and_crr_comes_from_the_tyres():
    assert bk.bike_weight_kg(PARTS) == pytest.approx(6.85)
    assert bk.bike_physics(PARTS) == (pytest.approx(6.85), 0.0035)


def test_missing_bike_or_missing_values_fall_back_to_the_generic_ones():
    assert bk.bike_physics(None) == (8.0, 0.005)
    empty = {'name': "New", 'crr': None}
    assert bk.bike_physics(empty) == (8.0, 0.005)                    # no weights entered yet
    assert bk.bike_physics({**PARTS, 'crr': 0.5})[1] == 0.005        # an absurd Crr is not used
    assert bk.bike_weight_kg({'frame_kg': 1.0}) == 1.0               # parts left empty count as 0


def test_a_bike_has_no_cda():
    assert not any('cda' in k.lower() for k in PARTS)


def test_save_list_update_delete(temp_db):
    db.save_user("Ana")
    ok, _ = db.save_bike("Ana", "Road", PARTS)
    assert ok
    assert not db.save_bike("Ana", "Road", PARTS)[0]                 # names are unique per rider
    assert not db.save_bike("Ana", "  ", PARTS)[0]
    (bike,) = db.get_bikes("Ana")
    assert bike['name'] == "Road" and bike['is_default'] == 1 and bike['crr'] == 0.0035
    assert db.update_bike(bike['id'], "Road 2", {**PARTS, 'tyres_kg': 0.7})[0]
    assert db.get_bikes("Ana")[0]['tyres_kg'] == 0.7
    db.delete_bike(bike['id'])
    assert db.get_bikes("Ana") == []


def test_default_bike_moves_with_set_and_delete(temp_db):
    db.save_user("Ana")
    db.save_bike("Ana", "A", PARTS)
    db.save_bike("Ana", "B", PARTS)
    a, b = sorted(db.get_bikes("Ana"), key=lambda x: x['name'])
    assert a['is_default'] and not b['is_default']                   # the first bike is the default
    db.set_default_bike("Ana", b['id'])
    assert db.get_bikes("Ana")[0]['name'] == "B"                     # default first
    db.delete_bike(b['id'])
    assert db.get_bikes("Ana")[0]['name'] == "A" and db.get_bikes("Ana")[0]['is_default']


def test_two_riders_can_use_the_same_bike_name(temp_db):
    db.save_user("Ana")
    db.save_user("Ben")
    assert db.save_bike("Ana", "Road", PARTS)[0] and db.save_bike("Ben", "Road", PARTS)[0]
    assert len(db.get_bikes("Ana")) == 1


def _save(user, content=b"ride-1", bike_id=None):
    import pandas as pd
    meta = {'date': "2026-10-01T08:00:00Z", 'dist': 30, 'ele': 100, 'speed': 28, 'np': 200, 'avg_p': 190, 'tss': 60,
            'if': .8, 'avg_hr': 140, 'bike_id': bike_id}
    return db.save_ride(user, "r.fit", content, meta, {60: 250.0})


def test_a_ride_remembers_its_bike_and_loses_it_when_the_bike_is_deleted(temp_db):
    db.save_user("Ana")
    db.save_bike("Ana", "Road", PARTS)
    bike_id = db.get_bikes("Ana")[0]['id']
    assert _save("Ana", b"ride-1", bike_id)[0]
    ride_id, saved_bike = db.get_ride_bike("Ana", b"ride-1")
    assert saved_bike == bike_id
    db.set_ride_bike(ride_id, None)
    assert db.get_ride_bike("Ana", b"ride-1")[1] is None
    db.set_ride_bike(ride_id, bike_id)
    db.delete_bike(bike_id)
    assert db.get_ride_bike("Ana", b"ride-1")[1] is None
    assert db.get_ride_bike("Ana", b"never-saved") is None


def test_recomputing_a_ride_keeps_its_bike(temp_db):
    db.save_user("Ana")
    db.save_bike("Ana", "Road", PARTS)
    bike_id = db.get_bikes("Ana")[0]['id']
    _save("Ana", b"ride-1", bike_id)
    ride_id, _ = db.get_ride_bike("Ana", b"ride-1")
    db.update_ride_metrics(ride_id, {'date': "2026-10-01T08:00:00Z", 'dist': 31}, {60: 260.0})
    assert db.get_ride_bike("Ana", b"ride-1")[1] == bike_id
