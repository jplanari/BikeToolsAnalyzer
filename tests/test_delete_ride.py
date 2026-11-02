"""Deleting a ride: everything of it goes, only the owner can, and a Strava/Wahoo ride is not synced back."""
from src.data import db
from tests.test_nutrition import seeded  # noqa: F401
from tests.test_strava import real_distances  # noqa: F401


def ids(user="Rider"):
    return sorted(r["id"] for r in db.get_user_rides(user))


def count(table, ride_id):
    with db.connect() as c:
        return c.execute(f"SELECT COUNT(*) FROM {table} WHERE ride_id = ?", (ride_id,)).fetchone()[0]


def test_delete_removes_the_ride_and_everything_stored_for_it(seeded):
    first, *others = ids()
    assert count("ride_files", first) == 1 and count("ride_zones", first) > 0
    ok, msg = db.delete_ride("Rider", first)
    assert ok, msg
    assert ids() == others
    for table in ("ride_files", "ride_zones", "ride_hist", "power_records"):
        assert count(table, first) == 0, table
    assert count("ride_files", others[0]) == 1                           # the others are untouched


def test_only_the_owner_can_delete(seeded):
    db.save_user("Other", 200, 170, 70.0, 175.0)
    target = ids()[0]
    ok, _ = db.delete_ride("Other", target)
    assert not ok and target in ids()
    assert db.delete_ride("Rider", 999999)[0] is False
    assert db.delete_ride("Nobody", target)[0] is False


def test_a_deleted_strava_ride_is_not_imported_again_until_allowed(seeded):
    rides = db.get_user_rides("Rider")
    victim = next(r for r in rides if r["filename"].startswith("strava_"))
    activity = int(victim["filename"][len("strava_"):].split(".")[0])
    assert activity in db.get_imported_strava_ids("Rider")
    db.delete_ride("Rider", victim["id"])
    assert activity in db.get_imported_strava_ids("Rider")               # remembered: the sync skips it
    assert db.count_deleted_rides("Rider") == 1
    assert db.forget_deleted_rides("Rider") == 1
    assert activity not in db.get_imported_strava_ids("Rider") and db.count_deleted_rides("Rider") == 0


def test_a_deleted_wahoo_ride_is_remembered_but_an_upload_is_not(seeded):
    with db.connect() as c:                                              # a ride that came from Wahoo, and a file upload
        uid = c.execute("SELECT id FROM users WHERE name = 'Rider'").fetchone()[0]
        for name, h in (("wahoo_555.fit", "h1"), ("morning.fit", "h2")):
            c.execute("INSERT INTO rides (user_id, filename, file_hash, date_time) VALUES (?, ?, ?, ?)",
                      (uid, name, h, "2026-01-01T07:00:00+00:00"))
        wahoo, upload = (c.execute("SELECT id FROM rides WHERE filename = ?", (n,)).fetchone()[0]
                         for n in ("wahoo_555.fit", "morning.fit"))
    db.delete_ride("Rider", wahoo)
    db.delete_ride("Rider", upload)
    assert 555 in db.get_imported_wahoo_ids("Rider")
    assert db.count_deleted_rides("Rider") == 1                          # the upload is not tombstoned


def test_deleting_a_rider_forgets_their_deleted_rides_too(seeded):
    db.delete_ride("Rider", ids()[0])
    db.delete_user("Rider")
    with db.connect() as c:
        assert c.execute("SELECT COUNT(*) FROM deleted_rides").fetchone()[0] == 0
