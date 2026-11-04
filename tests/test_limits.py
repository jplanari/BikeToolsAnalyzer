"""Limits that keep a shared server safe from oversized uploads and from one rider filling the disk."""
import io
import os
import zipfile

import pytest

from src import serverconf
from src.data import db
from src.data.parsers import upload
from tests.test_db import temp_db  # noqa: F401 (fixture)

GPX = (b'<?xml version="1.0"?><gpx version="1.1"><trk><trkseg>' + b'<trkpt lat="1" lon="1"><time>2026-01-01T00:00:00Z</time></trkpt>' * 40
       + b'</trkseg></trk></gpx>')


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("BIKETOOLS_SERVER_MODE", "BIKETOOLS_MAX_ACTIVITY_MB", "BIKETOOLS_MAX_UNPACKED_MB", "BIKETOOLS_RIDER_QUOTA_MB"):
        monkeypatch.delenv(name, raising=False)


def test_defaults_are_tighter_on_a_server(monkeypatch):
    assert (serverconf.max_activity_bytes(), serverconf.max_unpacked_bytes(), serverconf.rider_quota_bytes()) == (
        200 * 2 ** 20, 2048 * 2 ** 20, 0)
    monkeypatch.setenv("BIKETOOLS_SERVER_MODE", "1")
    assert (serverconf.max_activity_bytes(), serverconf.max_unpacked_bytes(), serverconf.rider_quota_bytes()) == (
        50 * 2 ** 20, 500 * 2 ** 20, 500 * 2 ** 20)
    monkeypatch.setenv("BIKETOOLS_RIDER_QUOTA_MB", "0")                   # 0 switches a limit off
    assert serverconf.rider_quota_bytes() == 0
    monkeypatch.setenv("BIKETOOLS_RIDER_QUOTA_MB", "not a number")        # a typo falls back to the default
    assert serverconf.rider_quota_bytes() == 500 * 2 ** 20


def test_a_ride_file_over_the_limit_is_refused(monkeypatch):
    acts, notes = upload.expand_upload("a.gpx", GPX)
    assert len(acts) == 1 and not notes
    monkeypatch.setenv("BIKETOOLS_MAX_ACTIVITY_MB", "0.001")              # about 1 KB
    acts, notes = upload.expand_upload("a.gpx", GPX)
    assert acts == [] and "too large" in notes[0]


def test_an_archive_that_unpacks_to_too_much_is_cut_off(monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(10):
            zf.writestr(f"ride{i}.gpx", GPX)
    monkeypatch.setenv("BIKETOOLS_MAX_UNPACKED_MB", str(len(GPX) * 3.5 / 2 ** 20))       # room for three files
    acts, notes = upload.expand_upload("bulk.zip", buf.getvalue())
    assert len(acts) == 3 and any("too large" in n for n in notes)


def test_a_gzip_that_expands_past_the_limit_is_refused(monkeypatch):
    import gzip
    packed = gzip.compress(GPX * 50)                                       # a small file that grows a lot
    monkeypatch.setenv("BIKETOOLS_MAX_ACTIVITY_MB", "0.01")
    acts, notes = upload.expand_upload("big.gpx.gz", packed)
    assert acts == [] and "too large" in notes[0]


def _save(rider, name, size):
    return db.save_ride(rider, name, os.urandom(size), {"date": "2026-10-01T08:00:00Z", "dist": 30}, {60: 300.0})


def test_a_rider_cannot_store_more_than_the_quota(temp_db, monkeypatch):
    monkeypatch.setenv("BIKETOOLS_RIDER_QUOTA_MB", str(3000 / 2 ** 20))    # about 3 KB
    db.save_user("Ana", 250, 170, 70.0, 175.0)
    db.save_user("Beni", 250, 170, 70.0, 175.0)
    assert _save("Ana", "1.fit", 1400)[0] and _save("Ana", "2.fit", 1400)[0]
    ok, msg = _save("Ana", "3.fit", 1400)
    assert not ok and db.is_quota_message(msg)
    assert 2800 <= db.stored_bytes("Ana") < 3000 and len(db.get_user_rides("Ana")) == 2          # nothing half-saved
    assert _save("Beni", "1.fit", 1400)[0]                                 # the limit is per rider
    db.delete_ride("Ana", db.get_user_rides("Ana")[0]["id"])
    assert _save("Ana", "4.fit", 1400)[0]                                  # deleting makes room


def test_no_quota_on_the_desktop(temp_db):
    db.save_user("Ana", 250, 170, 70.0, 175.0)
    for i in range(4):
        assert _save("Ana", f"{i}.fit", 1400)[0]
