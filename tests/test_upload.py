"""Uploads from any bike computer or platform: wrappers (gz, zip), TCX, damaged FIT, indoor rides, bulk import."""
import gzip
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.data import db, fit, pipeline, tcx, upload
from src.data import upload_import as ui

FIT_HEADER = b"\x0e\x10\x00\x00\x10\x00\x00\x00.FIT\x00\x00"       # 14-byte header with the ".FIT" signature


def fake_fit(extra=b""):
    return FIT_HEADER + b"payload" + extra


def tcx_ride(n=1800, sport="Biking", power=200, start="2026-10-01T07:00:00Z", position=True):
    pts = []
    t0 = pd.Timestamp(start)
    for i in range(n):
        pos = (f"<Position><LatitudeDegrees>{41 + i * 8 / 111_000:.6f}</LatitudeDegrees>"
               f"<LongitudeDegrees>2.0</LongitudeDegrees></Position>") if position else ""
        pts.append(f"<Trackpoint><Time>{(t0 + pd.Timedelta(seconds=i)):%Y-%m-%dT%H:%M:%SZ}</Time>{pos}"
                   f"<AltitudeMeters>{100 + i * 0.02:.2f}</AltitudeMeters><DistanceMeters>{i * 8.0}</DistanceMeters>"
                   f"<HeartRateBpm><Value>140</Value></HeartRateBpm><Cadence>90</Cadence>"
                   f"<Extensions><ns3:TPX xmlns:ns3=\"http://www.garmin.com/xmlschemas/ActivityExtension/v2\">"
                   f"<ns3:Speed>8.0</ns3:Speed><ns3:Watts>{power}</ns3:Watts></ns3:TPX></Extensions></Trackpoint>")
    return ("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            "<TrainingCenterDatabase xmlns=\"http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2\">"
            f"<Activities><Activity Sport=\"{sport}\"><Id>x</Id><Lap StartTime=\"{start}\"><Track>"
            + "".join(pts) + "</Track></Lap></Activity></Activities></TrainingCenterDatabase>").encode()


def zipped(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


# --- wrappers -----------------------------------------------------------------------

def test_content_decides_the_kind_not_the_name():
    assert upload.sniff_kind(fake_fit()) == "fit"
    assert upload.sniff_kind(tcx_ride(3)) == "tcx"
    assert upload.sniff_kind(b"\xef\xbb\xbf<?xml version='1.0'?><gpx version='1.1'></gpx>") == "gpx"
    assert upload.sniff_kind(gzip.compress(b"x")) == "gzip" and upload.sniff_kind(zipped({"a": b"b"})) == "zip"
    assert upload.sniff_kind(b"hello") is None


def test_plain_and_gzipped_files_get_the_right_name():
    acts, notes = upload.expand_upload("Morning Ride", fake_fit())
    assert [(a.name, a.kind) for a in acts] == [("Morning Ride.fit", "fit")] and not notes
    acts, _ = upload.expand_upload("ride.fit.gz", gzip.compress(fake_fit()))
    assert [a.name for a in acts] == ["ride.fit"]
    acts, _ = upload.expand_upload("activity.xml", tcx_ride(3))
    assert [(a.name, a.kind) for a in acts] == [("activity.tcx", "tcx")]


def test_bulk_zip_with_nested_archives_and_junk():
    inner = zipped({"2.fit.gz": gzip.compress(fake_fit(b"2"))})
    bulk = zipped({"DI_CONNECT/uploads/1.fit": fake_fit(b"1"), "DI_CONNECT/part.zip": inner,
                   "__MACOSX/._1.fit": fake_fit(), ".hidden.fit": fake_fit(), "profile.json": b"{}", "notes.txt": b"hi"})
    acts, notes = upload.expand_upload("export.zip", bulk)
    assert sorted(a.name for a in acts) == ["1.fit", "2.fit"]
    assert notes == []                                    # json / txt inside an archive are normal: no noise


def test_unusable_uploads_explain_themselves():
    acts, notes = upload.expand_upload("photo.png", b"\x89PNG....")
    assert acts == [] and "not a ride file" in notes[0]
    acts, notes = upload.expand_upload("broken.zip", b"PK\x03\x04garbage")
    assert acts == [] and notes
    acts, notes = upload.expand_upload("empty.zip", zipped({"readme.txt": b"x"}))
    assert acts == [] and notes == ["empty.zip: no rides found inside."]


def test_zip_bombs_are_not_unpacked(monkeypatch):
    monkeypatch.setattr(upload, "MAX_ACTIVITY_BYTES", 1000)
    acts, notes = upload.expand_upload("big.zip", zipped({"big.fit": fake_fit(b"x" * 5000)}))
    assert acts == [] and "too large" in notes[0]


# --- TCX ----------------------------------------------------------------------------

def test_tcx_channels_and_sport():
    df = tcx.parse_tcx(tcx_ride(60))
    assert len(df) == 60 and df['time'].dt.tz is None
    assert df['power'].eq(200).all() and df['hr'].eq(140).all() and df['cadence'].eq(90).all()
    assert df['speed'].eq(8.0).all() and df['dist_m'].iloc[-1] == 59 * 8.0
    assert df['lat'].iloc[0] == pytest.approx(41.0) and df['lon'].iloc[0] == 2.0
    assert tcx.tcx_sport(tcx_ride(3, sport="Running")) == "Running"


def test_tcx_without_position_and_damaged_xml():
    df = tcx.parse_tcx(tcx_ride(30, position=False))
    assert df['lat'].isna().all() and df['power'].notna().all()
    with pytest.raises(ValueError):
        tcx.parse_tcx(b"<TrainingCenterDatabase><unclosed>")
    assert tcx.parse_tcx(b"<TrainingCenterDatabase/>").empty


def test_a_tcx_goes_through_the_whole_pipeline_whatever_its_name():
    df, dist, ele = pipeline.prepare_ride("activity", tcx_ride(1800, position=False))   # recorded distance, no GPS
    assert dist == pytest.approx(1799 * 8, rel=0.02) and ele > 30
    s = pipeline.summarize_ride(df, 250, 170)
    assert s['avg_power'] == pytest.approx(200) and s['moving_time_s'] == 1800


# --- FIT quirks ---------------------------------------------------------------------

class Field:
    def __init__(self, name, value):
        self.name, self.value = name, value


def _records(n, **extra):
    start = pd.Timestamp('2026-10-01 07:00:00')
    return [[Field('timestamp', start + pd.Timedelta(seconds=i)), Field('heart_rate', 140), Field('cadence', 88),
             Field('power', 210)] + [Field(k, v(i) if callable(v) else v) for k, v in extra.items()]
            for i in range(n)]


def test_indoor_fit_without_gps_keeps_distance_and_gradient(monkeypatch):
    """Zwift / trainer rides: no position, so distance and gradient must come from the recorded distance and altitude."""
    records = _records(1800, distance=lambda i: i * 8.0, enhanced_altitude=lambda i: 100 + i * 0.4,
                       enhanced_speed=8.0)

    class FakeFit:
        def __init__(self, *_, **__): pass
        def get_messages(self, kind): return records if kind == 'record' else []

    monkeypatch.setattr(fit.fitparse, 'FitFile', FakeFit, raising=False)
    getattr(fit.parse_fit, 'clear', lambda: None)()
    df, dist, ele = pipeline.prepare_ride('zwift.fit', fake_fit())
    assert dist == pytest.approx(1799 * 8, rel=0.01) and df['dist_m'].iloc[-1] == pytest.approx(1799 * 8, rel=0.01)
    assert ele == pytest.approx(1799 * 0.4, rel=0.02)
    assert df['grade_smooth'].iloc[100:-100].mean() == pytest.approx(5.0, abs=0.5)      # 0.4 m per 8 m


def test_truncated_fit_keeps_the_samples_before_the_damage(monkeypatch):
    getattr(fit.parse_fit, 'clear', lambda: None)()       # st.cache_data would replay the other test's parse of the same bytes
    records = _records(1200, enhanced_speed=8.0)

    class FakeFit:
        def __init__(self, *args, **kw):
            assert kw.get('check_crc') is False            # damaged checksums must not stop the read
        def get_messages(self, kind):
            if kind != 'record':
                return []
            def gen():
                yield from records
                raise RuntimeError("unexpected end of file")
            return gen()

    monkeypatch.setattr(fit.fitparse, 'FitFile', FakeFit, raising=False)
    df = fit.parse_fit(fake_fit())
    assert len(df) == 1200 and df['power'].eq(210).all()


# --- bulk import --------------------------------------------------------------------

@pytest.fixture
def rider(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "u.db"))
    db.init_db()
    db.save_user("Rider", 250, 170, 70.0, 180.0)


def _acts(*rides):
    out = []
    for i, data in enumerate(rides):
        out.extend(upload.expand_upload(f"r{i}.tcx", data)[0])
    return out


def test_import_saves_each_ride_once_and_skips_what_is_not_a_ride(rider):
    ride = tcx_ride(1800, start="2026-10-01T07:00:00Z")
    other = tcx_ride(1800, start="2026-10-03T07:00:00Z", power=250)
    run = tcx_ride(1800, sport="Running", start="2026-10-05T07:00:00Z")
    empty = b"<TrainingCenterDatabase xmlns='http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'/>"
    res = ui.import_activities("Rider", _acts(ride, other, run, empty), 250, 170)
    assert (res.found, res.imported, res.no_data, res.already_imported) == (4, 2, 2, 0) and not res.failed
    assert len(db.get_user_rides("Rider")) == 2

    again = ui.import_activities("Rider", _acts(ride, other), 250, 170)
    assert (again.imported, again.already_imported) == (0, 2)
    near = tcx_ride(1800, start="2026-10-01T07:01:00Z")                 # same ride from another device, 60 s later
    assert ui.import_activities("Rider", _acts(near), 250, 170).already_imported == 1
    assert len(db.get_user_rides("Rider")) == 2


def test_two_copies_in_one_upload_are_one_ride(rider):
    ride = tcx_ride(1200)
    res = ui.import_activities("Rider", _acts(ride, ride), 250, 170)
    assert (res.imported, res.already_imported) == (1, 1)
