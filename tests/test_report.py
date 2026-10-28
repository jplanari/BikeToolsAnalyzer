import re

import pandas as pd
import pytest

from src.data import db
from src.data import strava as sv
from src.data.pipeline import prepare_ride, summarize_ride, metrics_for_storage
from src.data.report_data import collect_report_data
from src.viz.report_pdf import build_report_pdf
from tests.test_strava import make_streams, activity, real_distances  # noqa: F401 (fixture)


def _save_ride(i, day, power):
    streams = make_streams(n=2400, power=float(power), hr=120.0 + power / 10)
    payload = sv.build_ride_payload(activity(100 + i, f"{day:%Y-%m-%d}T07:00:00Z"), streams)
    df, dist, ele = prepare_ride(sv.ride_filename(100 + i), payload)
    s = summarize_ride(df, 250, 170)
    meta = db.build_ride_metadata(df, dist, ele, s['avg_speed_kph'], s['norm_power'], s['tss'], s['if'],
                                  filename=sv.ride_filename(100 + i), metrics=metrics_for_storage(s))
    ok, msg = db.save_ride("Rider", sv.ride_filename(100 + i), payload, meta, s['power_curve'])
    assert ok, msg


@pytest.fixture
def seeded(tmp_path, monkeypatch, real_distances):
    monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "r.db"))
    db.init_db()
    db.save_user("Rider", 250, 170, 70.0, 180.0)
    today = pd.Timestamp.today().normalize()
    for i in range(30):
        _save_ride(i, today - pd.Timedelta(days=3 * i + 1), 150 + (i % 6) * 20)
    db.add_structured_session("Rider", (today + pd.Timedelta(days=2)).strftime('%Y-%m-%d'),
                              [{'name': 'Sweet spot', 'reps': 3, 'work_min': 10, 'work_w': 230,
                                'rest_min': 5, 'rest_w': 120}], 250)


def _pages(pdf):
    return len(re.findall(rb"/Type\s*/Page\b", pdf))


@pytest.mark.parametrize("days", [28, 91, None])
def test_report_builds(seeded, days):
    pdf = build_report_pdf(collect_report_data("Rider", days))
    assert pdf.startswith(b"%PDF") and _pages(pdf) == 6


def test_report_without_rides(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "e.db"))
    db.init_db()
    db.save_user("Nobody", 250, 170, 70.0, 180.0)
    pdf = build_report_pdf(collect_report_data("Nobody", 91))
    assert pdf.startswith(b"%PDF") and _pages(pdf) == 6


def test_unknown_user(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "e.db"))
    db.init_db()
    with pytest.raises(ValueError):
        collect_report_data("ghost")
