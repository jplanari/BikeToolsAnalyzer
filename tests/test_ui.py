"""
Loads the real app with Streamlit's AppTest against a seeded database, clicks through the
pages and fails on any exception. Needs Streamlit (CI installs requirements.txt); where it
is missing locally the module is skipped, but in CI a missing library is an error.
"""
import os

import pandas as pd
import pytest

try:
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    _have_streamlit = True
except ImportError:
    _have_streamlit = False
if not _have_streamlit:
    if os.environ.get("CI"):
        raise RuntimeError("streamlit must be installed in CI to test the app")
    pytest.skip("streamlit not installed", allow_module_level=True)

from src.data import db
from src.data import strava as sv
from src.data.pipeline import prepare_ride, summarize_ride, metrics_for_storage
from tests.test_strava import make_streams, activity

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


@pytest.fixture
def empty_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "ui.db"))
    st.cache_data.clear()
    db.init_db()
    yield
    st.cache_data.clear()


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
def seeded_db(empty_db):
    db.save_user("Rider", 250, 170, 70.0, 180.0)
    today = pd.Timestamp.today().normalize()
    for i in range(30):
        _save_ride(i, today - pd.Timedelta(days=3 * i + 1), 150 + (i % 6) * 20)


# --- helpers -------------------------------------------------------------------------

def _app():
    return AppTest.from_file(APP, default_timeout=90).run()


def _widget(elements, label):
    for e in elements:
        if getattr(e, "label", None) == label:
            return e
    raise AssertionError(f"no widget labelled {label!r}; have {[getattr(e, 'label', None) for e in elements]}")


def _go(at, page):
    _widget(at.sidebar.radio, "Go to").set_value(page).run()
    return at


def _no_errors(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _texts(elements):
    return " ".join(str(getattr(e, "value", "")) for e in elements)


# --- tests ---------------------------------------------------------------------------

def test_every_page_loads_with_no_users(empty_db):
    at = _app()
    _no_errors(at)
    assert "Upload a GPX/FIT" in _texts(at.info)
    for page in ("Training", "User Corner", "Analyze Upload"):
        _no_errors(_go(at, page))


def test_training_page(seeded_db):
    at = _go(_app(), "Training")
    _no_errors(at)
    labels = [m.label for m in at.metric]
    assert {"Fitness (CTL)", "Fatigue (ATL)", "Form (TSB)"} <= set(labels)

    # the widgets that re-run the page
    _widget(at.radio, "Show").set_value("6 months").run()
    _no_errors(at)
    _widget(at.radio, "Group by").set_value("Month").run()
    _no_errors(at)
    _widget(at.radio, "Measure").set_value("TSS").run()
    _no_errors(at)


def test_every_training_section_renders_and_keeps_its_settings(seeded_db):
    at = _go(_app(), "Training")
    options = list(_widget(at.radio, "Section").options)
    assert len(options) == 4
    _widget(at.radio, "Show").set_value("6 months").run()
    for name in options[1:]:                       # only the selected section is drawn
        _widget(at.radio, "Section").set_value(name).run()
        _no_errors(at)
    _widget(at.radio, "Section").set_value(options[0]).run()
    _no_errors(at)
    assert _widget(at.radio, "Show").value == "6 months"      # survived the other sections


def test_user_corner_and_deeper_analytics(seeded_db):
    at = _go(_app(), "User Corner")
    _no_errors(at)
    assert "Deeper Analytics" in _texts(at.subheader)
    labels = [m.label for m in at.metric]
    assert "FTP" in labels and "Correlation (r)" in labels      # seeded rides have power and HR
    _widget(at.checkbox, "Compare with previous period").uncheck().run()
    _no_errors(at)
    for name in list(_widget(at.radio, "Section").options)[1:]:
        _widget(at.radio, "Section").set_value(name).run()
        _no_errors(at)


def test_pages_see_data_saved_after_the_first_run(seeded_db):
    """The cached reads must notice a write made between two runs."""
    at = _go(_app(), "Training")
    assert "older version" not in _texts(at.warning)
    today = pd.Timestamp.today().normalize()
    meta = {'date': f"{today:%Y-%m-%d}T07:00:00", 'dist': 10, 'ele': 100, 'speed': 25, 'np': 200,
            'avg_p': 180, 'tss': 50, 'if': 0.8, 'avg_hr': 140}          # no metrics: needs an update
    assert db.save_ride("Rider", "old.gpx", b"legacy", meta, {60: 300.0})[0]
    at.run()
    _no_errors(at)
    assert "1 rides were saved by an older version" in _texts(at.warning)


def test_build_a_session_from_intervals_and_remove_it(seeded_db):
    at = _go(_app(), "Training")
    _no_errors(at)
    assert {"Duration", "Normalized", "IF", "TSS"} <= {m.label for m in at.metric}   # builder preview

    at.button(key="add_session_Rider").click().run()                                  # default blocks
    _no_errors(at)
    today = pd.Timestamp.today().normalize()
    built = db.get_planned_sessions("Rider", after=today, structured=True)
    assert len(built) == 1 and 50 < built["tss"].iloc[0] < 100
    assert "With the planned sessions" in _texts(at.info)                             # forecast uses it

    at.button(key=f"del_plan_{int(built['id'].iloc[0])}").click().run()
    _no_errors(at)
    assert db.get_planned_sessions("Rider", after=today, structured=True).empty


def test_ftp_suggestion_can_be_accepted_with_one_click(seeded_db):
    # a hard 40-minute ride: 95 % of its best 20 min is well above the 250 W profile FTP
    _save_ride(99, pd.Timestamp.today().normalize() - pd.Timedelta(days=2), 330)
    at = _go(_app(), "Training")
    _no_errors(at)
    assert "Your FTP may be higher" in _texts(at.success)
    assert db.get_user_data("Rider")["ftp"] == 250                   # nothing changed yet
    at.button(key="accept_ftp_Rider").click().run()
    _no_errors(at)
    assert abs(db.get_user_data("Rider")["ftp"] - 313.5) <= 1         # 95 % of 330 W
    assert "Your FTP may be higher" not in _texts(at.success)
