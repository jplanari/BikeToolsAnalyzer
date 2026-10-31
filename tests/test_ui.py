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
    _widget(at.radio, "Go to").set_value(page).run()
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
    assert "create your profile" in _texts(at.subheader)           # no user: the welcome page is only the form
    assert not at.info or "Upload a GPX/FIT" not in _texts(at.info)
    for page in ("Training", "User Corner", "Analyze Upload"):
        _no_errors(_go(at, page))


def test_first_profile_form_creates_the_user_and_shows_the_gallery_hint(empty_db):
    at = _app()
    at.text_input(key="first_profile_name").set_value("Ana")
    next(b for b in at.button if b.label == "Create profile").click().run()
    _no_errors(at)
    assert db.get_user_data("Ana")
    assert "Upload a GPX/FIT" in _texts(at.info)


def test_first_profile_needs_only_a_name_and_explains_ftp_and_lthr(empty_db):
    at = _app()
    assert "What are FTP, LTHR and W′?" in [e.label for e in at.expander]
    at.text_input(key="first_profile_name").set_value("Bea")
    next(b for b in at.button if b.label == "Create profile").click().run()
    _no_errors(at)
    user = db.get_user_data("Bea")
    assert user["ftp"] == 200 and user["lthr"] == 170                       # empty fields fall back to the defaults
    assert user["w_prime"] == 20000


def test_first_profile_takes_the_riders_w_prime_in_kilojoules(empty_db):
    at = _app()
    at.text_input(key="first_profile_name").set_value("Cris")
    at.number_input(key="first_profile_wprime").set_value(26.5)
    next(b for b in at.button if b.label == "Create profile").click().run()
    _no_errors(at)
    assert db.get_user_data("Cris")["w_prime"] == 26500


def test_empty_ftp_and_lthr_follow_the_rides_in_the_history(empty_db):
    db.save_user("Auto", None, None, 70.0, 180.0)
    today = pd.Timestamp.today().normalize()
    streams = make_streams(n=2400, power=300.0, hr=160.0)
    payload = sv.build_ride_payload(activity(500, f"{today - pd.Timedelta(days=2):%Y-%m-%d}T07:00:00Z"), streams)
    df, dist, ele = prepare_ride(sv.ride_filename(500), payload)
    s = summarize_ride(df, 200, 170)
    meta = db.build_ride_metadata(df, dist, ele, s['avg_speed_kph'], s['norm_power'], s['tss'], s['if'],
                                  filename=sv.ride_filename(500), metrics=metrics_for_storage(s))
    assert db.save_ride("Auto", sv.ride_filename(500), payload, meta, s['power_curve'])[0]
    at = _app()
    _no_errors(at)
    user = db.get_user_data("Auto")
    assert abs(user["ftp"] - 285) <= 3 and 150 <= user["lthr"] <= 154          # 95 % of 300 W and of 160 bpm
    assert user["ftp_auto"] == 1 and user["lthr_auto"] == 1


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
    assert len(options) == 5
    _widget(at.radio, "Show").set_value("6 months").run()
    for name in options[1:]:                       # only the selected section is drawn
        _widget(at.radio, "Section").set_value(name).run()
        _no_errors(at)
    _widget(at.radio, "Section").set_value(options[0]).run()
    _no_errors(at)
    assert _widget(at.radio, "Show").value == "6 months"      # survived the other sections


def test_deeper_analytics_live_on_the_training_page(seeded_db):
    at = _go(_app(), "Training")
    _widget(at.radio, "Section").set_value("🔬 Deeper analytics").run()
    _no_errors(at)
    assert "Correlation (r)" in [m.label for m in at.metric]     # seeded rides have power and HR
    _widget(at.checkbox, "Compare with previous period").uncheck().run()
    _no_errors(at)
    inner = next(r for r in at.radio if r.key == "da_section")
    for name in list(inner.options)[1:]:
        next(r for r in at.radio if r.key == "da_section").set_value(name).run()
        _no_errors(at)
    _widget(at.radio, "Section").set_value("📈 Load").run()      # leaving and coming back keeps the choice
    _widget(at.radio, "Section").set_value("🔬 Deeper analytics").run()
    _no_errors(at)
    assert _widget(at.checkbox, "Compare with previous period").value is False


def test_user_corner_keeps_the_profile_and_maintenance_only(seeded_db):
    at = _go(_app(), "User Corner")
    _no_errors(at)
    assert {"FTP", "LTHR", "Weight"} <= {m.label for m in at.metric}
    assert "Correlation (r)" not in [m.label for m in at.metric]
    assert not at.subheader or "Deeper Analytics" not in _texts(at.subheader)


def test_power_profile_section_with_the_critical_power_expander_renders(seeded_db):
    at = _go(_app(), "Training")
    _widget(at.radio, "Section").set_value("⚡ Power profile").run()
    _no_errors(at)


def test_ride_analysis_has_no_settings_in_the_sidebar(seeded_db):
    at = _app()
    assert not at.sidebar.checkbox
    assert "Settings" not in _texts(at.sidebar.header)


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


def test_wahoo_source_without_credentials_explains_the_setup(seeded_db):
    at = _app()
    _widget(at.sidebar.radio, "Source").set_value("Wahoo").run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Wahoo isn't set up yet" in _texts(at.sidebar.warning)


def test_polar_source_without_credentials_explains_the_setup(seeded_db):
    at = _app()
    _widget(at.sidebar.radio, "Source").set_value("Polar").run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Polar isn't set up yet" in _texts(at.sidebar.warning)


def test_looks_can_be_changed_and_navigation_is_a_top_bar(seeded_db):
    at = _app()
    _no_errors(at)
    assert not [r for r in at.sidebar.radio if r.label == "Go to"]          # navigation is not in the sidebar
    for look in ("Dashboard", "Classic", "Compact"):
        _widget(at.sidebar.selectbox, "Look").set_value(look).run()
        _no_errors(at)
    assert not [r for r in at.sidebar.radio if r.label == "Navigation"]      # no placement choice any more
    _widget(at.radio, "Go to").set_value("Training").run()
    _no_errors(at)
    assert {"Fitness (CTL)", "Fatigue (ATL)", "Form (TSB)"} <= {m.label for m in at.metric}


def test_deleting_a_user_needs_confirmation(seeded_db):
    at = _app()
    assert not [b for b in at.sidebar.button if b.label == "Delete User"]    # no one-click delete any more
    assert db.get_user_data("Rider")
    at.button(key="confirm_delete_user").click().run()
    assert not db.get_user_data("Rider")


def test_gallery_opens_a_saved_ride_and_draws_only_the_chosen_section(seeded_db, monkeypatch):
    import src.viz.ride_analysis as ra
    monkeypatch.setattr(ra, "fetch_ride_weather", lambda df: df)          # no network in tests
    at = _app()
    _no_errors(at)
    assert "Last 7 days" in {m.label for m in at.metric}
    next(b for b in at.button if b.label == "Open ride").click().run()
    _no_errors(at)
    section = next(r for r in at.radio if r.label == "Section")
    assert section.options[0] == "📊 Summary" and "📐 Segments" in section.options
    assert "🗺️ Route" not in section.options                              # the route lives in the Summary
    assert {"Performance Metrics", "Route"} <= {s.value for s in at.subheader}
    assert "Power Curve" not in _texts(at.subheader)                       # other sections' charts are not built
    section.set_value("⚡ Power").run()
    _no_errors(at)
    assert "Power Curve" in _texts(at.subheader) and at.get("plotly_chart")
    section.set_value("📈 Time series").run()
    _no_errors(at)
    assert at.get("plotly_chart") and "Channels under the elevation profile" in [m.label for m in at.multiselect]
    assert not [s for s in at.selectbox if s.label == "Smoothing"]            # always 5 s
    assert not [r for r in at.radio if r.label in ("Layout", "X axis")]
    next(b for b in at.button if b.key == "back_to_gallery").click().run()
    assert "Last 7 days" in {m.label for m in at.metric}


def test_user_corner_adds_a_bike_from_its_parts(seeded_db):
    at = _go(_app(), "User Corner")
    _no_errors(at)
    assert "🚲 Bikes" in _texts(at.subheader)
    tag = f"Rider_new"
    at.text_input(key=f"bike_name_{tag}").set_value("Road")
    at.number_input(key=f"kg_frame_{tag}").set_value(1.2)
    at.number_input(key=f"kg_wheelset_{tag}").set_value(1.6)
    at.number_input(key=f"kg_groupset_{tag}").set_value(2.5)
    at.number_input(key=f"kg_tyres_{tag}").set_value(0.7)
    at.number_input(key=f"crr_{tag}").set_value(0.004)
    next(b for b in at.button if b.label == "Add bike").click().run()
    _no_errors(at)
    (bike,) = db.get_bikes("Rider")
    assert bike["name"] == "Road" and bike["crr"] == pytest.approx(0.004)
    assert bike["frame_kg"] + bike["wheelset_kg"] + bike["groupset_kg"] + bike["tyres_kg"] == pytest.approx(6.0)
    assert any("Road" in e.label and "6.00 kg" in e.label for e in at.expander)
