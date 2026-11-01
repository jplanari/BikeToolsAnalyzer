"""Layer 2 of the nutrition analysis: what the rider ate, stored per ride and compared with what the ride burned."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.analysis import nutrition as nut
from src.data import db
from tests.test_nutrition import seeded, Recorder, texts  # noqa: F401 (fixture and helpers)
from tests.test_strava import real_distances  # noqa: F401 (fixture)


def ride_id_of(user="Rider", index=0):
    return db.get_user_rides(user)[index]['id']


# --- stored -----------------------------------------------------------------------------------------------------

def test_fuel_is_stored_cleared_and_validated(seeded):
    rid = ride_id_of()
    assert db.get_ride_fuel(rid) == (None, None)                   # nothing logged yet
    assert db.set_ride_fuel(rid, 240, 1500) == (True, "")
    assert db.get_ride_fuel(rid) == (240.0, 1500.0)
    assert db.set_ride_fuel(rid, 0, None)[0] and db.get_ride_fuel(rid) == (0.0, None)   # 0 is "nothing", None "not logged"
    assert db.set_ride_fuel(rid, float('nan'), float('nan'))[0] and db.get_ride_fuel(rid) == (None, None)
    for bad in ((-5, 100), (100, -1), (99999, 0), (0, 10 ** 6), ("abc", 0)):
        ok, msg = db.set_ride_fuel(rid, *bad)
        assert not ok and msg
    assert db.get_ride_fuel(rid) == (None, None)                   # a refused entry changes nothing
    assert db.set_ride_fuel(10 ** 6, 10, 10) == (False, "ride not found")


def test_logged_fuel_is_in_the_stats_and_survives_a_metrics_update(seeded):
    rid = ride_id_of()
    db.set_ride_fuel(rid, 180, 900)
    stats = db.get_ride_stats("Rider", seeded - pd.Timedelta(days=10), seeded)
    row = stats[stats['ride_id'] == rid].iloc[0]
    assert row['carb_eaten_g'] == 180 and row['fluid_ml'] == 900 and row['if_factor'] > 0 and row['filename']
    assert stats['carb_eaten_g'].notna().sum() == 1
    # "Update ride history" recomputes the metrics from the file; the rider's log is not a metric
    filename, content = db.get_ride_file(rid)
    from src.data.processing.pipeline import prepare_ride, summarize_ride, metrics_for_storage
    df, dist, ele = prepare_ride(filename, content)
    s = summarize_ride(df, 250, 170)
    meta = db.build_ride_metadata(df, dist, ele, s['avg_speed_kph'], s['norm_power'], s['tss'], s['if'],
                                  filename=filename, metrics=metrics_for_storage(s))
    db.update_ride_metrics(rid, meta, s['power_curve'])
    assert db.get_ride_fuel(rid) == (180.0, 900.0)


# --- the numbers ------------------------------------------------------------------------------------------------

def test_intake_rate_and_coverage():
    assert nut.intake_rate(180, 7200) == pytest.approx(90) and nut.intake_rate(None, 7200) is None
    assert nut.intake_rate(100, 0) is None and nut.intake_rate(float('nan'), 3600) is None
    assert nut.coverage(150, 300) == 0.5 and nut.coverage(300, 200) == 1.5
    assert nut.coverage(None, 300) is None and nut.coverage(50, 0) is None


def test_intake_status_against_the_guide():
    # 3 h ride: guide 60-90 g/h
    assert nut.intake_status(30, 3 * 3600) == 'below'
    assert nut.intake_status(75, 3 * 3600) == 'on'
    assert nut.intake_status(110, 3 * 3600) == 'above'
    assert nut.intake_status(0, 30 * 60) == 'none'          # a ride that needs nothing is not "below the guide"
    assert nut.intake_status(None, 3600) is None
    assert nut.intake_status(0, 3600) == 'below'            # an hour and a half: guide starts at 30


def test_ride_intake_summarises_a_log():
    summary = {'moving_time_s': 3 * 3600, 'carb_g': 400.0, 'if': 0.7}
    r = nut.ride_intake(summary, 240, 1800, 70.0)
    assert r['carb_rate'] == pytest.approx(80) and r['fluid_rate'] == pytest.approx(600)
    assert r['coverage'] == pytest.approx(0.6) and r['status'] == 'on' and r['net_g'] == pytest.approx(160)
    assert r['store_used'] == pytest.approx(160 / 420)
    empty = nut.ride_intake({'moving_time_s': 3600}, None)
    assert all(v is None for v in empty.values())


def logs(eaten):
    n = len(eaten)
    return pd.DataFrame({
        'date': pd.date_range('2026-10-05', periods=n, freq='D'), 'moving_s': [3 * 3600] * n, 'if_factor': 0.7,
        'carb_g': 300.0, 'carb_eaten_g': eaten, 'vi': 1.05, 'ef': 1.4,
        'decoupling_pct': [9, 8, 10, 9, 2, 3, 3, 2, 4][:n] + [3] * max(0, n - 9)})


def test_logged_rides_adds_rate_guide_status_and_coverage():
    s = nut.logged_rides(logs([90, np.nan, 270]))
    assert len(s) == 2 and list(s['status']) == ['below', 'above' if 90 < 270 / 3 else 'on']
    assert list(s['guide_low']) == [60, 60] and s['coverage'].iloc[1] == pytest.approx(0.9)
    assert nut.logged_rides(pd.DataFrame()).empty and nut.logged_rides(None).empty
    assert nut.logged_rides(logs([np.nan, np.nan])).empty


def test_totals_count_only_rides_that_needed_fuel():
    s = logs([90, 270, 240, np.nan])
    s.loc[3, 'moving_s'] = 1800                      # a short ride: no fuel needed, so not counted as missing
    t = nut.intake_totals(s)
    assert t['eligible'] == 3 and t['logged'] == 3 and (t['below'], t['on'], t['above']) == (1, 2, 0)
    assert t['rate_g_h'] == pytest.approx((90 + 270 + 240) / 9) and t['coverage'] == pytest.approx(600 / 900)
    assert nut.intake_totals(pd.DataFrame())['logged'] == 0


def test_burned_and_eaten_by_week():
    per = nut.intake_by_period(logs([90, 270, 240, 300, 60, 100, 100, 100]), 'W')
    assert list(per['rides']) == [7, 1] and per['burned_g'].iloc[0] == 2100 and per['eaten_g'].iloc[0] == 90 + 270 + 240 + 300 + 60 + 100 + 100
    assert nut.intake_by_period(logs([np.nan, np.nan]), 'W').empty


def test_fade_comparison_needs_enough_rides_on_both_sides():
    few = nut.fade_comparison(logs([90, 270, 240]))
    assert few and not few['enough'] and few['difference'] is None
    full = nut.fade_comparison(logs([60, 70, 80, 90, 250, 240, 260, 300]))
    assert full['enough'] and full['under']['n'] == 4 and full['fuelled']['n'] == 4
    assert full['under']['decoupling'] == pytest.approx(9.0) and full['fuelled']['decoupling'] == pytest.approx(2.5)
    assert full['difference'] == pytest.approx(-6.5) and full['rho'] < -0.7
    assert len(full['rides']) == 8


def test_fade_comparison_leaves_out_short_variable_or_unfuelled_need_rides():
    s = logs([60, 70, 80, 90, 250, 240, 260, 300])
    s.loc[0, 'moving_s'] = 60 * 60                    # under 90 minutes
    s.loc[1, 'vi'] = 1.3                              # a variable ride
    s.loc[2, 'decoupling_pct'] = np.nan               # nothing to compare
    assert len(nut.fade_comparison(s)['rides']) == 5
    assert nut.fade_comparison(logs([np.nan] * 4)) is None


# --- the screens ------------------------------------------------------------------------------------------------

class FakeSt:
    """Records what a page draws and answers inputs with chosen values."""
    def __init__(self, **answers):
        self.shown, self.answers = [], answers
        self.session_state = {}
        self.column_config = SimpleNamespace(NumberColumn=lambda *a, **k: None)

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.shown.append((name, args, kwargs))
            if name in self.answers:
                v = self.answers[name]
                return v(*args, **kwargs) if callable(v) else v
            if name == "columns":
                n = args[0] if isinstance(args[0], int) else len(args[0])
                return [self for _ in range(n)]
            return self
        return call

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def metrics(rec):
    return {a[0]: a[1] for n, a, k in rec.shown if n == "metric"}


def test_ride_log_form_saves_and_shows_the_result(seeded, monkeypatch):
    from src.viz import fuel
    rid = ride_id_of()
    inputs = iter([240.0, 1500.0])
    st = FakeSt(number_input=lambda *a, **k: next(inputs), form_submit_button=True)
    monkeypatch.setattr(fuel, "st", st)
    summary = {'kj': 2000.0, 'carb_g': 360.0, 'moving_time_s': 3 * 3600, 'if': 0.7}
    fuel.render_fuel_log(summary, 70.0, rid)
    assert db.get_ride_fuel(rid) == (240.0, 1500.0)
    m = metrics(st)
    assert m["Intake rate"] == "80 g/h" and m["Burned carbohydrate replaced"] == "67%" and m["Fluid"] == "500 ml/h"
    assert "Within the guide" in texts(st)


def test_ride_log_form_reports_a_refused_value_and_shows_nothing_when_empty(seeded, monkeypatch):
    from src.viz import fuel
    rid = ride_id_of()
    st = FakeSt(number_input=lambda *a, **k: None, form_submit_button=True)
    monkeypatch.setattr(fuel, "st", st)
    fuel.render_fuel_log({'kj': 1000.0, 'carb_g': 100.0, 'moving_time_s': 7200}, 70.0, rid)
    assert not metrics(st) and db.get_ride_fuel(rid) == (None, None)
    st = FakeSt(number_input=lambda *a, **k: -4.0, form_submit_button=True)
    monkeypatch.setattr(fuel, "st", st)
    fuel.render_fuel_log({'kj': 1000.0, 'carb_g': 100.0, 'moving_time_s': 7200}, 70.0, rid)
    assert any(n == "error" for n, a, k in st.shown)


def test_ride_summary_offers_the_log_only_for_a_saved_ride(seeded, monkeypatch):
    from src.viz import fuel
    summary = {'kj': 2000.0, 'carb_g': 360.0, 'moving_time_s': 3 * 3600, 'if': 0.7}
    st = FakeSt(number_input=lambda *a, **k: None, form_submit_button=False)
    monkeypatch.setattr(fuel, "st", st)
    fuel.render_fuel_summary(summary, 70.0, ride_id=ride_id_of(), user_name="Rider")
    assert any(n == "form" for n, a, k in st.shown)
    st = FakeSt()
    monkeypatch.setattr(fuel, "st", st)
    fuel.render_fuel_summary(summary, 70.0)
    assert not any(n == "form" for n, a, k in st.shown) and "Choose a user" in texts(st)


def test_training_log_table_saves_the_changed_rows(seeded, monkeypatch):
    from src.viz import training_page as tpg
    stats = db.get_ride_stats("Rider", seeded - pd.Timedelta(days=10), seeded).assign(moving_s=2 * 3600)  # long enough to need fuel

    def edit(frame, **k):
        out = frame.copy()
        out.loc[out.index[0], "Eaten (g)"] = 150.0
        out.loc[out.index[1], "Fluid (ml)"] = 800.0
        return out

    st = FakeSt(data_editor=edit, button=True)
    monkeypatch.setattr(tpg, "st", st)
    tpg._intake_editor(stats)
    rides = stats.sort_values('date', ascending=False)
    first, second = int(rides['ride_id'].iloc[0]), int(rides['ride_id'].iloc[1])
    assert db.get_ride_fuel(first) == (150.0, None) and db.get_ride_fuel(second) == (None, 800.0)
    assert any(n == "rerun" for n, a, k in st.shown)


def test_training_log_table_does_nothing_without_changes(seeded, monkeypatch):
    from src.viz import training_page as tpg
    stats = db.get_ride_stats("Rider", seeded - pd.Timedelta(days=10), seeded).assign(moving_s=2 * 3600)
    st = FakeSt(data_editor=lambda frame, **k: frame, button=True)
    monkeypatch.setattr(tpg, "st", st)
    tpg._intake_editor(stats)
    assert not any(n in ("button", "rerun") for n, a, k in st.shown)
    assert all(db.get_ride_fuel(r['id']) == (None, None) for r in db.get_user_rides("Rider"))


def test_intake_section_shows_totals_chart_and_the_fade_note(monkeypatch):
    from src.viz import training_page as tpg
    st = FakeSt(data_editor=lambda frame, **k: frame)
    monkeypatch.setattr(tpg, "st", st)
    charts = []
    monkeypatch.setattr(tpg, "_chart", lambda fig: charts.append(fig))
    s = logs([60, 70, 80, 90, 250, 240, 260, 300])
    s = s.assign(ride_id=range(len(s)), filename="r.fit", fluid_ml=np.nan)
    tpg._render_intake("Rider", s, "light")
    m = metrics(st)
    assert m["Rides logged"] == "8 / 8" and m["Below the guide"] == "4 / 8"
    assert len(charts) == 2 and "Fuel and fading" in texts(st) and "decoupling of" in texts(st)


def test_intake_section_without_logs_asks_for_them(monkeypatch):
    from src.viz import training_page as tpg
    st = FakeSt(data_editor=lambda frame, **k: frame)
    monkeypatch.setattr(tpg, "st", st)
    s = logs([np.nan, np.nan]).assign(ride_id=[1, 2], filename="r.fit", fluid_ml=np.nan)
    tpg._render_intake("Rider", s, "light")
    assert "No logs yet" in texts(st)


def test_short_rides_are_not_offered_for_logging(seeded, monkeypatch):
    from src.viz import training_page as tpg
    stats = db.get_ride_stats("Rider", seeded - pd.Timedelta(days=10), seeded)        # 40 minutes each
    st = FakeSt(data_editor=lambda frame, **k: frame)
    monkeypatch.setattr(tpg, "st", st)
    tpg._intake_editor(stats)
    assert not any(n == "data_editor" for n, a, k in st.shown)
