"""Energy and fuelling estimates: the model, the stored per-ride number, period totals and the weekly line."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import nutrition as nut
from src.data import db
from src.data.processing.pipeline import prepare_ride, summarize_ride, metrics_for_storage
from src.data.providers import strava as sv
from tests.test_strava import make_streams, activity, real_distances  # noqa: F401 (fixture)


def hour(watts):
    return pd.Series([float(watts)] * 3600)


def test_kcal_is_about_kj():
    assert nut.kcal_from_kj(1000) == pytest.approx(996, abs=1)
    assert nut.kcal_from_kj(None) is None and nut.kcal_from_kj(float('nan')) is None


def test_carbohydrate_share_rises_with_intensity_and_is_bounded():
    shares = [nut.carb_share(x) for x in (0.0, 0.5, 0.7, 0.9, 1.0, 1.3, 2.0)]
    assert shares == sorted(shares) and 0.2 < shares[0] < 0.3 and shares[-1] == 1.0


def test_ride_energy_energy_adds_up():
    e = nut.ride_energy(hour(200), 250)
    assert e['kj'] == pytest.approx(720) and e['kcal'] == pytest.approx(717, abs=1)
    # carbohydrate and fat together account for all the energy
    assert e['carb_g'] * nut.CARB_KCAL_PER_G + e['fat_g'] * nut.FAT_KCAL_PER_G == pytest.approx(e['kcal'])
    assert 0.6 < e['carb_share'] < 0.7                      # 80 % of FTP


def test_harder_riding_burns_more_carbohydrate_per_kcal_and_threshold_is_all_carbohydrate():
    easy, hard = nut.ride_energy(hour(125), 250), nut.ride_energy(hour(320), 250)
    assert hard['carb_share'] > easy['carb_share'] and hard['fat_g'] == pytest.approx(0)


def test_without_power_or_ftp_there_is_no_estimate():
    assert nut.ride_energy(None, 250) is None and nut.ride_energy(hour(0), 250) is None
    assert nut.ride_energy(hour(200), 0) is None
    assert nut.ride_metrics(None, 250) == {'carb_g': None}


def test_coasting_is_not_energy_and_bad_samples_are_ignored():
    p = pd.Series([200.0, np.nan, -5.0, 0.0] * 900)
    assert nut.ride_energy(p, 250)['kj'] == pytest.approx(180.0)


def test_fuelling_guide_grows_with_ride_length_and_follows_intensity():
    assert nut.target_rate(30 * 60) == (0, 0)
    assert nut.target_rate(50 * 60)[1] == 30
    assert nut.target_rate(90 * 60) == (30, 60)
    assert nut.target_rate(4 * 3600) == (60, 90)
    assert nut.target_rate(90 * 60, 0.5) == (15, 30)                   # very easy: half
    assert nut.target_rate(90 * 60, 0.9)[0] > nut.target_rate(90 * 60, 0.7)[0]   # hard: aim higher
    low, high = nut.target_total(2 * 3600 + 1)
    assert 90 < low < high


def test_glycogen_store_and_load_words():
    assert nut.glycogen_store(70) == pytest.approx(420) and nut.glycogen_store(None) is None
    assert nut.store_used(210, 70) == pytest.approx(0.5)
    assert nut.store_used(210, 70, eaten_g=100) == pytest.approx(110 / 420)
    assert nut.store_used(100, 70, eaten_g=500) == 0.0 and nut.store_used(None, 70) is None
    assert [nut.load(x) for x in (0.1, 0.3, 0.6, 1.2)] == ['light', 'moderate', 'heavy', 'extreme']
    assert nut.load(None) is None


def test_fat_grams_matches_the_energy_left_after_carbohydrate():
    assert nut.fat_grams(1000, 0) == pytest.approx(nut.kcal_from_kj(1000) / 9)
    assert nut.fat_grams(100, 10_000) == 0.0 and nut.fat_grams(None, 5) is None


# --- periods ----------------------------------------------------------------------------------------------------

def rides(days, kj, carb, hours=1.0):
    return pd.DataFrame({'date': pd.to_datetime(days), 'kj': kj, 'carb_g': carb,
                         'moving_s': [hours * 3600] * len(days), 'duration_s': [hours * 3600] * len(days)})


def test_weekly_totals_group_by_week_and_keep_empty_weeks():
    s = rides(['2026-10-05', '2026-10-07', '2026-10-26'], [800, 600, 1000], [150, 100, 200])
    w = nut.by_period(s, 'W')
    assert list(w['rides']) == [2, 0, 0, 1] and w['kcal'].iloc[0] == pytest.approx(1400 * nut.KCAL_PER_KJ)
    assert w['carb_g'].iloc[0] == 250 and w['hours'].iloc[0] == 2.0
    assert 0 < w['carb_pct'].iloc[0] < 100 and np.isnan(w['carb_pct'].iloc[1])
    m = nut.by_period(s, 'M')
    assert list(m['rides']) == [3]


def test_rides_without_carbohydrate_count_for_energy_only():
    s = rides(['2026-10-05', '2026-10-06'], [1000, 1000], [None, 200])
    w = nut.by_period(s, 'W')
    assert w['kcal'].iloc[0] == pytest.approx(2000 * nut.KCAL_PER_KJ) and w['carb_g'].iloc[0] == 200
    # the split only covers the ride that has one
    assert w['carb_pct'].iloc[0] == pytest.approx(200 * 4 / (1000 * nut.KCAL_PER_KJ) * 100)
    assert nut.totals(s)['carb_g'] == 200


def test_timezone_aware_dates_and_power_less_rides():
    s = rides(['2026-10-05T08:00:00+02:00', '2026-10-06T08:00:00+02:00'], [0, 500], [None, 90])
    assert nut.by_period(s, 'W')['rides'].iloc[0] == 1             # the ride without work is left out


def test_empty_periods_are_handled():
    assert nut.by_period(pd.DataFrame(), 'W').empty and nut.by_period(None).empty
    t = nut.totals(pd.DataFrame())
    assert t['kcal'] is None and t['rides'] == 0


# --- stored with the ride -----------------------------------------------------------------------------------------

@pytest.fixture
def seeded(tmp_path, monkeypatch, real_distances):
    monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "n.db"))
    db.init_db()
    db.save_user("Rider", 250, 170, 70.0, 180.0)
    today = pd.Timestamp.today().normalize()
    for i in range(4):
        streams = make_streams(n=2400, power=180.0 + 20 * i, hr=130.0)
        payload = sv.build_ride_payload(activity(700 + i, f"{(today - pd.Timedelta(days=i + 1)):%Y-%m-%d}T07:00:00Z"),
                                        streams)
        name = sv.ride_filename(700 + i)
        df, dist, ele = prepare_ride(name, payload)
        s = summarize_ride(df, 250, 170)
        meta = db.build_ride_metadata(df, dist, ele, s['avg_speed_kph'], s['norm_power'], s['tss'], s['if'],
                                      filename=name, metrics=metrics_for_storage(s))
        ok, msg = db.save_ride("Rider", name, payload, meta, s['power_curve'])
        assert ok, msg
    return today


def test_carbohydrate_is_stored_per_ride_and_harder_rides_burn_more(seeded):
    stats = db.get_ride_stats("Rider", seeded - pd.Timedelta(days=10), seeded)
    assert stats['carb_g'].notna().all() and (stats['carb_g'] > 0).all()
    ordered = stats.sort_values('kj')
    assert ordered['carb_g'].is_monotonic_increasing
    assert nut.totals(stats)['rides'] == 4


def test_the_ride_summary_carries_the_estimate():
    df = pd.DataFrame({'time': pd.date_range('2026-01-01', periods=3600, freq='s'), 'power': 200.0,
                       'speed': 8.0, 'hr': 140.0})
    assert nut.ride_metrics(df['power'], 250)['carb_g'] == pytest.approx(nut.ride_energy(df['power'], 250)['carb_g'])


def test_weekly_email_mentions_the_energy():
    from src.data.reports import weekly_report as wr
    base = {'rides': 2, 'distance_km': 80, 'ascent_m': 900, 'hours': 3.0, 'tss': 150, 'pages': 5}
    monday, sunday = pd.Timestamp('2026-10-26'), pd.Timestamp('2026-11-01')
    txt = wr.body_text("Rider", monday, sunday, {**base, 'kcal': 2400, 'carb_g': 380})
    assert "2,400 kcal" in txt and "380 g carbohydrate" in txt
    assert "kcal burned on the bike." in wr.body_text("Rider", monday, sunday, {**base, 'kcal': 2400, 'carb_g': 0})
    assert "kcal" not in wr.body_text("Rider", monday, sunday, {**base, 'kcal': 0, 'carb_g': 0})


# --- the screens (streamlit is a stub in tests: record what they would show) ------------------------------------------

class Recorder:
    def __init__(self):
        self.shown = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.shown.append((name, args, kwargs))
            if name == "columns":
                n = args[0] if isinstance(args[0], int) else len(args[0])
                return [Recorder() for _ in range(n)]
            return self
        return call

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def texts(rec):
    out = []
    for name, args, kwargs in rec.shown:
        out.extend(str(a) for a in args if isinstance(a, str))
    return " | ".join(out)


def test_ride_summary_shows_energy_fuel_and_load(monkeypatch):
    from src.viz import fuel as ra
    rec = Recorder()
    monkeypatch.setattr(ra, "st", rec)
    e = nut.ride_energy(hour(200), 250)
    summary = {'kj': e['kj'], 'carb_g': e['carb_g'], 'moving_time_s': 3 * 3600, 'if': 0.8}
    ra.render_fuel_summary(summary, 70.0)
    shown = texts(rec)
    assert "Energy and fuel" in shown and "carbohydrate load" in shown
    caption = [a[0] for n, a, k in rec.shown if n == "caption"][0]
    assert "glycogen" in caption and "%" in caption and "about 0%" not in caption


def test_a_guide_that_replaces_everything_is_worded_as_such(monkeypatch):
    from src.viz import fuel
    rec = Recorder()
    monkeypatch.setattr(fuel, "st", rec)
    fuel.render_fuel_summary({'kj': 700.0, 'carb_g': 100.0, 'moving_time_s': 4 * 3600, 'if': 0.7}, 70.0)
    assert "covers it" in texts(rec)


def test_ride_summary_for_a_short_ride_has_no_target(monkeypatch):
    from src.viz import fuel as ra
    rec = Recorder()
    monkeypatch.setattr(ra, "st", rec)
    ra.render_fuel_summary({'kj': 300.0, 'carb_g': 40.0, 'moving_time_s': 1800, 'if': 0.6}, 70.0)
    assert any(n == "caption" and "In total" not in a[0] for n, a, k in rec.shown)


def test_fuel_section_renders_for_a_seeded_history(seeded, monkeypatch):
    from src.viz import training_page as tpg
    rec = Recorder()
    monkeypatch.setattr(tpg, "st", rec)
    monkeypatch.setattr(tpg, "remembered", lambda fn, key, default, *a, **k: default)
    monkeypatch.setattr(tpg, "_chart", lambda fig: rec.shown.append(("chart", (), {})))
    tpg._render_fuel("Rider", db.get_user_data("Rider"), seeded, "light")
    assert any(n == "chart" for n, a, k in rec.shown)
    assert "Heaviest" in texts(rec) and "kcal" in texts(rec)


def test_fuel_section_without_rides_says_so(tmp_path, monkeypatch):
    from src.viz import training_page as tpg
    monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "e.db"))
    db.init_db()
    db.save_user("Rider", 250, 170, 70.0, 180.0)
    rec = Recorder()
    monkeypatch.setattr(tpg, "st", rec)
    monkeypatch.setattr(tpg, "remembered", lambda fn, key, default, *a, **k: default)
    tpg._render_fuel("Rider", db.get_user_data("Rider"), pd.Timestamp.today().normalize(), "light")
    assert "No rides with power" in texts(rec)
