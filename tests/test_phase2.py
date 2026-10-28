import numpy as np
import pandas as pd
import pytest

from src.analysis import training_load as tl
from src.analysis import power_profile as pp
from src.analysis import durability as du
from src.analysis import intensity as it
from src.analysis.power import MMP_DURATIONS
from src.data import db
from src.data import strava as sv
from tests.test_strava import make_streams, activity, real_distances  # noqa: F401 (fixture)


# --- Training load -------------------------------------------------------------

def test_forecast_with_planned_sessions():
    history = [{'date': d, 'tss': 80} for d in pd.date_range('2026-08-01', '2026-10-28')]
    plan = [{'date': '2026-10-29', 'tss': 100}, {'date': '2026-10-30', 'tss': 0},
            {'date': '2026-10-28', 'tss': 999}]                       # today: ignored
    pmc = tl.compute_pmc(history, today='2026-10-28', planned=plan)
    assert pmc.index[-1] == pd.Timestamp('2026-10-30')
    assert pmc['planned'].tolist()[-3:] == [False, True, True]
    assert pmc.loc['2026-10-28', 'tss'] == 80
    assert pmc.loc['2026-10-29', 'tss'] == 100
    # the model just keeps running: day 30 is a rest day after the planned 100
    k = 1 - np.exp(-1 / 7)
    atl29 = pmc.loc['2026-10-29', 'ATL']
    assert pmc.loc['2026-10-30', 'ATL'] == pytest.approx(atl29 * (1 - k))
    assert pmc.loc['2026-10-30', 'TSB'] == pytest.approx(pmc.loc['2026-10-29', 'CTL'] - atl29)


def test_no_plan_keeps_old_behaviour():
    pmc = tl.compute_pmc([{'date': '2026-10-01', 'tss': 50}], until='2026-10-05', today='2026-10-05')
    assert len(pmc) == 5 and not pmc['planned'].any()


def test_ramp_rate_and_acwr_steady_state():
    days = pd.date_range('2025-01-01', periods=500)
    pmc = tl.compute_pmc([{'date': d, 'tss': 70} for d in days], until=days[-1], today=days[-1])
    assert tl.ramp_rate(pmc).iloc[-1] == pytest.approx(0, abs=0.05)
    assert tl.acwr(pmc).iloc[-1] == pytest.approx(1.0, abs=0.01)
    assert tl.acwr(pmc).iloc[:28].isna().all()                         # not enough history
    assert tl.acwr(pmc).iloc[28:].notna().all()


def test_ramp_rate_when_load_increases():
    days = pd.date_range('2026-01-01', periods=200)
    loads = [50] * 150 + [100] * 50
    pmc = tl.compute_pmc([{'date': d, 'tss': t} for d, t in zip(days, loads)],
                         until=days[-1], today=days[-1])
    assert tl.ramp_rate(pmc).loc['2026-06-07'] > 0


def test_monotony_and_strain():
    days = pd.date_range('2026-10-01', periods=7)
    pmc = tl.compute_pmc([{'date': d, 'tss': t} for d, t in zip(days, [100, 0, 100, 0, 100, 0, 100])],
                         until=days[-1], today=days[-1])
    ms = tl.monotony_strain(pmc).iloc[-1]
    loads = np.array([100, 0, 100, 0, 100, 0, 100])
    assert ms['monotony'] == pytest.approx(loads.mean() / loads.std())
    assert ms['strain'] == pytest.approx(400 * loads.mean() / loads.std())
    same = tl.compute_pmc([{'date': d, 'tss': 60} for d in days], until=days[-1], today=days[-1])
    assert np.isnan(tl.monotony_strain(same)['monotony'].iloc[-1])     # sd = 0


def test_weekly_and_monthly_volume():
    stats = pd.DataFrame({
        'date': pd.to_datetime(['2026-10-05 08:00', '2026-10-07 08:00', '2026-10-21 08:00']),
        'moving_s': [3600, 5400, 7200], 'distance_km': [30, 45, 60],
        'elevation_m': [300, 500, 800], 'kj': [700, 1000, 1500], 'tss': [50, 70, 100]})
    w = tl.volume(stats, 'W')
    assert list(w.index) == list(pd.to_datetime(['2026-10-05', '2026-10-12', '2026-10-19']))
    assert w.loc['2026-10-05', 'hours'] == 2.5 and w.loc['2026-10-05', 'rides'] == 2
    assert w.loc['2026-10-12'].sum() == 0                              # empty week kept
    m = tl.volume(stats, 'M')
    assert m.loc['2026-10-01', 'km'] == 135 and m.loc['2026-10-01', 'tss'] == 220


# --- Power profile -------------------------------------------------------------

def _records():
    return pd.DataFrame({
        'duration': [60, 60, 60, 1200, 1200],
        'watts': [400, 420, 420, 280, 300],
        'date': pd.to_datetime(['2026-01-10', '2026-09-01', '2026-10-01', '2026-01-10', '2026-10-20']),
        'ride_id': [1, 2, 3, 1, 4]})


def test_best_curve_keeps_date_and_ride_of_each_record():
    best = pp.best_curve(_records())
    row = best.set_index('duration')
    assert row.loc[60, 'watts'] == 420 and row.loc[60, 'ride_id'] == 2          # tie -> first set
    assert row.loc[1200, 'date'] == pd.Timestamp('2026-10-20')
    early = pp.best_curve(_records(), end='2026-06-30').set_index('duration')
    assert early.loc[1200, 'watts'] == 280


def test_window_curves():
    w = pp.window_curves(_records(), today='2026-10-28')
    assert list(w) == ['Last 42 days', 'Last 90 days', 'Season 2026', 'All time']
    assert pp.as_dict(w['Last 42 days']) == {60: 420.0, 1200: 300.0}
    assert pp.as_dict(w['All time'])[60] == 420


def test_cp3_recovers_model_parameters():
    curve = {d: pp._morton(d, 260, 18000, 1100) for d in MMP_DURATIONS}
    fit = pp.fit_cp3(curve)
    assert fit['cp'] == pytest.approx(260, abs=0.5)
    assert fit['w_prime'] == pytest.approx(18000, rel=0.01)
    assert fit['pmax'] == pytest.approx(1100, rel=0.02)
    assert pp.fit_cp3({60: 400, 300: 300}) is None


def test_cp_history_tracks_improvement():
    recs = []
    for i, day in enumerate(pd.date_range('2026-06-01', periods=20, freq='7D')):
        cp = 240 + i                                                    # +1 W per week
        for d in (180, 300, 600, 1200):
            recs.append({'duration': d, 'watts': cp + 20000 / d, 'date': day, 'ride_id': i})
    hist = pp.cp_history(pd.DataFrame(recs), window_days=28, today='2026-10-18')
    assert hist['cp'].dropna().is_monotonic_increasing
    assert hist['cp'].iloc[-1] == pytest.approx(259, abs=0.5)
    assert hist['ftp_20min'].iloc[-1] == pytest.approx(0.95 * (259 + 20000 / 1200))


def test_power_profile():
    prof = pp.power_profile({5: 1000, 60: 500, 300: 330, 1200: 280}, weight_kg=70, ftp=266)
    row = prof.set_index('duration')
    assert row.loc['5 min', 'w_per_kg'] == pytest.approx(330 / 70)
    assert row.loc['1 min', 'x_ftp'] == pytest.approx(500 / 266)
    assert pp.power_profile({}, 70, 250)['watts'].isna().all()


# --- Durability ------------------------------------------------------------------

def test_fatigued_curves():
    p = pd.Series([250.0] * 3600 + [200.0] * 3600)   # 900 kJ, then 720 kJ
    fc = du.fatigued_curves(p)
    assert set(fc) == {1000}                          # 2000 / 3000 kJ never reached
    assert fc[1000][60] == pytest.approx(200)         # only the 200 W part remains
    assert du.fatigued_curves(pd.Series([], dtype=float)) == {}


def test_durability_table():
    fresh = pd.DataFrame({'duration': [300, 300], 'watts': [320, 300],
                          'date': pd.to_datetime(['2026-10-01', '2026-10-10'])})
    tired = pd.DataFrame({'kj': [1000, 2000], 'duration': [300, 300], 'watts': [288, 256],
                          'date': pd.to_datetime(['2026-10-10', '2026-10-10'])})
    t = du.durability_table(fresh, tired).set_index('kj')
    assert t.loc[1000, 'retention_pct'] == pytest.approx(90)
    assert t.loc[2000, 'retention_pct'] == pytest.approx(80)
    later = du.durability_table(fresh, tired, start='2026-10-05').set_index('kj')
    assert later.loc[1000, 'retention_pct'] == pytest.approx(96)   # fresh best is 300 in window


# --- Intensity distribution --------------------------------------------------------

def test_polarization_index_matches_paper_example():
    assert it.polarization_index(80, 8, 12) == pytest.approx(np.log10(120))   # 2.08 > 2: polarized
    assert it.polarization_index(70, 20, 10) < 2                                # pyramidal
    assert it.polarization_index(20, 10, 70) is None                            # Z3 > Z1
    assert it.polarization_index(90, 10, 0) == 0.0
    assert it.polarization_index(85, 0, 15) == pytest.approx(np.log10(0.85 / 0.01 * 0.14 * 100))
    assert it.polarization_index(0, 0, 0) is None


def test_three_zone_mapping():
    assert it.three_zone({'Z1': 10, 'Z2': 20, 'Z3': 5, 'Z4': 5, 'Z5': 1, 'Z6': 1, 'Z7': 1}) == \
        {'low': 30, 'mid': 10, 'high': 3}
    assert it.zone_group('Z4 (Threshold)') == 'mid' and it.zone_group('Z6 (Anaerobic)') == 'high'


def test_weekly_distribution():
    rows = pd.DataFrame({
        'date': pd.to_datetime(['2026-10-05'] * 3 + ['2026-10-14'] * 2),
        'zone': ['Z1', 'Z3', 'Z5', 'Z2', 'Z4'],
        'seconds': [8000, 800, 1200, 3600, 3600]})
    d = it.distribution(rows)
    wk1 = d.loc['2026-10-05']
    assert (wk1['low'], wk1['mid'], wk1['high']) == pytest.approx((0.8, 0.08, 0.12))
    assert wk1['pi'] == pytest.approx(np.log10(120)) and wk1['hours'] == pytest.approx(10000 / 3600)
    assert d.loc['2026-10-12', 'pi'] == 0.0                                  # no high zone


# --- Storage ------------------------------------------------------------------------

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "test.db"))
    db.init_db()
    db.save_user("Rider", 250, 170, 70.0, 180.0)


def test_phase2_queries(temp_db):
    from src.data.pipeline import prepare_ride, summarize_ride, metrics_for_storage
    payload = sv.build_ride_payload(activity(5, "2026-10-01T07:00:00Z"), make_streams(n=6000, power=250.0))
    df, dist, ele = prepare_ride("strava_5.json", payload)
    s = summarize_ride(df, 250, 170)
    meta = db.build_ride_metadata(df, dist, ele, s['avg_speed_kph'], s['norm_power'], s['tss'], s['if'],
                                  filename="strava_5.json", metrics=metrics_for_storage(s))
    assert db.save_ride("Rider", "strava_5.json", payload, meta, s['power_curve'])[0]

    kj = db.get_kj_records("Rider")
    assert set(kj['kj']) == {1000}                      # 6000 s x 250 W = 1500 kJ
    assert kj.set_index('duration').loc[60, 'watts'] == pytest.approx(250)

    zones = db.get_zone_rows("Rider", 'power', "2026-10-01", "2026-10-01")
    assert zones.set_index('zone').loc['Z4', 'seconds'] == s['moving_time_s']
    assert db.get_zone_rows("Rider", 'power', "2026-11-01").empty

    recs = db.get_power_records("Rider")
    assert pp.as_dict(pp.best_curve(recs))[1200] == pytest.approx(250)
    assert db.get_ride_stats("Rider")['kj'].iloc[0] == pytest.approx(1500, rel=0.01)


# --- FTP suggestion ------------------------------------------------------------------

def _eff(day, watts20, ride_id=1):
    return [{'duration': d, 'watts': w, 'date': pd.Timestamp(day), 'ride_id': ride_id}
            for d, w in ((300, watts20 + 30), (600, watts20 + 15), (1200, watts20))]


def test_suggest_threshold_up_when_recent_20min_beats_ftp():
    recs = pd.DataFrame(_eff('2026-10-10', 300))
    s = pp.suggest_threshold(recs, current_ftp=250, today='2026-10-28')
    assert s['direction'] == 'up' and s['ftp'] == 285 and s['change_w'] == 35
    assert s['change_pct'] == pytest.approx(14.0) and s['best_20min_date'] == pd.Timestamp('2026-10-10')
    assert s['cp'] > 280                                         # CP from the 5/10/20 min points


def test_suggest_threshold_quiet_when_close_or_unsupported():
    close = pd.DataFrame(_eff('2026-10-10', 262))                # 0.95 * 262 = 248.9: within 3 %
    assert pp.suggest_threshold(close, 250, today='2026-10-28') is None
    old = pd.DataFrame(_eff('2026-05-01', 320))                  # outside the 90-day window
    assert pp.suggest_threshold(old, 250, today='2026-10-28') is None
    no20 = pd.DataFrame(_eff('2026-10-10', 300)).query('duration < 1200')
    assert pp.suggest_threshold(no20, 250, today='2026-10-28') is None
    assert pp.suggest_threshold(pd.DataFrame(), 250) is None and pp.suggest_threshold(close, 0) is None


def test_suggest_threshold_down_is_flagged_as_such():
    s = pp.suggest_threshold(pd.DataFrame(_eff('2026-10-10', 232)), 250, today='2026-10-28')
    assert s['direction'] == 'down' and s['ftp'] == 220
