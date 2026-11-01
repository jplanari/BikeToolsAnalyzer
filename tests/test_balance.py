"""Left/right leg balance: reading the FIT field, per-ride numbers, stored histograms, the period analytics and PDF."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import balance as bal
from src.analysis import ride_metrics as rm
from src.data import db
from src.data.processing.pipeline import prepare_ride, summarize_ride, metrics_for_storage
from src.data.providers import strava as sv
from tests.test_strava import make_streams, activity, real_distances  # noqa: F401 (fixture)


# --- the FIT field ----------------------------------------------------------------------------------------------

def test_right_percent_follows_the_fit_flag_and_drops_missing_values():
    raw = pd.Series([0x80 | 52, 0x80 | 48, 45, 0, 128, 255, 'right', None, 0x80 | 101])
    out = bal.right_percent(raw)
    assert out.iloc[0] == 52 and out.iloc[1] == 48          # flag set: the percentage is the right pedal's
    assert out.iloc[2] == 55                                # flag clear: it is the left pedal's
    assert np.isnan(out.iloc[3]) and np.isnan(out.iloc[4]) and np.isnan(out.iloc[6]) and np.isnan(out.iloc[7])
    assert np.isnan(out.iloc[8])                            # more than 100 %: a bad reading


def test_pedal_percent_channels_reject_nonsense():
    out = bal.percent_channel(pd.Series([62.5, 0, 140, None, 'x']))
    assert out.iloc[0] == 62.5 and out.iloc[1:].isna().all()


def test_parse_fit_exposes_the_balance_channels(monkeypatch):
    from src.data.parsers import fit

    class Field:
        def __init__(self, name, value):
            self.name, self.value = name, value

    class FakeFile:
        def __init__(self, *a, **k):
            pass

        def get_messages(self, kind):
            base = pd.Timestamp('2026-10-01 08:00:00')
            for i in range(5):
                yield [Field('timestamp', (base + pd.Timedelta(seconds=i)).to_pydatetime()), Field('power', 200),
                       Field('left_right_balance', 0x80 | 47), Field('left_torque_effectiveness', 71.0),
                       Field('right_pedal_smoothness', 22.5)]

    monkeypatch.setattr(fit.fitparse, 'FitFile', FakeFile)
    df = fit.parse_fit.__wrapped__(b'x') if hasattr(fit.parse_fit, '__wrapped__') else fit.parse_fit(b'x')
    assert list(df['lr_balance']) == [47.0] * 5
    assert list(df['te_l']) == [71.0] * 5 and list(df['ps_r']) == [22.5] * 5
    assert 'te_r' not in df.columns                          # only what the file has


# --- one ride --------------------------------------------------------------------------------------------------

def _ride(n=3 * 3600 // 3, right=51.0, late=None):
    power = np.full(n, 200.0)
    balance = np.full(n, right)
    if late is not None:
        balance[-n // 3:] = late
    return pd.DataFrame({'power': power, 'cadence': 90.0, 'lr_balance': balance, 'te_l': 70.0, 'te_r': 75.0})


def test_ride_metrics_are_power_weighted_and_show_drift():
    df = _ride(right=50.0, late=56.0)
    m = bal.ride_metrics(df)
    assert 51.0 < m['bal_right_pct'] < 52.5
    assert m['bal_first_pct'] == pytest.approx(50.0) and m['bal_last_pct'] == pytest.approx(56.0)
    assert m['te_l'] == 70.0 and m['te_r'] == 75.0 and m['bal_s'] == len(df)
    heavy = df.copy()
    heavy.loc[heavy.index[-200:], 'power'] = 1000.0          # hard seconds count more than easy ones
    assert bal.ride_metrics(heavy)['bal_right_pct'] > m['bal_right_pct']


def test_short_rides_have_no_drift_and_rides_without_balance_have_nothing():
    m = bal.ride_metrics(_ride(n=600))
    assert m['bal_right_pct'] == pytest.approx(51.0) and m['bal_first_pct'] is None
    assert all(v is None for v in bal.ride_metrics(_ride().drop(columns='lr_balance')).values())


def test_verdict_and_leg_watts():
    assert bal.verdict(50.5) == 'even' and bal.verdict(53) == 'slight' and bal.verdict(56) == 'uneven'
    assert bal.verdict(None) is None
    assert bal.leg_watts(200, 52) == pytest.approx((96.0, 104.0))
    assert bal.split_text(52.0) == '48 / 52'


# --- histograms and the period ----------------------------------------------------------------------------------

def _cells(hist):
    return pd.DataFrame([(x, y, s) for (x, y), s in hist.items()], columns=['x', 'y', 'seconds'])


def test_histograms_hold_power_and_cadence_against_balance():
    df = _ride(n=1200)
    h = rm.joint_histograms(df)
    assert sum(h['p_bal'].values()) == 1200 and set(h['p_bal']) == {(200, 50)}      # 51 % falls in the 50-52 bin
    assert set(h['c_bal']) == {(90, 50)}
    assert 'p_bal' not in rm.joint_histograms(df.drop(columns='lr_balance'))


def test_period_analytics_from_cells():
    # the right leg takes more as power rises: 50 % at 150 W, 54 % at 350 W
    hist = {}
    for power, right in ((150, 50), (200, 51), (250, 52), (300, 53), (350, 54)):
        hist[(power, right - right % 2)] = 600
    c = _cells(hist)
    assert 51 < bal.right_share(c) < 54
    assert bal.right_share(_cells({(200, 50): 300})) is None                          # under 10 minutes
    assert bal.intensity_slope(c) == pytest.approx(2.0, abs=0.6)                      # about +2 pts per 100 W
    assert bal.mean_power(c) == pytest.approx(255, abs=1)
    zones = bal.by_zone(c, 250)
    assert list(zones['zone'])[0] < list(zones['zone'])[-1] and zones['right'].is_monotonic_increasing
    assert 0 <= bal.time_even(c) <= 1 and bal.spread(c) > 0
    assert len(bal.by_power(c)) == 5


def test_cadence_view():
    c = _cells({(60, 48): 400, (90, 50): 400, (110, 54): 400})
    t = bal.by_cadence(c)
    assert list(t['cadence']) == [62.5, 92.5, 112.5] and t['mean'].is_monotonic_increasing


# --- stored and reported ----------------------------------------------------------------------------------------

@pytest.fixture
def seeded(tmp_path, monkeypatch, real_distances):
    monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "b.db"))
    db.init_db()
    db.save_user("Rider", 250, 170, 70.0, 180.0)
    today = pd.Timestamp.today().normalize()
    for i in range(6):
        streams = make_streams(n=2400, power=180.0 + 10 * i, hr=130.0)
        payload = sv.build_ride_payload(activity(500 + i, f"{(today - pd.Timedelta(days=i + 1)):%Y-%m-%d}T07:00:00Z"),
                                        streams)
        name = sv.ride_filename(500 + i)
        df, dist, ele = prepare_ride(name, payload)
        df['lr_balance'] = 49.0 + (i % 3)
        s = summarize_ride(df, 250, 170)
        meta = db.build_ride_metadata(df, dist, ele, s['avg_speed_kph'], s['norm_power'], s['tss'], s['if'],
                                      filename=name, metrics=metrics_for_storage(s))
        ok, msg = db.save_ride("Rider", name, payload, meta, s['power_curve'])
        assert ok, msg
    return today


def test_balance_is_stored_per_ride_and_in_histograms(seeded):
    start = seeded - pd.Timedelta(days=10)
    stats = db.get_ride_stats("Rider", start, seeded)
    assert stats['bal_right_pct'].between(48, 52).all() and stats['bal_s'].min() > 1000
    assert not db.get_hist("Rider", 'p_bal', start, seeded).empty
    assert db.period_coverage("Rider", start, seeded)['p_bal'] == 6


def test_deeper_data_and_pdf_get_a_balance_page_only_with_data(seeded, tmp_path, monkeypatch):
    from src.data.reports.deeper_data import collect_deeper_data
    from src.viz.deeper_pdf import deeper_figures
    data = collect_deeper_data("Rider", seeded - pd.Timedelta(days=10), seeded)
    b = data['balance']
    assert b and 48 < b['right'] < 52 and b['by_zone'] is not None and not b['curve'].empty
    assert len(deeper_figures(data)) == 4
    data['balance'] = None
    assert len(deeper_figures(data)) == 3
