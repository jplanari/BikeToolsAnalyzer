import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import clean_channels
from src.analysis.power import power_curve, NP


def df(power=None, hr=None, cad=None, n=20):
    d = pd.DataFrame({'time': pd.date_range('2026-10-01', periods=n, freq='s')})
    if power is not None:
        d['power'] = power
    if hr is not None:
        d['hr'] = hr
    if cad is not None:
        d['cadence'] = cad
    return d


def test_lone_power_spike_is_interpolated_from_its_neighbours():
    p = [200.0] * 20
    p[10] = 6000.0
    out, fixed = clean_channels(df(power=p))
    assert fixed['power'] == 1 and out['power'].iloc[10] == pytest.approx(200)
    assert out['power'].max() == 200


def test_long_bad_stretch_becomes_zero_not_a_guess():
    p = [200.0] * 20
    p[5:12] = [9999.0] * 7
    out, fixed = clean_channels(df(power=p))
    assert fixed['power'] == 7
    assert out['power'].iloc[5:12].tolist() == [0.0] * 7
    assert out['power'].iloc[4] == 200 and out['power'].iloc[12] == 200


def test_glitch_at_the_edge_has_no_neighbour_and_becomes_zero():
    out, _ = clean_channels(df(power=[7000.0] + [200.0] * 4, n=5))
    assert out['power'].iloc[0] == 0


def test_real_hard_efforts_are_untouched():
    p = [200.0] * 5 + [2200.0, 2400.0] + [200.0] * 13               # a big sprint, still possible
    out, fixed = clean_channels(df(power=p))
    assert fixed['power'] == 0 and out['power'].tolist() == p


def test_negative_power_and_missing_values():
    p = [200.0, -50.0, 200.0, np.nan, 200.0]
    out, fixed = clean_channels(df(power=p, n=5))
    assert fixed['power'] == 1 and out['power'].iloc[1] == pytest.approx(200)
    assert np.isnan(out['power'].iloc[3])                            # real gaps stay gaps


def test_heart_rate_dropouts_are_dropped_cadence_too():
    hr = [140.0, 0.0, 141.0, 255.0, 142.0, np.nan] + [140.0] * 14
    cad = [90.0] * 5 + [400.0] + [90.0] * 14
    out, fixed = clean_channels(df(hr=hr, cad=cad))
    assert fixed['hr'] == 2 and fixed['cadence'] == 1
    assert out['hr'].isna().sum() == 3 and out['hr'].max() == 142
    assert np.isnan(out['cadence'].iloc[5])


def test_no_channels_and_input_not_modified():
    d = df(power=[100.0, 5000.0, 100.0], n=3)
    out, fixed = clean_channels(d)
    assert d['power'].iloc[1] == 5000                                # original untouched
    assert clean_channels(df(n=3))[1] == {'power': 0, 'hr': 0, 'cadence': 0}


def test_a_glitch_no_longer_becomes_the_one_second_record():
    p = [220.0] * 600
    p[300] = 8000.0
    cleaned, _ = clean_channels(df(power=p, n=600))
    assert power_curve(pd.Series(p))[1] == 8000
    assert power_curve(cleaned['power'])[1] == pytest.approx(220)
    assert NP(cleaned['power']) == pytest.approx(220, abs=1)


def test_pipeline_cleans_before_scoring(tmp_path):
    from src.data import strava as sv
    from src.data.pipeline import prepare_ride, summarize_ride
    from tests.test_strava import make_streams, activity, real_distances  # noqa: F401
    s = make_streams(n=1800, power=220.0)
    s['watts']['data'][900] = 9000.0
    s['heartrate']['data'][100] = 0.0
    payload = sv.build_ride_payload(activity(1), s)
    df, _, _ = prepare_ride("strava_1.json", payload)
    assert df['power'].max() == pytest.approx(220)
    summ = summarize_ride(df, 250, 170)
    assert summ['cleaned'] == {'power': 1, 'hr': 1, 'cadence': 0}
    assert summ['power_curve'][1] == pytest.approx(220)
