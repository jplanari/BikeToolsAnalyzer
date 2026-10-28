"""
Builds the analytics figures with real data. Plotly validates every property
at construction, so this catches API misuse. Skipped where plotly isn't
installed (CI installs it from requirements.txt).
"""
import numpy as np
import pandas as pd
import pytest

import os

try:
    import plotly.graph_objects as go
    _real_plotly = type(go).__name__ == "module"
except ImportError:
    _real_plotly = False
if not _real_plotly:
    if os.environ.get("CI"):
        # CI installs requirements.txt: a skip there would hide broken figures
        raise RuntimeError("plotly must be installed in CI to validate the figures")
    pytest.skip("plotly not installed", allow_module_level=True)

from src.analysis import ride_metrics as rm
from src.analysis import correlations as co
from src.viz import plots


@pytest.fixture
def hists():
    rng = np.random.default_rng(3)
    p = pd.Series(np.repeat(rng.uniform(100, 350, 90), 60))
    df = pd.DataFrame({'power': p, 'hr': 60 + 0.4 * p, 'cadence': 70 + 0.08 * p})
    h = rm.joint_histograms(df)
    to = lambda d: pd.DataFrame([(x, y, s) for (x, y), s in d.items()], columns=['x', 'y', 'seconds'])
    return to(h['p_hr']), to(h['p_cad'])


def _check(fig, min_traces=1):
    d = fig.to_dict()
    assert len(d['data']) >= min_traces
    fig.to_json()  # fully serialisable for st.plotly_chart
    return d


def test_joint_heatmap(hists):
    hr, cad = hists
    d = _check(plots.plot_joint_heatmap(co.aggregate(hr), 5, "Heart rate (bpm, 30 s avg)", "t",
                                        overlay=co.mean_y_by_power(hr)), 2)
    assert d['data'][0]['type'] == 'heatmap'
    _check(plots.plot_joint_heatmap(co.aggregate(cad), 5, "Cadence (rpm)", "t"))
    assert len(plots.plot_joint_heatmap(pd.DataFrame(columns=['x', 'y', 'seconds']), 5, "y", "t").data) == 0


def test_curves_and_trends(hists):
    hr, _ = hists
    _check(plots.plot_y_by_power({"A": co.mean_y_by_power(hr), "B": co.mean_y_by_power(hr)}, "bpm", "t"), 2)
    s = pd.Series([1.4, 1.45, 1.42, 1.5, 1.48, 1.52],
                  index=pd.date_range("2026-09-01", periods=6, freq="7D"))
    _check(plots.plot_trend({"EF": s}, "NP/HR", "t", rolling=3), 2)
    _check(plots.plot_trend({"HR": s}, "bpm", "t"))


def test_quadrants_and_zone_bars(hists):
    _, cad = hists
    agg = co.aggregate(cad)
    ref = co.weighted_median_y(cad)
    d = _check(plots.plot_quadrants(agg, 250, ref, 0.1725), 2)
    assert len(d['layout']['shapes']) == 2 and len(d['layout']['annotations']) == 4
    _check(plots.plot_cadence_by_zone(co.cadence_by_zone(cad, 250)))


# --- Training page charts --------------------------------------------------------

from src.analysis import training_load as tl, power_profile as pp, durability as du, intensity as it
from src.viz import training_plots as tp


@pytest.fixture(params=['light', 'dark'])
def mode(request):
    return request.param


@pytest.fixture
def pmc():
    days = pd.date_range('2026-06-01', '2026-10-28')
    hist = [{'date': d, 'tss': 60 + 40 * (d.dayofweek == 5)} for d in days]
    plan = [{'date': '2026-10-30', 'tss': 120}, {'date': '2026-11-02', 'tss': 40}]
    return tl.compute_pmc(hist, today='2026-10-28', planned=plan)


def test_training_load_charts(pmc, mode):
    today = pd.Timestamp('2026-10-28')
    d = _check(tp.plot_training_load(pmc, today, mode), 5)       # TSB + CTL/ATL actual + plan
    assert any(t.get('line', {}).get('dash') == 'dash' for t in d['data'])
    _check(tp.plot_ramp_rate(tl.ramp_rate(pmc), mode))
    _check(tp.plot_acwr(tl.acwr(pmc), mode))
    stats = pd.DataFrame({'date': pd.date_range('2026-09-01', periods=10, freq='3D'), 'moving_s': 3600,
                          'distance_km': 30.0, 'elevation_m': 300.0, 'kj': 700.0, 'tss': 60})
    _check(tp.plot_volume(tl.volume(stats), 'hours', 'Hours', mode))
    assert len(tp.plot_training_load(pmc.iloc[0:0], today, mode).data) == 0


def test_power_and_durability_charts(mode):
    recs = pd.DataFrame([{'duration': d, 'watts': 260 + 20000 / d, 'date': day, 'ride_id': i}
                         for i, day in enumerate(pd.date_range('2026-06-01', periods=12, freq='10D'))
                         for d in (5, 60, 300, 1200)])
    curves = pp.window_curves(recs, today='2026-10-28')
    _check(tp.plot_power_windows(curves, mode, weight=70), 4)
    _check(tp.plot_cp_history(pp.cp_history(recs, today='2026-10-28'), mode))
    table = pd.DataFrame({'kj': [1000, 1000, 2000], 'duration': [60, 300, 60],
                          'fresh': [400, 300, 400], 'fatigued': [360, 255, 320],
                          'retention_pct': [90, 85, 80]})
    _check(tp.plot_durability(table, mode), 2)


def test_intensity_charts(mode):
    rows = pd.DataFrame({'date': pd.to_datetime(['2026-10-05'] * 3 + ['2026-10-12'] * 3),
                         'zone': ['Z1', 'Z3', 'Z5'] * 2, 'seconds': [8000, 800, 1200, 6000, 3000, 1000]})
    dist = it.distribution(rows)
    d = _check(tp.plot_intensity_distribution(dist, mode), 3)
    assert d['layout']['barmode'] == 'stack'
    _check(tp.plot_polarization(dist, mode))


def test_ride_power_curve_with_recent_overlay():
    d = _check(plots.plot_power_curve({60: 400, 300: 300}, {60: 450, 300: 320}, {60: 420, 300: 310}), 3)
    assert [t['name'] for t in d['data']] == ['This Ride', 'All-Time Best', 'Last 90 Days Best']
