"""
Builds the analytics figures with real data. Plotly validates every property
at construction, so this catches API misuse. Skipped where plotly isn't
installed (CI installs it from requirements.txt).
"""
import numpy as np
import pandas as pd
import pytest

go = pytest.importorskip("plotly.graph_objects")
if not hasattr(go, "Figure") or type(go).__name__ == "MagicMock":
    pytest.skip("plotly is stubbed", allow_module_level=True)

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
