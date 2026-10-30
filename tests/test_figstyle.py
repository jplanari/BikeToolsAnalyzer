"""Titles, legends and plot areas must never overlap. Skipped where plotly is missing (CI has it)."""
import os

import numpy as np
import pandas as pd
import pytest

try:
    import plotly.graph_objects as go
    _real = type(go).__name__ == "module"
except ImportError:
    _real = False
if not _real:
    if os.environ.get("CI"):
        raise RuntimeError("plotly must be installed in CI to validate the figures")
    pytest.skip("plotly not installed", allow_module_level=True)

from src.viz import figstyle, plots, training_plots as tp


def _two_lines(title="Fitness, fatigue and form"):
    fig = go.Figure([go.Scatter(x=[1, 2], y=[1, 2], name="Fitness (CTL)"),
                     go.Scatter(x=[1, 2], y=[2, 1], name="Fatigue (ATL)")])
    fig.update_layout(title=title, legend=dict(orientation="h", y=1.12))     # the old, overlapping setup
    return fig


def test_title_and_legend_get_their_own_bands_above_the_plot():
    fig = figstyle.tidy(_two_lines())
    assert fig.layout.title.yref == "container" and fig.layout.title.yanchor == "top"
    assert fig.layout.legend.yref == "paper" and fig.layout.legend.y == 1 and fig.layout.legend.yanchor == "bottom"
    assert fig.layout.margin.t >= figstyle.BASE_TOP_PX + figstyle.TITLE_PX + figstyle.LEGEND_ROW_PX


def test_tidy_is_idempotent_and_keeps_other_margins():
    fig = _two_lines()
    fig.update_layout(margin=dict(l=7, r=9, b=11))
    once = figstyle.tidy(fig).layout.margin.t
    assert figstyle.tidy(fig).layout.margin.t == once
    assert (fig.layout.margin.l, fig.layout.margin.r, fig.layout.margin.b) == (7, 9, 11)


def test_figures_without_title_or_legend_are_left_alone():
    fig = go.Figure(go.Scatter(x=[1, 2], y=[1, 2]))
    fig.update_layout(margin=dict(t=10))
    assert figstyle.tidy(fig).layout.margin.t == 10


def test_a_hidden_legend_costs_no_room():
    fig = _two_lines()
    fig.update_layout(showlegend=False)
    assert figstyle.tidy(fig).layout.margin.t == figstyle.BASE_TOP_PX + figstyle.TITLE_PX


def test_long_legends_get_extra_rows():
    fig = go.Figure([go.Scatter(x=[1], y=[1], name=f"A rather long series name {i}") for i in range(8)])
    fig.update_layout(title="t")
    assert figstyle.tidy(fig).layout.margin.t > figstyle.BASE_TOP_PX + figstyle.TITLE_PX + figstyle.LEGEND_ROW_PX


def test_every_training_figure_is_tidied():
    days = pd.date_range("2026-09-01", periods=40)
    pmc = pd.DataFrame({'CTL': np.linspace(40, 50, 40), 'ATL': np.linspace(50, 60, 40),
                        'TSB': np.linspace(-10, 5, 40), 'planned': False}, index=days)
    fig = tp.plot_training_load(pmc, days[-1])
    assert fig.layout.title.yref == "container"
    assert fig.layout.margin.t >= figstyle.BASE_TOP_PX + figstyle.TITLE_PX + figstyle.LEGEND_ROW_PX


def test_plot_builders_are_wrapped_once():
    assert getattr(plots.plot_power_curve, "_tidy", False) and getattr(tp.plot_ramp_rate, "_tidy", False)
    fig = plots.plot_zone_distribution({"Z1": 10, "Z2": 20})
    assert fig.layout.title.yref == "container"
