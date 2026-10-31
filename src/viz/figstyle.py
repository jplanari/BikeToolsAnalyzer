"""Keeps a plotly figure's title, legend and plot area from overlapping.

`tidy(fig)` pins the title to the top-left corner of the figure, puts a horizontal legend directly
above the plot area (under the title) and grows the top margin to fit both, whatever the builder
asked for. `tidy_builders(namespace)` wraps every `plot_*` function of a module with it, so a new
chart is covered without remembering to call it.
"""
import functools
import math

import numpy as np

TITLE_PX = 40            # room for a one-line title at the top of the figure
LEGEND_ROW_PX = 26       # one row of legend entries
BASE_TOP_PX = 16         # breathing room above everything
ASSUMED_WIDTH_PX = 700   # used only to guess how many rows a long legend wraps to


def _title_text(fig):
    title = fig.layout.title
    return getattr(title, "text", None) if title is not None else None


def _legend_names(fig):
    """Names that would appear in the legend, or [] when it is hidden."""
    if fig.layout.showlegend is False:
        return []
    names = [t.name for t in fig.data
             if getattr(t, "name", None) and getattr(t, "showlegend", None) is not False
             and getattr(t, "visible", True) is not False]
    if fig.layout.showlegend is None and len(names) < 2:      # plotly itself hides a one-trace legend
        return []
    return names


def _legend_rows(names):
    width = sum(8 * len(str(n)) + 50 for n in names)
    return max(1, math.ceil(width / ASSUMED_WIDTH_PX))


def tidy(fig):
    """Title top-left, legend just above the plot, margin big enough for both. Idempotent."""
    title = _title_text(fig)
    names = _legend_names(fig)
    if not title and not names:
        return fig

    top = BASE_TOP_PX
    if title:
        fig.update_layout(title=dict(text=title, x=0, xref="paper", xanchor="left",
                                     y=1, yref="container", yanchor="top", pad=dict(t=6, b=0)))
        top += TITLE_PX
    if names:
        fig.update_layout(legend=dict(orientation="h", x=0, xref="paper", xanchor="left",
                                      y=1, yref="paper", yanchor="bottom", title=dict(text="")))
        top += LEGEND_ROW_PX * _legend_rows(names)
    fig.update_layout(margin=dict(t=top))
    return fig


def tidy_builders(namespace):
    """Wrap every `plot_*` function in `namespace` (a module's globals()) so it returns tidy(figure)."""
    for name, fn in list(namespace.items()):
        if name.startswith("plot_") and callable(fn) and not getattr(fn, "_tidy", False):
            namespace[name] = _wrapped(fn)


def _wrapped(fn):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        fig = fn(*args, **kwargs)
        return tidy(fig) if hasattr(fig, "update_layout") else fig
    inner._tidy = True
    return inner


ELEVATION_PAD = 0.05            # fraction of the elevation span added below the lowest and above the highest point
ELEVATION_MIN_SPAN_M = 10.0     # a flat ride still gets a readable axis


def elevation_range(*series, pad=ELEVATION_PAD, min_span=ELEVATION_MIN_SPAN_M):
    """
    [low, high] for an elevation axis: the lowest point minus `pad` of the span to the highest point plus `pad`
    of the span, so the profile fills the plot instead of starting at 0 m. Several series (a whole ride and a
    selection of it) share one range. None when there is no elevation data.
    """
    arrays = [np.asarray(s, dtype=float) for s in series if s is not None]
    arrays = [a[np.isfinite(a)] for a in arrays]
    arrays = [a for a in arrays if a.size]
    if not arrays:
        return None
    lo, hi = min(a.min() for a in arrays), max(a.max() for a in arrays)
    if hi - lo < min_span:
        mid = (hi + lo) / 2
        lo, hi = mid - min_span / 2, mid + min_span / 2
    margin = pad * (hi - lo)
    return [float(lo - margin), float(hi + margin)]
