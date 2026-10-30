"""Keeps a plotly figure's title, legend and plot area from overlapping.

`tidy(fig)` pins the title to the top-left corner of the figure, puts a horizontal legend directly
above the plot area (under the title) and grows the top margin to fit both, whatever the builder
asked for. `tidy_builders(namespace)` wraps every `plot_*` function of a module with it, so a new
chart is covered without remembering to call it.
"""
import functools
import math

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
