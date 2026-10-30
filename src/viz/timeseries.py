"""Time series section: pick any of the ride's channels and see them stacked under the elevation profile."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# label -> (column, unit, colour). Colours are the Okabe-Ito palette used across the app.
CHANNELS = {
    "Power": ("power", "W", "#0072B2"),
    "Heart rate": ("hr", "bpm", "#D55E00"),
    "Speed": ("speed_kmh", "km/h", "#009E73"),
    "Cadence": ("cadence", "rpm", "#CC79A7"),
    "Gradient": ("grade_smooth", "%", "#E69F00"),
    "W′ balance": ("w_prime_balance", "J", "#56B4E9"),
}
SMOOTHING_S = 5                  # moving-average window, seconds (samples at 1 Hz)
MAX_POINTS = 3000
BASE_FILL = "rgba(128,128,128,0.25)"
BASE_LINE = "rgba(128,128,128,0.7)"


def available(df):
    """Channels this ride has data for, in display order."""
    return [name for name, (col, _, _) in CHANNELS.items() if col in df.columns and df[col].notna().any()]


def _x(df):
    if 'dist_m' in df.columns and df['dist_m'].notna().any():
        return df['dist_m'] / 1000.0, "Distance (km)"
    if 'time' in df.columns:                       # no distance (indoor, no GPS): elapsed minutes
        t = pd.to_datetime(df['time'])
        return (t - t.iloc[0]).dt.total_seconds() / 60.0, "Time (min)"
    return pd.Series(np.arange(len(df)) / 60.0, index=df.index), "Time (min)"


def _thin(x, y):
    """At most MAX_POINTS points: mean per bin, so spikes are not skipped by plain striding."""
    step = int(np.ceil(len(x) / MAX_POINTS))
    if step <= 1:
        return x.to_numpy(), y.to_numpy()
    g = np.arange(len(x)) // step
    return (pd.Series(x.to_numpy()).groupby(g).mean().to_numpy(),
            pd.Series(y.to_numpy()).groupby(g).mean().to_numpy())


def _smooth(s, window):
    return s.rolling(window, min_periods=1, center=True).mean() if window > 1 else s


def build_figure(df, series):
    """
    series: labels from CHANNELS. One row per channel under the elevation profile, all sharing the distance axis
    (time when the file has no distance), so hovering anywhere shows every channel at that point of the route.
    """
    x, x_title = _x(df)
    series = [s for s in series if s in CHANNELS and CHANNELS[s][0] in df.columns]
    has_ele = 'ele' in df.columns and df['ele'].notna().any()

    rows = (1 if has_ele else 0) + len(series)
    heights = ([0.8] if has_ele else []) + [1.0] * len(series)
    fig = make_subplots(rows=max(rows, 1), cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[h / sum(heights) for h in heights] if heights else None)
    r = 1
    if has_ele:
        bx, by = _thin(x, df['ele'])
        fig.add_trace(go.Scatter(x=bx, y=by, mode='lines', name="Elevation", fill='tozeroy', fillcolor=BASE_FILL,
                                 line=dict(color=BASE_LINE, width=1), hovertemplate="%{y:.0f} m"), row=r, col=1)
        fig.update_yaxes(title_text="Elevation (m)", row=r, col=1)
        r += 1
    for label in series:
        col, unit, color = CHANNELS[label]
        lx, ly = _thin(x, _smooth(df[col], SMOOTHING_S))
        fig.add_trace(go.Scatter(x=lx, y=ly, mode='lines', name=label, line=dict(color=color, width=1.6),
                                 hovertemplate="%{y:.1f} " + unit), row=r, col=1)
        fig.update_yaxes(title_text=f"{label} ({unit})", row=r, col=1)
        r += 1
    fig.update_xaxes(title_text=x_title, row=max(rows, 1), col=1)
    fig.update_yaxes(fixedrange=True)                     # zoom and pan only along the route: every channel keeps its full range
    fig.update_layout(height=140 * max(rows, 1) + 120, showlegend=False, hovermode="x unified", dragmode="zoom",
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig


# Plotly toolbar for the chart: drag a box (or use the wheel) to zoom along the route, double-click to reset;
# the selection tools have nothing to select here.
PLOT_CONFIG = {"scrollZoom": True, "displaylogo": False,
               "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]}
