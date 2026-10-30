"""Time series section: pick any of the ride's channels and draw them over a base elevation profile."""
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
SMOOTHING = {"None": 1, "5 s": 5, "10 s": 10, "30 s": 30}
LAYOUTS = ["Overlay", "Stacked"]
X_AXES = ["Distance", "Time"]
MAX_POINTS = 3000
AXIS_STEP = 0.07                 # paper width taken by each extra y axis in the overlay layout
BASE_FILL = "rgba(128,128,128,0.25)"
BASE_LINE = "rgba(128,128,128,0.7)"


def available(df):
    """Channels this ride has data for, in display order."""
    return [name for name, (col, _, _) in CHANNELS.items() if col in df.columns and df[col].notna().any()]


def _x(df, x_mode):
    if x_mode == "Time" and 'time' in df.columns:
        t = pd.to_datetime(df['time'])
        return (t - t.iloc[0]).dt.total_seconds() / 60.0, "Time (min)"
    if 'dist_m' in df.columns:
        return df['dist_m'] / 1000.0, "Distance (km)"
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


def build_figure(df, series, x_mode="Distance", layout="Overlay", smoothing="5 s"):
    """
    series: labels from CHANNELS. Overlay: one chart, the elevation as the shaded base on the left axis and each
    channel on its own colour-matched axis on the right. Stacked: elevation on top, one row per channel, shared x axis.
    """
    window = SMOOTHING.get(smoothing, 1)
    x, x_title = _x(df, x_mode)
    series = [s for s in series if s in CHANNELS and CHANNELS[s][0] in df.columns]
    has_ele = 'ele' in df.columns and df['ele'].notna().any()

    def base_trace(**kw):
        bx, by = _thin(x, df['ele'])
        return go.Scatter(x=bx, y=by, mode='lines', name="Elevation", fill='tozeroy', fillcolor=BASE_FILL,
                          line=dict(color=BASE_LINE, width=1), hovertemplate="%{y:.0f} m", **kw)

    def line_trace(label, **kw):
        col, unit, color = CHANNELS[label]
        lx, ly = _thin(x, _smooth(df[col], window))
        return go.Scatter(x=lx, y=ly, mode='lines', name=label, line=dict(color=color, width=1.6),
                          hovertemplate="%{y:.1f} " + unit, **kw)

    if layout == "Stacked":
        rows = (1 if has_ele else 0) + len(series)
        heights = ([0.8] if has_ele else []) + [1.0] * len(series)
        fig = make_subplots(rows=max(rows, 1), cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            row_heights=[h / sum(heights) for h in heights] if heights else None)
        r = 1
        if has_ele:
            fig.add_trace(base_trace(), row=r, col=1)
            fig.update_yaxes(title_text="Elevation (m)", row=r, col=1)
            r += 1
        for label in series:
            fig.add_trace(line_trace(label), row=r, col=1)
            fig.update_yaxes(title_text=f"{label} ({CHANNELS[label][1]})", row=r, col=1)
            r += 1
        fig.update_xaxes(title_text=x_title, row=max(rows, 1), col=1)
        fig.update_layout(height=140 * max(rows, 1) + 120, showlegend=False)
    else:
        fig = go.Figure()
        n = len(series)
        right = 0.97 - AXIS_STEP * max(n - 1, 0)          # where the plot area ends; the extra axes sit to its right
        if has_ele:
            fig.add_trace(base_trace(yaxis="y"))
        layout_axes = {"xaxis": dict(domain=[0.07, right], title=x_title),
                       "yaxis": dict(title="Elevation (m)", side="left", showgrid=False)}
        for k, label in enumerate(series):
            col, unit, color = CHANNELS[label]
            name = "y2" if k == 0 else f"y{k + 2}"
            fig.add_trace(line_trace(label, yaxis=name))
            layout_axes["yaxis" + name[1:]] = dict(
                title=dict(text=f"{label} ({unit})", font=dict(color=color)), tickfont=dict(color=color),
                overlaying="y", side="right", anchor="free", position=min(1.0, right + AXIS_STEP * k),
                showgrid=(k == 0), zeroline=False)
        fig.update_layout(**layout_axes, height=460, showlegend=True,
                          legend=dict(orientation="h", y=1.08, x=0.07))
    fig.update_layout(hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
    return fig
