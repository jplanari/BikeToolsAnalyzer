"""Segment analysis section of the ride page: pick any stretch of the ride, or a lap, and get the numbers of a climb."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium
from plotly.subplots import make_subplots

from src.analysis import segments as sg
from src.viz.climb_estimator import render_climb_estimator
from src.viz.figstyle import elevation_range
from src.viz.formatting import format_duration
from src.viz.maps import create_segment_map
from src.viz.plots import plot_detailed_climb
from src.viz.sections import remembered

DISTANCE_STEP_KM = 0.01            # 10 m
TIME_STEP_MIN = 0.1
MIN_ESTIMATOR_LENGTH_M = 300
MIN_ESTIMATOR_GRADE_PCT = 1.0


def _fmt(v, spec, unit=""):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:{spec}}{unit}"


def _time(seconds):
    return "n/a" if seconds is None else format_duration(int(round(seconds)))


MAX_SELECT_POINTS = 1500            # invisible markers that carry the drag-selection


def _overview(df, i0, i1):
    """Whole-ride elevation with the selection shaded. Drag across it to choose a new range."""
    x = df['dist_m'] / 1000.0 if 'dist_m' in df.columns else pd.Series(np.arange(len(df)) / 60.0, index=df.index)
    fig = go.Figure(go.Scatter(x=x, y=df['ele'], mode='lines', line=dict(color='lightgray', width=2),
                               fill='tozeroy', fillcolor='rgba(200,200,200,0.2)', hoverinfo='skip'))
    sel = df.iloc[i0:i1 + 1]
    fig.add_trace(go.Scatter(x=x.iloc[i0:i1 + 1], y=sel['ele'], mode='lines', line=dict(color='#d62728', width=3),
                             name='Selection', hoverinfo='skip'))
    step = max(1, len(df) // MAX_SELECT_POINTS)                  # invisible markers: Plotly reports a box selection
    fig.add_trace(go.Scatter(x=x.iloc[::step], y=df['ele'].iloc[::step], mode='markers', hoverinfo='skip',
                             marker=dict(size=8, opacity=0), showlegend=False,
                             selected=dict(marker=dict(opacity=0)), unselected=dict(marker=dict(opacity=0))))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      xaxis_title="Distance (km)" if 'dist_m' in df.columns else "Minutes", yaxis_title="m",
                      dragmode="select", selectdirection="h")     # a horizontal drag picks a stretch of the route
    rng = elevation_range(df['ele'])
    if rng:
        fig.update_yaxes(range=rng)
    return fig


def _drawn_range(df, i0, i1, ride_id, lo, hi, step):
    """
    Draws the overview and returns the (start, end) the rider dragged across it, if any. The chart is keyed on the
    current selection, so a drawn box disappears once it has been applied (the red line shows it instead).
    """
    try:
        event = st.plotly_chart(_overview(df, i0, i1), use_container_width=True, key=f"seg_over_{ride_id}_{i0}_{i1}",
                                on_select="rerun", selection_mode=("box",),
                                config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "autoScale2d"]})
    except TypeError:                                              # Streamlit without chart selections
        st.plotly_chart(_overview(df, i0, i1), use_container_width=True, key=f"seg_over_{ride_id}_{i0}_{i1}")
        return None
    sel = None
    try:
        sel = event["selection"] if "selection" in event else getattr(event, "selection", None)
    except Exception:
        pass
    return sg.selection_range(sel, lo, hi, step)


def _channels(df, i0, i1):
    """Power, heart rate, speed and cadence along the selection (the ones the file has)."""
    seg = df.iloc[i0:i1 + 1]
    x = seg['dist_m'] / 1000.0 if 'dist_m' in seg.columns else pd.Series(np.arange(len(seg)) / 60.0, index=seg.index)
    spec = [('power', 'Power (W)', '#1f77b4'), ('hr', 'HR (bpm)', '#d62728'),
            ('speed_kmh', 'Speed (km/h)', '#2ca02c'), ('cadence', 'Cadence (rpm)', '#9467bd')]
    have = [s for s in spec if s[0] in seg.columns and seg[s[0]].notna().any()]
    if not have:
        return None
    fig = make_subplots(rows=len(have), cols=1, shared_xaxes=True, vertical_spacing=0.04)
    for r, (col, title, color) in enumerate(have, 1):
        y = seg[col].rolling(5, min_periods=1, center=True).mean() if col == 'power' else seg[col]
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color=color, width=1.5), name=title), row=r, col=1)
        fig.update_yaxes(title_text=title, row=r, col=1)
    fig.update_layout(height=170 * len(have) + 40, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    fig.update_xaxes(title_text="Distance (km)" if 'dist_m' in seg.columns else "Minutes", row=len(have), col=1)
    return fig


def _metrics(s):
    cells = [
        ("Time", _time(s.get('elapsed_s')), "Elapsed; moving " + _time(s.get('moving_s'))),
        ("Distance", _fmt(s.get('dist_km'), ".2f", " km"), None),
        ("Ascent", _fmt(s.get('ascent_m'), ".0f", " m"), f"Descent {_fmt(s.get('descent_m'), '.0f', ' m')}"),
        ("Avg gradient", _fmt(s.get('avg_grade_pct'), ".1f", " %"), None),
        ("VAM", _fmt(s.get('vam_m_h'), ".0f", " m/h"), None),
        ("Avg speed", _fmt(s.get('avg_speed_kph'), ".1f", " km/h"), f"max {_fmt(s.get('max_speed_kph'), '.1f', ' km/h')}"),
        ("Avg power", _fmt(s.get('avg_power'), ".0f", " W"),
         f"{_fmt(s.get('w_per_kg'), '.2f', ' W/kg')} · max {_fmt(s.get('max_power'), '.0f', ' W')}"),
        ("Normalized power", _fmt(s.get('norm_power'), ".0f", " W"), "Needs at least 30 s of power data."),
        ("Avg HR", _fmt(s.get('avg_hr'), ".0f", " bpm"), f"max {_fmt(s.get('max_hr'), '.0f', ' bpm')}"),
        ("Avg cadence", _fmt(s.get('avg_cadence'), ".0f", " rpm"), "While pedalling."),
        ("Intensity factor", _fmt(s.get('if'), ".2f"), None),
        ("TSS", _fmt(s.get('tss'), ".0f"), None),
        ("Work", _fmt(s.get('kj'), ".0f", " kJ"), None),
        ("Variability index", _fmt(s.get('vi'), ".2f"), None),
        ("Efficiency factor", _fmt(s.get('ef'), ".2f"), "Normalized power / average HR."),
    ]
    for r in range(0, len(cells), 3):
        cols = st.columns(3)
        for col, (label, value, help_) in zip(cols, cells[r:r + 3]):
            col.metric(label, value, help=help_)


def _lap_table(df, spans, ftp, weight):
    t = sg.stats_table(df, spans, ftp, weight)
    t['Time'] = t['Time'].map(_time)
    return t.style.format({'Start (km)': "{:.1f}", 'Distance (km)': "{:.2f}", 'Ascent (m)': "{:.0f}", 'Grade (%)': "{:.1f}",
                           'Speed (km/h)': "{:.1f}", 'Avg power (W)': "{:.0f}", 'NP (W)': "{:.0f}", 'Avg HR': "{:.0f}",
                           'Cadence': "{:.0f}"}, na_rep="–")


def render_segment_analysis(df, settings, lap_time_spans=None, climbs=None, ride_id="0"):
    """
    lap_time_spans: DataFrame(start, elapsed_s) from the FIT file, or None (not a FIT file).
    climbs: detect_climbs() output. Both appear as ready-made selections next to the free range.
    """
    if len(df) < 30 or 'ele' not in df.columns:
        return
    st.subheader("📐 Segment analysis")
    ftp, weight = settings.get('ftp'), settings.get('weight')
    laps = sg.lap_spans(df, lap_time_spans)
    climb_spans = sg.climb_spans(df, climbs)
    if lap_time_spans is not None and not laps:
        n = lap_time_spans.attrs.get('lap_messages', len(lap_time_spans))
        st.caption("No laps could be matched to this ride" + (f" ({n} lap record(s) in the file)." if n else
                   ": the file has no laps (the whole ride is one)."))
    modes = ["Custom selection"] + (["Lap"] if laps else []) + (["Climb"] if climb_spans else [])
    mode_key, range_key, pending_key = f"seg_mode_{ride_id}", f"seg_range_{ride_id}", f"seg_pending_{ride_id}"
    pending = st.session_state.pop(pending_key, None)             # a range dragged on the profile on the last run
    if pending:                                                   # widget values can only be set before they are drawn
        st.session_state[mode_key] = modes[0]
        st.session_state[range_key] = pending
    mode = remembered(st.radio, mode_key, modes[0], "Analyse", modes, horizontal=True)

    chosen = None                                   # the detected climb when one is picked
    if mode == "Lap":
        label_of = {}
        for label, i0, i1 in laps:
            s = sg.segment_stats(df, i0, i1)
            label_of[label] = f"{label} · {_fmt(s.get('dist_km'), '.1f', ' km')} · {_time(s.get('elapsed_s'))}"
        pick = remembered(st.selectbox, f"seg_lap_{ride_id}", laps[0][0], "Lap", [l for l, *_ in laps],
                          format_func=label_of.get)
        _, i0, i1 = next(x for x in laps if x[0] == pick)
    elif mode == "Climb":
        label_of = {}
        for label, i0, i1, c in climb_spans:
            label_of[label] = (f"{label} · {c['type']} · {c['length_m'] / 1000:.1f} km · "
                               f"{c['avg_gradient_pct']:.1f} %")
        pick = remembered(st.selectbox, f"seg_climb_{ride_id}", climb_spans[0][0], "Climb",
                          [x[0] for x in climb_spans], format_func=label_of.get)
        _, i0, i1, chosen = next(x for x in climb_spans if x[0] == pick)
    elif 'dist_m' in df.columns and df['dist_m'].notna().any():
        total_km = float(df['dist_m'].max()) / 1000.0
        a, b = remembered(st.slider, range_key, (0.0, round(min(total_km, 5.0), 2)),
                          "Distance range (km)", 0.0, round(total_km, 2), step=DISTANCE_STEP_KM, format="%.2f")
        i0, i1 = sg.span_by_distance(df, a * 1000.0, b * 1000.0)
    else:
        minutes = len(df) / 60.0
        a, b = remembered(st.slider, range_key, (0.0, round(min(minutes, 10.0), 1)),
                          "Time range (min)", 0.0, round(minutes, 1), step=TIME_STEP_MIN)
        i0, i1 = sg._clip(df, int(a * 60), int(b * 60))

    has_dist = 'dist_m' in df.columns and df['dist_m'].notna().any()
    top = round(float(df['dist_m'].max()) / 1000.0, 2) if has_dist else round(len(df) / 60.0, 1)
    drawn = _drawn_range(df, i0, i1, ride_id, 0.0, top, DISTANCE_STEP_KM if has_dist else TIME_STEP_MIN)
    st.caption("Drag across the profile to choose a stretch of the route. In *Custom selection* the range slider "
               f"fine-tunes it in {'10 m' if has_dist else '0.1 min'} steps.")
    if drawn and mode == "Custom selection" and drawn == tuple(st.session_state.get(range_key) or ()):
        drawn = None                                              # already the current range
    if drawn:
        st.session_state[pending_key] = drawn
        st.rerun()
    if i1 - i0 < 5:
        st.info("The selection is too short to analyse. Widen the range.")
        return
    stats = sg.segment_stats(df, i0, i1, ftp, weight)

    seg_map = create_segment_map(df, i0, i1)
    if seg_map is not None:
        left, right = st.columns([2, 3])
        with left:
            try:
                st_folium(seg_map, height=520, use_container_width=True, returned_objects=[],
                          key=f"seg_map_{ride_id}")
            except TypeError:                                # older streamlit-folium without use_container_width
                st_folium(seg_map, width=480, height=520, returned_objects=[], key=f"seg_map_{ride_id}")
        with right:
            _metrics(stats)
    else:
        _metrics(stats)

    fig = _channels(df, i0, i1)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, key=f"seg_ch_{ride_id}_{i0}_{i1}")

    climb = chosen or sg.as_climb(df, i0, i1)
    if climb['length_m'] >= MIN_ESTIMATOR_LENGTH_M and 'dist_m' in df.columns:
        with st.expander("Gradient profile and performance estimator", expanded=chosen is not None):
            st.plotly_chart(plot_detailed_climb(df.iloc[i0:i1 + 1]), use_container_width=True,
                            key=f"seg_prof_{ride_id}_{i0}_{i1}")
            if climb['avg_gradient_pct'] >= MIN_ESTIMATOR_GRADE_PCT:
                render_climb_estimator(df, climb, f"seg_{i0}_{i1}", settings, ride_id)
            else:
                st.caption("The performance estimator is for climbs: this stretch averages less than "
                           f"{MIN_ESTIMATOR_GRADE_PCT:.0f} % gradient.")
    if laps:
        with st.expander("Compare all laps"):
            st.dataframe(_lap_table(df, laps, ftp, weight), use_container_width=True, hide_index=True)
    if climb_spans:
        with st.expander("Compare all climbs"):
            st.dataframe(_lap_table(df, [(l, a, b) for l, a, b, _ in climb_spans], ftp, weight),
                         use_container_width=True, hide_index=True)
