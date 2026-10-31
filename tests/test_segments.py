"""Segment analysis: numbers for any stretch of a ride and for laps."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import segments as sg


def ride(n=600, **cols):
    t = np.arange(n)
    df = pd.DataFrame({
        'time': pd.date_range('2026-10-01 08:00:00', periods=n, freq='1s'),
        'dist_m': t * 8.0, 'speed': 8.0, 'ele': 100.0 + t * 0.4,            # 5 % climb
        'power': np.where(t < 300, 200.0, 300.0), 'hr': 140.0 + t * 0.05,
        'cadence': np.where(t % 10 < 8, 90.0, 0.0)})
    for k, v in cols.items():
        df[k] = v
    return df


def test_span_by_distance_selects_the_samples_between_two_distances():
    df = ride()
    i0, i1 = sg.span_by_distance(df, 800, 1600)
    assert (df['dist_m'].iloc[i0], df['dist_m'].iloc[i1]) == (800.0, 1600.0)


def test_span_by_time_and_clipping():
    df = ride()
    i0, i1 = sg.span_by_time(df, '2026-10-01 08:01:00', '2026-10-01 08:02:00')
    assert (i0, i1) == (60, 120)
    assert sg.span_by_distance(df, -50, 10 ** 7) == (0, len(df) - 1)


def test_stats_of_a_steady_climb():
    df = ride()
    s = sg.segment_stats(df, 0, 299, ftp=250, weight=70)
    assert s['elapsed_s'] == 300 and s['dist_km'] == pytest.approx(2.392, abs=1e-3)
    assert s['avg_power'] == pytest.approx(200) and s['norm_power'] == pytest.approx(200, abs=0.5)
    assert s['w_per_kg'] == pytest.approx(200 / 70) and s['vi'] == pytest.approx(1.0, abs=0.01)
    assert s['if'] == pytest.approx(0.8, abs=0.01) and s['tss'] == pytest.approx(300 / 3600 * 0.64 * 100, rel=0.02)
    assert s['kj'] == pytest.approx(60.0)
    assert s['ascent_m'] == pytest.approx(0.4 * 299, rel=0.05) and s['avg_grade_pct'] == pytest.approx(5.0, abs=0.3)
    assert s['avg_speed_kph'] == pytest.approx(28.8) and s['avg_hr'] == pytest.approx(140 + 0.05 * 149.5, abs=0.2)
    assert s['vam_m_h'] == pytest.approx(s['ascent_m'] / 300 * 3600)


def test_segments_differ_from_each_other():
    df = ride()
    a, b = sg.segment_stats(df, 0, 299), sg.segment_stats(df, 300, 599)
    assert a['avg_power'] == pytest.approx(200) and b['avg_power'] == pytest.approx(300)
    assert b['avg_hr'] > a['avg_hr'] and b['ef'] == pytest.approx(b['norm_power'] / b['avg_hr'])


def test_cadence_averages_pedalling_samples_only():
    s = sg.segment_stats(ride(), 0, 299)
    assert s['avg_cadence'] == pytest.approx(90.0) and s['pedalling_share'] == pytest.approx(0.8)


def test_short_segment_has_no_normalized_power():
    s = sg.segment_stats(ride(), 100, 115)
    assert 'norm_power' not in s and s['avg_power'] == pytest.approx(200)


def test_missing_channels_leave_those_numbers_out():
    df = ride().drop(columns=['power', 'hr', 'cadence'])
    s = sg.segment_stats(df, 0, 299)
    assert s['dist_km'] > 0 and 'avg_power' not in s and 'avg_hr' not in s and 'avg_cadence' not in s


def test_stationary_segment_does_not_divide_by_zero():
    df = ride(speed=0.0, dist_m=1000.0, power=0.0, cadence=0.0)
    s = sg.segment_stats(df, 0, 100)
    assert s['moving_s'] == 0 and s.get('avg_speed_kph') is None


def test_noise_in_elevation_is_not_counted_as_climbing():
    rng = np.random.default_rng(0)
    df = ride(ele=100 + rng.normal(0, 0.5, 600))
    assert sg.segment_stats(df, 0, 599)['ascent_m'] < 40          # raw diffs would give hundreds of metres


def test_laps_are_mapped_to_rows_by_start_time():
    df = ride()
    spans = pd.DataFrame({'start': pd.to_datetime(['2026-10-01 08:00:00', '2026-10-01 08:05:00']),
                          'elapsed_s': [300.0, 300.0]})
    laps = sg.lap_spans(df, spans)
    assert laps == [("Lap 1", 0, 299), ("Lap 2", 300, 599)]
    t = sg.stats_table(df, laps, ftp=250, weight=70)
    assert list(t['Segment']) == ["Lap 1", "Lap 2"] and list(t['Avg power (W)'].round()) == [200, 300]


def test_laps_outside_the_ride_are_ignored_and_no_laps_is_fine():
    df = ride()
    spans = pd.DataFrame({'start': pd.to_datetime(['2025-01-01 08:00:00']), 'elapsed_s': [300.0]})
    assert sg.lap_spans(df, spans) == [] and sg.lap_spans(df, None) == []


def test_selection_behaves_like_a_detected_climb():
    df = ride()
    c = sg.as_climb(df, 100, 400)
    assert set(c) >= {'start_idx', 'end_idx', 'length_m', 'elev_gain_m', 'avg_gradient_pct', 'vam_mph', 'type'}
    assert c['length_m'] == pytest.approx(2400, abs=10) and c['avg_gradient_pct'] == pytest.approx(5.0, abs=0.3)
    assert len(df.loc[c['start_idx']:c['end_idx']]) == 301


def test_ride_page_wiring_imports():
    pytest.importorskip("streamlit_folium")
    import src.viz.segment_analysis as ui           # the UI module and its helpers load
    assert ui._fmt(None, ".1f") == "n/a" and ui._fmt(3.14159, ".1f", " m") == "3.1 m"


def test_detected_climbs_become_selectable_spans():
    df = ride()
    climbs = [{'start_idx': 100, 'end_idx': 250, 'length_m': 1200, 'type': 'Cat 4', 'avg_gradient_pct': 5.0}]
    spans = sg.climb_spans(df, climbs)
    assert spans[0][:3] == ("Climb 1", 100, 250) and spans[0][3] is climbs[0]
    assert sg.climb_spans(df, [{'start_idx': 9999, 'end_idx': 99999}]) == [] and sg.climb_spans(df, None) == []


def test_lap_times_survive_missing_start_times():
    pytest.importorskip("fitparse")
    from src.data.fit import lap_spans_from_messages
    ts = pd.Timestamp("2026-10-01 08:10:00")
    spans = lap_spans_from_messages([
        {'start_time': pd.Timestamp("2026-10-01 08:00:00"), 'total_elapsed_time': 300.0},
        {'timestamp': ts, 'total_timer_time': 300.0},                    # only its end time
        {'total_elapsed_time': 120.0},                                   # nothing: follows the previous lap
        {'total_elapsed_time': 0}])                                      # empty lap: skipped
    assert list(spans['start']) == [pd.Timestamp("2026-10-01 08:00:00"), pd.Timestamp("2026-10-01 08:05:00"),
                                    ts]
    assert list(spans['elapsed_s']) == [300.0, 300.0, 120.0]
    laps = sg.lap_spans(ride(n=800), spans)
    assert [l[0] for l in laps] == ["Lap 1", "Lap 2", "Lap 3"] and laps[0][1:] == (0, 299)


def test_segment_map_highlights_the_selection():
    pytest.importorskip("folium")
    from src.viz.maps import create_segment_map
    df = ride(lat=41 + np.arange(600) * 1e-5, lon=2.0 + np.arange(600) * 1e-5)
    m = create_segment_map(df, 100, 300)
    html = m.get_root().render()
    assert "#d62728" in html and "Start" in html and "End" in html
    assert create_segment_map(ride(), 0, 10) is None                    # no GPS


# --- range drawn on the elevation profile ------------------------------------------------------------------------

def test_selection_range_uses_the_box_rounded_to_10_m():
    from src.analysis.segments import selection_range
    assert selection_range({'box': [{'x': [12.3456, 14.2], 'y': [0, 9]}]}, 0.0, 40.0) == (12.35, 14.2)
    assert selection_range({'box': [{'x': [14.2, 12.3456]}]}, 0.0, 40.0) == (12.35, 14.2)     # drawn right to left


def test_selection_range_is_clipped_to_the_ride():
    from src.analysis.segments import selection_range
    assert selection_range({'box': [{'x': [-3.0, 55.0]}]}, 0.0, 40.0) == (0.0, 40.0)


def test_selection_range_falls_back_to_the_selected_points():
    from src.analysis.segments import selection_range
    pts = {'points': [{'x': 5.0}, {'x': 7.25}, {'x': 6.1}]}
    assert selection_range(pts, 0.0, 40.0) == (5.0, 7.25)


def test_selection_range_ignores_nothing_and_slivers():
    from src.analysis.segments import selection_range
    assert selection_range(None, 0.0, 40.0) is None and selection_range({}, 0.0, 40.0) is None
    assert selection_range({'box': [], 'points': []}, 0.0, 40.0) is None
    assert selection_range({'box': [{'x': [3.0, 3.01]}]}, 0.0, 40.0) is None                 # narrower than two steps


def test_lap_table_still_renders_without_a_recent_jinja2(monkeypatch):
    """pandas' Styler raises ImportError with an old jinja2; the comparison tables must not crash the page."""
    pytest.importorskip("streamlit")
    pytest.importorskip("streamlit_folium")
    pytest.importorskip("plotly")
    from src.viz import segment_analysis as sa
    t = pd.DataFrame({'Start (km)': [1.234], 'Distance (km)': [5.678], 'Time': ['00:10:00'], 'Avg HR': [float('nan')]})
    monkeypatch.setattr(sa.sg, 'stats_table', lambda *a, **k: t.copy())

    class Boom:
        def __get__(self, obj, cls):
            raise ImportError("Pandas requires version '3.1.2' or newer of 'jinja2'")
    monkeypatch.setattr(pd.DataFrame, 'style', Boom(), raising=False)
    out = sa._lap_table(None, [], None, None)
    assert out['Distance (km)'].iloc[0] == "5.68" and out['Avg HR'].iloc[0] == "–"
