"""
PDF report of one analysed ride (A4, light theme for printing), drawn with matplotlib in the style of the
training report (src.viz.report_pdf, whose page furniture it reuses).

build_ride_report_pdf(data) takes the dict from src.data.ride_report_data.collect_ride_report and returns
the PDF as bytes. Four pages: summary and time series; power curve and zones; route and climbs;
aerodynamics, heat and laps. A section without data says so instead of leaving a hole.
"""
import io
from datetime import datetime

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from src.viz.report_pdf import (INK, LEFT, PAL, PAGE, RIGHT, SERIES, SOFT, _axes, _empty, _fmt, _footer,
                                _legend, _new_page, _note, _style, _table, _tile_row)

KEY_S = (1, 5, 15, 30, 60, 300, 600, 1200, 1800, 3600, 5400, 7200)


def _hms(seconds):
    if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)):
        return 'n/a'
    s = int(round(seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _elapsed_axis(ax, max_min, label=True):
    """x axis in elapsed time (h:mm) with a round step."""
    step = next((s for s in (5, 10, 15, 30, 60, 120, 180) if max_min / s <= 8), 240)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(step))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v // 60)}:{int(v % 60):02d}"))
    ax.set_xlim(0, max_min)
    if label:
        ax.set_xlabel('elapsed time (h:mm)', fontsize=7.5, color=SOFT)


def _line(ax, s, col, title, ylabel, color, fill=False, zero=False, label=True):
    _style(ax, title, ylabel)
    if col not in s.columns or s[col].dropna().empty:
        return _empty(ax, f'No {title.lower()} data in this ride.')
    ax.plot(s['t_min'], s[col], color=color, lw=1.2)
    if fill:
        ax.fill_between(s['t_min'], s[col], s[col].min() - 5, color=color, alpha=0.12, lw=0)
    if zero:
        ax.set_ylim(bottom=0)
    _elapsed_axis(ax, float(s['t_min'].iloc[-1]) or 1.0, label)


def _power_chart(ax, s, ftp):
    _line(ax, s, 'power', 'Power (30 s average)', 'watts', SERIES[0], zero=True, label=False)
    if ftp and 'power' in s.columns:
        ax.axhline(ftp, color=PAL['muted'], lw=0.8, ls=(0, (4, 2)))
        ax.text(ax.get_xlim()[1], ftp, f' FTP {ftp} W', fontsize=7, color=SOFT, va='bottom', ha='right')


def _page_summary(d):
    s, st = d['series'], d['summary']
    date = f"{d['start']:%a %d %b %Y, %H:%M}" if d['start'] is not None else ''
    wx = d['weather']
    wtxt = ''
    if wx:
        wtxt = f"  ·  {wx['temp_c']:.0f} °C" + (f", wind ~{wx['wind_kmh']:.0f} km/h" if wx['wind_kmh'] is not None else '')
    fig = _new_page(d['name'], f"{date}{wtxt}" + (f"  ·  {d['user']}" if d['user'] else ''))
    power = st['avg_power']
    hr = st['avg_hr']
    tss_label = 'TSS (HR)' if st['tss_source'] == 'hr' else 'TSS'
    _tile_row(fig, 0.895, [
        ('Distance', f"{d['distance_km']:.1f} km", f"{d['ascent_m']:.0f} m climbing"),
        ('Moving time', _hms(st['moving_time_s']), f"elapsed {_hms(st['duration_s'])}"),
        ('Average speed', f"{st['avg_speed_kph']:.1f} km/h", 'while moving'),
        ('Average power', _fmt(power if power else None, '.0f', ' W'),
         f"max {d['max_power']:.0f} W" if d['max_power'] else ''),
        ('Average HR', _fmt(hr if hr else None, '.0f', ' bpm'), f"max {d['max_hr']:.0f}" if d['max_hr'] else ''),
        ('Normalized power', _fmt(st['norm_power'] or None, '.0f', ' W'), ''),
        ('Intensity factor', _fmt(st['if'] if st['norm_power'] else None, '.2f'),
         f"FTP {d['ftp']} W" if d['ftp'] else ''),
        (tss_label, _fmt(st['tss'] if st['tss_source'] else None, '.0f'), 'from heart rate' if st['tss_source'] == 'hr' else ''),
        ('Work', _fmt(st['kj'] if st['norm_power'] else None, '.0f', ' kJ'), ''),
        ('Variability index', _fmt(st['vi'], '.2f'), 'NP / average power'),
        ('Efficiency factor', _fmt(st['ef'], '.2f'), 'NP / average HR'),
        ('Aerobic decoupling', _fmt(st['decoupling_pct'], '.1f', ' %'), 'power:HR drift'),
        ('HR recovery (1 min)', _fmt(st['hrr60'], '.0f', ' bpm'), ''),
    ])
    ax = _axes(fig, 0.60, 0.115)
    _line(ax, s, 'ele', 'Elevation', 'metres', PAL['muted'], fill=True, label=False)
    _power_chart(_axes(fig, 0.405, 0.115), s, d['ftp'])
    _line(_axes(fig, 0.21, 0.115), s, 'hr', 'Heart rate', 'bpm', SERIES[1])
    cleaned = st.get('cleaned') or {}
    notes = []
    if sum(cleaned.values()):
        notes.append("Impossible sensor readings were removed before scoring: " + ", ".join(
            f"{n} {name}" for name, n in cleaned.items() if n) + " samples.")
    notes.append("Normalized power, IF and TSS follow Coggan; time-series charts are binned to about 900 points. "
                 "Efficiency factor and decoupling are meaningful for steady rides of 20+ minutes.")
    _note(fig, 0.055, ' '.join(notes))
    return fig


def _chart_power_curve(ax, d):
    _style(ax, 'Power curve', 'watts')
    drawn = False
    for color, label, curve, lw in ((SERIES[0], 'this ride', d['power_curve'], 2.0),
                                    (SERIES[1], 'best, last 90 days', d['recent_curve'], 1.4),
                                    (SERIES[2], 'best, all time', d['best_curve'], 1.4)):
        pts = sorted((k, v) for k, v in (curve or {}).items() if v and k >= 1)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw, label=label)
            drawn = True
    if not drawn:
        return _empty(ax, 'No power data in this ride.')
    ax.set_xscale('log')
    ticks = [t for t in KEY_S]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}s" if t < 60 else (f"{t // 60}m" if t < 3600 else f"{t / 3600:g}h") for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(left=1)
    _legend(ax, ncol=3)


def _chart_zones(ax, title, zones, labels=None):
    _style(ax, title, 'minutes in zone')
    if not zones or not sum(zones.values()):
        return _empty(ax, 'Not available (needs FTP / LTHR and the sensor).')
    keys = list(zones)
    mins = np.array([zones[k] for k in keys]) / 60.0
    share = mins / mins.sum() * 100
    short = [k.split(' ')[0] for k in keys]
    ax.bar(range(len(keys)), mins, color=SERIES[0], width=0.62)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(short)
    for i, (m, p) in enumerate(zip(mins, share)):
        ax.text(i, m, f"{m:.0f}\n{p:.0f}%", ha='center', va='bottom', fontsize=7, color=INK)
    ax.set_ylim(0, mins.max() * 1.25)
    ax.grid(axis='x', visible=False)


def _page_effort(d):
    fig = _new_page('Power and zones', d['name'])
    _chart_power_curve(_axes(fig, 0.855, 0.24), d)
    z = d['zones']
    _chart_zones(_axes(fig, 0.53, 0.2, LEFT, 0.48), 'Time in power zones', z.get('power'))
    _chart_zones(_axes(fig, 0.53, 0.2, 0.55, RIGHT), 'Time in heart-rate zones', z.get('hr'))
    s = d['series']
    if 'w_prime_balance' in s.columns and s['w_prime_balance'].notna().any():
        ax = _axes(fig, 0.265, 0.17)
        _line(ax, s, 'w_prime_balance', "W' balance", 'joules', SERIES[1])
    else:
        fig.text(LEFT, 0.265, "W' balance", fontsize=9.5, fontweight='bold', color=INK, va='top')
        fig.text(LEFT, 0.24, "Not available for this ride (needs power and an FTP).", fontsize=8.5, color=SOFT, va='top')
    _note(fig, 0.085, "Power zones are Coggan's (percent of FTP), heart-rate zones Friel's (percent of LTHR). "
                      "The all-time and 90-day curves are the rider's saved history, when there is one.")
    return fig


def _page_route(d):
    fig = _new_page('Route and climbs', d['name'])
    r = d['route']
    ax = fig.add_axes([LEFT, 0.55, 0.38, 0.32])
    ax.set_axis_off()
    if r is not None and r['lat'].notna().any():
        lat, lon = r['lat'].to_numpy(), r['lon'].to_numpy()
        ax.plot(lon, lat, color=SERIES[0], lw=1.6)
        ax.plot(lon[0], lat[0], 'o', color=SERIES[2], ms=5)
        ax.plot(lon[-1], lat[-1], 's', color=SERIES[1], ms=5)
        ax.set_aspect(1.0 / max(np.cos(np.radians(np.nanmean(lat))), 0.2))
        ax.text(0, -0.03, 'start ●   finish ■', transform=ax.transAxes, fontsize=7, color=SOFT, va='top')
    else:
        _empty(ax, 'No GPS coordinates in this ride.')
    ax2 = _axes(fig, 0.87, 0.3, 0.55, RIGHT)
    _line(ax2, d['series'], 'ele', 'Elevation', 'metres', PAL['muted'], fill=True)
    fig.text(LEFT, 0.5, f"Detected climbs ({len(d['climbs'])})", fontsize=9.5, fontweight='bold', color=INK, va='top')
    ax3 = _axes(fig, 0.485, 0.40)
    if d['climbs']:
        rows = [[c['n'], c['type'], _fmt(c['start_km'], '.1f'), f"{c['length_km']:.2f}", f"{c['gradient']:.1f}",
                 f"{c['ascent_m']:.0f}", f"{c['vam']:.0f}", _hms(c['seconds']), _fmt(c['power'], '.0f'),
                 _fmt(c['hr'], '.0f')] for c in d['climbs']]
        _table(ax3, ['#', 'Type', 'At km', 'Length km', 'Grade %', 'Ascent m', 'VAM m/h', 'Time', 'Avg W', 'Avg HR'], rows,
               [0.05, 0.19, 0.08, 0.1, 0.09, 0.1, 0.1, 0.1, 0.09, 0.1], aligns=['left', 'left'] + ['right'] * 8)
    else:
        _empty(ax3, 'No significant climbs detected on this route.')
    _note(fig, 0.085, "VAM = vertical metres climbed per hour. Climbs are detected from the smoothed grade "
                      "(see the ride page for the rules).")
    return fig


def _chart_budget(ax, budget):
    _style(ax, 'Where the watts went (ride average)', None, grid=False)
    names = list(budget)
    vals = [budget[n] for n in names]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=SERIES[0], height=0.55)
    ax.axvline(0, color=INK, lw=0.7)
    span = max(abs(v) for v in vals) or 1.0
    for yi, v in zip(y, vals):
        ax.text(v + (span * 0.02 if v >= 0 else -span * 0.02), yi, f"{v:.0f} W", va='center', fontsize=7.5, color=INK,
                ha='left' if v >= 0 else 'right')
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.5, color=INK)
    ax.set_xlabel('watts', fontsize=7.5, color=SOFT)
    ax.set_xlim(min(0, min(vals)) - span * 0.25, max(vals) + span * 0.25)
    ax.grid(axis='x', color=PAL['grid'], lw=0.6)


def _page_aero(d):
    fig = _new_page('Aerodynamics, heat and laps', d['name'])
    a = d['aero']
    if a:
        _tile_row(fig, 0.895, [
            ('Estimated CdA', _fmt(a['cda'], '.3f', ' m²'),
             f"{a['seconds'] // 60} min used" if a['cda'] else 'not enough stable data'),
            ('Mean pedal power', _fmt(a.get('pedal_w'), '.0f', ' W'), 'whole ride'),
        ], cols=5)
    ax = _axes(fig, 0.80, 0.15, 0.30, RIGHT)
    if a and a['budget']:
        _chart_budget(ax, a['budget'])
    else:
        _empty(ax, 'Power budget needs speed and power.')
    fig.text(LEFT, 0.585, 'Core temperature (thermal model)', fontsize=9.5, fontweight='bold', color=INK, va='top')
    th = d['thermal']
    ax = _axes(fig, 0.54, 0.155)
    if th:
        s = th['series']
        _style(ax, None, '°C')
        ax.plot(s['t_min'], s['core'], color=SERIES[1], lw=1.4, label='core')
        ax.plot(s['t_min'], s['skin'], color=SERIES[0], lw=1.2, label='skin')
        ax.axhline(38.5, color=PAL['muted'], lw=0.7, ls=(0, (4, 2)))
        _elapsed_axis(ax, float(s['t_min'].iloc[-1]) or 1.0)
        _legend(ax, ncol=2)
        fig.text(LEFT, 0.322, f"Peak core temperature {th['max_core']:.1f} °C, {th['end_core']:.1f} °C at the end; "
                              f"{th['minutes_above_385']:.0f} min above 38.5 °C and {th['minutes_above_39']:.0f} min above 39 °C.",
                 fontsize=8.5, color=INK, va='top')
    else:
        _empty(ax, "Not simulated for this ride: run the thermal simulation on the ride page before creating the report.")
    fig.text(LEFT, 0.29, 'Laps', fontsize=9.5, fontweight='bold', color=INK, va='top')
    ax = _axes(fig, 0.275, 0.19)
    laps = d['laps']
    if laps is not None:
        cols = list(laps.columns)[:8]
        rows = [[(f"{v:.1f}" if isinstance(v, float) else v) for v in r] for r in laps[cols].head(12).itertuples(index=False)]
        w = [0.12] + [0.88 / (len(cols) - 1)] * (len(cols) - 1) if len(cols) > 1 else [1.0]
        _table(ax, cols, rows, w, aligns=['left'] + ['right'] * (len(cols) - 1))
    else:
        _empty(ax, 'No laps recorded in this file.')
    _note(fig, 0.085, "CdA is a least-squares fit over the fast, non-braking part of the ride and absorbs errors in weight, "
                      "rolling resistance, wind and gradient: read it as about ±0.03 m². The thermal model is "
                      "uncalibrated: use it to compare rides, not as a core-temperature measurement.")
    return fig


def build_ride_report_pdf(data):
    """The report as PDF bytes."""
    pages = [_page_summary(data), _page_effort(data), _page_route(data), _page_aero(data)]
    left = f"BikeTools Analyzer  ·  {data['name']}  ·  generated {datetime.now():%d %b %Y %H:%M}"
    buf = io.BytesIO()
    with PdfPages(buf, metadata={'Title': f"Ride report - {data['name']}", 'Author': 'BikeTools Analyzer',
                                 'CreationDate': datetime.now()}) as pdf:
        for n, fig in enumerate(pages, 1):
            _footer(fig, left, n, len(pages))
            pdf.savefig(fig)
    return buf.getvalue()
