"""
Deeper-analytics pages for the PDF reports (matplotlib, A4, same furniture as src.viz.report_pdf).
deeper_figures(data) takes the dict from src.data.reports.deeper_data.collect_deeper_data and returns three figures:
power x heart rate, power x cadence, recovery and efficiency.
"""
import numpy as np
import pandas as pd

from src.analysis import correlations as co
from src.viz.report_pdf import (INK, LEFT, PAL, RIGHT, SERIES, SOFT, _axes, _dates, _empty, _fmt, _legend,
                                _new_page, _note, _style, _table, _tile_row)
from src.i18n import fdate, t, tr


def _label(a, b):
    return f"{fdate(a, '%d %b')} – {fdate(b, '%d %b %Y')}"


def _delta(a, b, spec='+.1f', unit=''):
    return '' if a is None or b is None else f"{a - b:{spec}}{unit} vs previous"


def _heatmap(ax, cells, y_bin, ylabel, overlay=None):
    _style(ax, None, ylabel, grid=False)
    if cells is None or cells.empty:
        return _empty(ax, t('No data in this period.'))
    pivot = cells.pivot_table(index='y', columns='x', values='seconds', aggfunc='sum', fill_value=0)
    xs = pivot.columns.to_numpy(dtype=float)
    ys = pivot.index.to_numpy(dtype=float)
    ax.imshow(pivot.to_numpy() / 60, origin='lower', aspect='auto', cmap='Blues',
              extent=[xs.min(), xs.max() + 10, ys.min(), ys.max() + y_bin])
    if overlay is not None and not overlay.empty:
        ax.plot(overlay['power'], overlay['mean'], color=SERIES[1], lw=1.4)
    ax.set_xlabel(t('Power (W)'), fontsize=7.5, color=SOFT)


def _curves(ax, curves, ylabel):
    _style(ax, None, ylabel)
    drawn = False
    for i, (label, c) in enumerate(curves.items()):
        if c is not None and not c.empty:
            ax.plot(c['power'], c['mean'], color=[SERIES[0], PAL['muted']][i % 2], lw=1.5, label=label)
            drawn = True
    if drawn:
        ax.set_xlabel(t('Power (W)'), fontsize=7.5, color=SOFT)
        _legend(ax, ncol=2)
    else:
        _empty(ax, t('Not enough data.'))


def _trend(ax, series_by_label, ylabel):
    _style(ax, None, ylabel)
    drawn = False
    for i, (label, s) in enumerate(series_by_label.items()):
        if s is not None and len(s):
            ax.plot(s.index, s.values, color=SERIES[i % len(SERIES)], lw=1.3, marker='o', ms=3, label=label)
            drawn = True
    if drawn:
        _dates(ax)
        if len(series_by_label) > 1:
            _legend(ax, ncol=2)
    else:
        _empty(ax, t('Not enough data.'))


def _page_hr(d):
    fig = _new_page(t('Deeper analytics: power and heart rate'), _label(d['start'], d['end']))
    h = d['hr']
    if not h:
        _empty(_axes(fig, 0.8, 0.2), t('No ride in this period has both power and heart rate while pedalling.'))
        return fig
    fit, prev = h['fit'], h['fit_prev']
    _tile_row(fig, 0.895, [
        (t('Correlation (r)'), _fmt(fit and fit['r'], '.2f'), _delta(fit and fit['r'], prev and prev['r'], '+.2f')),
        (t('HR rise per +10 W'), _fmt(fit and fit['slope'] * 10, '.1f', ' bpm'),
         _delta(fit and fit['slope'] * 10, prev and prev['slope'] * 10)),
        (f"HR at {h['ref_power']} W", _fmt(h['at_ref'], '.0f', ' bpm'), _delta(h['at_ref'], h['at_ref_prev'], '+.0f')),
    ], cols=3)
    fig.text(LEFT, 0.80, t('Time at power × heart rate (minutes)'), fontsize=9.5, fontweight='bold', color=INK, va='top')
    _heatmap(_axes(fig, 0.775, 0.22), h['cells'], 5, t('Heart rate (bpm)'), h['curve'])
    fig.text(LEFT, 0.505, t('Heart rate at each power'), fontsize=9.5, fontweight='bold', color=INK, va='top')
    _curves(_axes(fig, 0.48, 0.17), {_label(d['start'], d['end']): h['curve'],
                                     _label(d['prev_start'], d['prev_end']): h['curve_prev']}, t('Average HR (bpm)'))
    fig.text(LEFT, 0.275, t('Weekly heart rate at {ref_power} W', ref_power=h['ref_power']), fontsize=9.5, fontweight='bold', color=INK, va='top')
    _trend(_axes(fig, 0.25, 0.14), {t("HR at {ref_p} W", ref_p=h['ref_power']): h['weekly']}, t('bpm'))
    _note(fig, 0.075, t("A curve that moves down between periods means the same power at a lower heart rate: the "
                      "clearest sign of improving aerobic fitness (or, if it rises, fatigue, heat or dehydration)."))
    return fig


def _page_cadence(d):
    fig = _new_page(t('Deeper analytics: power and cadence'), _label(d['start'], d['end']))
    c = d['cadence']
    if not c:
        _empty(_axes(fig, 0.8, 0.2), t('No ride in this period has both power and cadence while pedalling.'))
        return fig
    q, qp = c['quadrants'], c['quadrants_prev']
    hf = q and q['QI'] + q['QII']
    hfp = qp and qp['QI'] + qp['QII']
    fit = c['fit']
    _tile_row(fig, 0.895, [
        (t('Median cadence'), _fmt(c['median'], '.0f', ' rpm'), _delta(c['median'], c['median_prev'], '+.0f', ' rpm')),
        (t('High-force time (QI+QII)'), _fmt(hf, '.0%'), _delta(hf and hf * 100, hfp and hfp * 100, '+.0f', ' pp')),
        (t('Cadence rise per +50 W'), _fmt(fit and fit['slope'] * 50, '.1f', ' rpm'), ''),
    ], cols=3)
    fig.text(LEFT, 0.80, t('Time at power × cadence (minutes)'), fontsize=9.5, fontweight='bold', color=INK, va='top')
    _heatmap(_axes(fig, 0.775, 0.2, LEFT, 0.55), c['cells'], 5, t('Cadence (rpm)'), c['curve'])
    fig.text(0.63, 0.80, t('Cadence by power zone'), fontsize=9.5, fontweight='bold', color=INK, va='top')
    ax = _axes(fig, 0.775, 0.2, 0.63, RIGHT)
    z = c['by_zone']
    _style(ax, None, t('rpm'))
    if z is not None and not z.empty:
        ax.bar(z['zone'], z['cadence'], color=SERIES[0], width=0.6)
        for x, v, s in zip(z['zone'], z['cadence'], z['share']):
            ax.text(x, v, f"{v:.0f}\n{s:.0%}", ha='center', va='bottom', fontsize=6.5, color=INK)
        ax.set_ylim(0, float(z['cadence'].max()) * 1.25)
    else:
        _empty(ax, t('No data.'))
    fig.text(LEFT, 0.535, t('Quadrant analysis'), fontsize=9.5, fontweight='bold', color=INK, va='top')
    ax = _axes(fig, 0.51, 0.2, LEFT, 0.55)
    _style(ax, None, t('Pedal force (N)'))
    cells, ftp = c['cells'], d['ftp']
    if q and ftp and c['median']:
        aepf, cpv = co.pedal_force_velocity(cells['x'] + 5, cells['y'] + 2.5, c['crank_mm'] / 1000)
        minutes = (cells['seconds'] / 60).to_numpy()
        ax.scatter(cpv, aepf, s=np.clip(np.sqrt(minutes) * 6, 2, 40), c=minutes, cmap='Blues', alpha=0.7, lw=0)
        cpv_ref = c['median'] * c['crank_mm'] / 1000 * 2 * np.pi / 60
        ax.axvline(cpv_ref, color=PAL['muted'], lw=0.7, ls='--')
        ax.axhline(ftp / cpv_ref, color=PAL['muted'], lw=0.7, ls='--')
        xs = np.linspace(max(float(np.nanmin(cpv)), 0.3), float(np.nanmax(cpv)), 50)
        ax.plot(xs, ftp / xs, color=SERIES[1], lw=0.9)
        ax.set_ylim(0, float(np.nanpercentile(aepf, 99.5)) * 1.1)
        ax.set_xlabel(t('Circumferential pedal velocity (m/s)'), fontsize=7.5, color=SOFT)
    else:
        _empty(ax, t('No data.'))
    ax = _axes(fig, 0.51, 0.2, 0.63, RIGHT)
    if q:
        rows = [[k, tr(co.QUADRANT_NAMES[k]).replace(' · ', ', '), f"{v:.0%}",
                 f"{qp[k]:.0%}" if qp else ''] for k, v in q.items()]
        _table(ax, ['', t('Meaning'), t('Now'), t('Prev.')], rows, [0.14, 0.52, 0.17, 0.17], aligns=['left', 'left', 'right', 'right'],
               fontsize=6.5, row_h_in=0.3)
    else:
        ax.set_axis_off()
    fig.text(LEFT, 0.275, t('Weekly median cadence'), fontsize=9.5, fontweight='bold', color=INK, va='top')
    _trend(_axes(fig, 0.25, 0.14), {t('Median cadence'): c['weekly']}, t('rpm'))
    _note(fig, 0.075, t("Lines at your median cadence and at the pedal force that gives FTP at that cadence; the red "
                      "curve is FTP. QII-heavy riding builds muscular force (climbing, low-cadence work); QIV is "
                      "fast, light spinning."))
    return fig


def _page_recovery(d):
    fig = _new_page(t('Deeper analytics: recovery and efficiency'), _label(d['start'], d['end']))
    r = d['recovery']
    _tile_row(fig, 0.895, [
        (t('HR recovery (60 s)'), _fmt(r['hrr'], '.0f', ' bpm'), _delta(r['hrr'], r['hrr_prev'], '+.0f')),
        (t('EF, endurance rides'), _fmt(r['ef'], '.2f'), _delta(r['ef'], r['ef_prev'], '+.2f')),
        (t('Decoupling, endurance rides'), _fmt(r['decoupling'], '.1f', ' %'), _delta(r['decoupling'], r['decoupling_prev'], '+.1f', ' pp')),
    ], cols=3)
    fig.text(LEFT, 0.80, t('Heart-rate recovery after hard efforts'), fontsize=9.5, fontweight='bold', color=INK, va='top')
    _trend(_axes(fig, 0.775, 0.2), {t('HRR60 (bpm drop in 60 s)'): r['hrr_series']}, t('bpm'))
    fig.text(LEFT, 0.51, t('Efficiency factor on endurance rides'), fontsize=9.5, fontweight='bold', color=INK, va='top')
    _trend(_axes(fig, 0.485, 0.2), {'EF (NP / HR)': r['ef_series']}, 'NP / HR')
    _note(fig, 0.20, t("HR recovery: median drop in the minute after hard efforts (1 min or more above 105 % FTP followed "
                     "by easy riding). EF and decoupling: rides of 60+ min with VI 1.10 or less. Decoupling below about "
                     "5 % suggests solid aerobic endurance. The trends show the previous period too."))
    return fig


def deeper_figures(data):
    return [_page_hr(data), _page_cadence(data), _page_recovery(data)]
