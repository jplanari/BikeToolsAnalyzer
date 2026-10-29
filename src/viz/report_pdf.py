"""
PDF training report (A4, light theme for printing), drawn with matplotlib.

build_report_pdf(data) takes the dict from src.data.report_data.collect_report_data and returns
the PDF as bytes. Colours come from the Training page palette, one y-axis per chart, a legend
whenever there is more than one series, and text in ink colours (never the series colour).
"""
import io
import json
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from src.analysis import workout as wo
from src.analysis.labels import (acwr_status, dur_label, form_status, monotony_status, ramp_status,
                                 tid_label)
from src.viz.training_plots import palette

PAGE = (8.27, 11.69)                      # A4 portrait, inches
INK, SOFT = '#2b2a27', '#6b6a66'
LEFT, RIGHT = 0.08, 0.92                  # horizontal margins (figure fraction)
PAL = dict(palette('light'))
PAL['grid'] = (0, 0, 0, 0.1)              # the page palette uses CSS rgba(), matplotlib wants a tuple
SERIES = PAL['series']
KEY_DURATIONS = (1, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600, 5400, 7200, 10800, 18000)
_DROP = str.maketrans('', '', '⚠✅⛔️️')


def _plain(text):
    """Status labels carry emoji for the screen; the PDF font has none of them."""
    return text.translate(_DROP).strip() if text else ''


# --- Page furniture -----------------------------------------------------------------------

def _new_page(title, subtitle=None):
    fig = Figure(figsize=PAGE)
    fig.text(LEFT, 0.955, title, fontsize=16, fontweight='bold', color=INK, va='top')
    if subtitle:
        fig.text(LEFT, 0.928, subtitle, fontsize=8.5, color=SOFT, va='top')
    fig.add_artist(mlines.Line2D([LEFT, RIGHT], [0.915, 0.915], color=PAL['grid'],
                                                         lw=0.8, transform=fig.transFigure))
    return fig


def _footer(fig, left, n, total):
    fig.text(LEFT, 0.022, left, fontsize=7, color=SOFT)
    fig.text(RIGHT, 0.022, f"Page {n} of {total}", fontsize=7, color=SOFT, ha='right')


def _axes(fig, top, height, left=LEFT, right=RIGHT):
    """Axes placed by its top edge (figure fraction) so pages read top-down."""
    return fig.add_axes([left, top - height, right - left, height])


def _style(ax, title=None, ylabel=None, grid=True):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(PAL['muted'])
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(labelsize=7.5, colors=SOFT, length=2.5, width=0.6)
    if grid:
        ax.grid(axis='y', color=PAL['grid'], lw=0.6)
        ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc='left', fontsize=9.5, fontweight='bold', color=INK, pad=24)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7.5, color=SOFT)
    return ax


def _legend(ax, handles=None, ncol=4):
    leg = ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=ncol, frameon=False,
                    fontsize=7.5, borderaxespad=0.2, handlelength=1.6)
    for t in leg.get_texts():
        t.set_color(INK)


def _empty(ax, message):
    ax.set_axis_off()
    ax.text(0.0, 0.5, message, fontsize=8.5, color=SOFT, va='center')


def _dates(ax):
    loc = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _tile_row(fig, top, items, cols=5):
    """Big-number tiles: label, value, small status line."""
    width = (RIGHT - LEFT) / cols
    for i, (label, value, note) in enumerate(items):
        x = LEFT + (i % cols) * width
        y = top - (i // cols) * 0.075
        fig.text(x, y, label, fontsize=7.5, color=SOFT, va='top')
        fig.text(x, y - 0.014, value, fontsize=17, fontweight='bold', color=INK, va='top')
        if note:
            fig.text(x, y - 0.044, note, fontsize=7.5, color=SOFT, va='top')


def _table(ax, header, rows, widths, aligns=None, fontsize=7.5, row_h_in=0.2):
    """Light table: bold header, hairlines, no fills. widths are fractions of the axes width."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    h_in = ax.get_position().height * PAGE[1]
    rh = row_h_in / h_in
    aligns = aligns or ['left'] + ['right'] * (len(header) - 1)
    xs = np.concatenate([[0], np.cumsum(widths)])[:-1]
    for j, text in enumerate(header):
        x = xs[j] if aligns[j] == 'left' else xs[j] + widths[j] - 0.01
        ax.text(x, 1 - rh * 0.6, text, fontsize=fontsize, fontweight='bold', color=INK,
                ha=aligns[j], va='center')
    ax.plot([0, 1], [1 - rh, 1 - rh], color=PAL['muted'], lw=0.6)
    shown = 0
    for i, row in enumerate(rows):
        y = 1 - rh * (i + 1.6)
        if y < 0:
            break
        for j, text in enumerate(row):
            x = xs[j] if aligns[j] == 'left' else xs[j] + widths[j] - 0.01
            ax.text(x, y, str(text), fontsize=fontsize, color=INK, ha=aligns[j], va='center')
        ax.plot([0, 1], [y - rh / 2, y - rh / 2], color=PAL['grid'], lw=0.5)
        shown += 1
    ax.set_ylim(0, 1)
    return shown


def _note(fig, y, text, size=8, width=104, color=SOFT):
    """Wrapped paragraph; returns the y below it."""
    import textwrap
    lines = textwrap.wrap(text, width)
    for k, line in enumerate(lines):
        fig.text(LEFT, y - k * 0.0135, line, fontsize=size, color=color, va='top')
    return y - len(lines) * 0.0135 - 0.008


def _fmt(v, spec, unit=''):
    return 'n/a' if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:{spec}}{unit}"


# --- Charts ------------------------------------------------------------------------------------

def chart_load(ax, pmc, today):
    _style(ax, 'Fitness, fatigue and form')
    if pmc is None or pmc.empty:
        return _empty(ax, 'No rides with a training load in this period.')
    actual, plan = pmc[~pmc['planned']], pmc[pmc['planned']]
    for part, alpha in ((actual, 0.6), (plan, 0.3)):
        if part.empty:
            continue
        colors = np.where(part['TSB'] >= 0, PAL['pos'], PAL['neg'])
        ax.bar(part.index, part['TSB'], width=0.85, color=colors, alpha=alpha, linewidth=0)
    for col, color in (('CTL', SERIES[0]), ('ATL', SERIES[1])):
        ax.plot(actual.index, actual[col], color=color, lw=1.6)
        if not plan.empty:
            joined = pd.concat([actual.tail(1), plan])
            ax.plot(joined.index, joined[col], color=color, lw=1.6, ls=(0, (4, 2)))
    ax.axvline(today, color=PAL['muted'], lw=0.8, ls=':')
    ax.text(today, 1.0, ' today', transform=ax.get_xaxis_transform(), fontsize=7, color=SOFT, va='top')
    ax.axhline(0, color=PAL['muted'], lw=0.5)
    _dates(ax)
    _legend(ax, [plt_line(SERIES[0], 'Fitness (CTL)'), plt_line(SERIES[1], 'Fatigue (ATL)'),
                 Patch(color=PAL['pos'], alpha=0.6, label='Form (TSB) +'),
                 Patch(color=PAL['neg'], alpha=0.6, label='Form (TSB) −')]
            + ([plt_line(PAL['muted'], 'planned (dashed)', dashed=True)] if not plan.empty else []), ncol=5)


def plt_line(color, label, dashed=False):
    return mlines.Line2D([0], [0], color=color, lw=1.6, ls=(0, (4, 2)) if dashed else '-', label=label)


def chart_volume(ax, vol, freq):
    _style(ax, f"Riding time per {'week' if freq == 'W' else 'month'}", 'hours')
    if vol is None or vol.empty:
        return _empty(ax, 'No rides in this period.')
    width = 5.5 if freq == 'W' else 24
    ax.bar(vol.index + pd.Timedelta(days=width / 2 * 0.2), vol['hours'], width=width, color=SERIES[0],
           edgecolor='white', linewidth=0.8)
    if len(vol) <= 14:
        for x, h in zip(vol.index, vol['hours']):
            if h > 0:
                ax.text(x + pd.Timedelta(days=width / 2 * 1.2), h, f"{h:.1f}", ha='center', va='bottom',
                        fontsize=7, color=INK)
    _dates(ax)


def chart_ramp(ax, ramp):
    _style(ax, 'Ramp rate (fitness change per week)', 'CTL points / week')
    weekly = ramp.resample('W-SUN').last().dropna() if ramp is not None and len(ramp) else pd.Series(dtype=float)
    if weekly.empty:
        return _empty(ax, 'Needs at least a week of history.')
    ax.bar(weekly.index, weekly.values, width=5.5, color=SERIES[0], edgecolor='white', linewidth=0.8)
    ax.axhline(8, color=PAL['muted'], lw=0.8, ls=(0, (4, 2)))
    ax.text(weekly.index[0], 8, ' +8: aggressive', fontsize=7, color=SOFT, va='bottom')
    ax.axhline(0, color=PAL['muted'], lw=0.5)
    _dates(ax)


def chart_acwr(ax, ratio):
    _style(ax, 'Acute : chronic load (ATL / CTL)', 'ratio')
    r = ratio.dropna() if ratio is not None else pd.Series(dtype=float)
    if r.empty:
        return _empty(ax, 'Needs 4 weeks of history.')
    ax.axhspan(0.8, 1.3, color=PAL['pos'], alpha=0.12, lw=0)
    ax.text(r.index[0], 1.3, ' 0.8–1.3', fontsize=7, color=SOFT, va='bottom')
    ax.plot(r.index, r.values, color=SERIES[0], lw=1.6)
    _dates(ax)


def chart_power_windows(ax, curves, weight):
    _style(ax, 'Best power by period', 'watts')
    drawn = []
    for i, (label, c) in enumerate(curves.items()):
        if c is None or c.empty:
            continue
        c = c.sort_values('duration')
        ax.plot(c['duration'], c['watts'], color=SERIES[i % 4], lw=2.0 if i == 0 else 1.5, label=label)
        drawn.append(label)
    if not drawn:
        return _empty(ax, 'No power data yet.')
    ax.set_xscale('log')
    ticks = [1, 5, 15, 60, 300, 1200, 3600, 18000]
    ax.set_xticks(ticks)
    ax.set_xticklabels(['1s', '5s', '15s', '1m', '5m', '20m', '1h', '5h'])
    ax.minorticks_off()
    ax.set_xlim(left=1)
    ax.set_xlabel('duration', fontsize=7.5, color=SOFT)
    _legend(ax, ncol=4)


def chart_cp_history(ax, hist):
    _style(ax, 'Threshold estimates (rolling 90-day best)', 'watts')
    drawn = False
    for i, (col, name) in enumerate((('cp', 'Critical power'), ('ftp_20min', '95 % of best 20 min'))):
        s = hist[col].dropna() if hist is not None and not hist.empty else pd.Series(dtype=float)
        if s.empty:
            continue
        ax.plot(s.index, s.values, color=SERIES[i], lw=1.6, marker='o', ms=3.5, label=name)
        drawn = True
    if not drawn:
        return _empty(ax, 'Needs maximal efforts between 3 and 20 minutes.')
    _dates(ax)
    _legend(ax, ncol=2)


def chart_durability(ax, table):
    _style(ax, 'Power kept after accumulated work', '% of fresh best')
    if table is None or table.empty:
        return _empty(ax, 'No ride in this period went beyond 1000 kJ.')
    labels = {5: '5 s', 60: '1 min', 300: '5 min', 1200: '20 min', 3600: '60 min'}
    durations = sorted(table['duration'].unique())
    groups = sorted(table['kj'].unique())
    w = 0.8 / len(groups)
    for i, kj in enumerate(groups):
        g = table[table['kj'] == kj].set_index('duration')['retention_pct']
        xs = [durations.index(d) + (i - (len(groups) - 1) / 2) * w for d in g.index]
        ax.bar(xs, g.values, width=w * 0.92, color=SERIES[i % 4], edgecolor='white', linewidth=0.8,
               label=f"after {kj / 1000:g} MJ")
    ax.set_xticks(range(len(durations)))
    ax.set_xticklabels([labels.get(d, f"{d}s") for d in durations])
    ax.axhline(100, color=PAL['muted'], lw=0.8, ls=':')
    ax.set_ylim(0, 112)
    _legend(ax, ncol=3)


def chart_intensity(ax, dist, freq):
    _style(ax, 'Training intensity distribution', '% of riding time')
    if dist is None or dist.empty:
        return _empty(ax, 'No zone data in this period.')
    width = 5.5 if freq == 'W' else 24
    bottom = np.zeros(len(dist))
    for (col, name), color in zip((('low', 'Easy (Z1–2)'), ('mid', 'Moderate (Z3–4)'), ('high', 'Hard (Z5+)')),
                                  PAL['intensity']):
        vals = dist[col].to_numpy() * 100
        ax.bar(dist.index, vals, bottom=bottom, width=width, color=color, edgecolor='white', linewidth=0.8,
               label=name)
        bottom += vals
    ax.set_ylim(0, 100)
    _dates(ax)
    _legend(ax, ncol=3)


def chart_polarization(ax, dist):
    _style(ax, 'Polarization index', 'a.u.')
    s = dist['pi'].dropna() if dist is not None and not dist.empty else pd.Series(dtype=float)
    if s.empty:
        return _empty(ax, 'Not defined (more hard than easy time, or no zone data).')
    ax.plot(s.index, s.values, color=SERIES[0], lw=1.6, marker='o', ms=3.5)
    ax.axhline(2.0, color=PAL['muted'], lw=0.8, ls=(0, (4, 2)))
    ax.text(s.index[0], 2.0, ' > 2.0: polarized', fontsize=7, color=SOFT, va='bottom')
    _dates(ax)


# --- Pages ---------------------------------------------------------------------------------------

def _period_label(data):
    d, today, start = data['period_days'], data['today'], data['start']
    if start is None:
        return f"Whole history, to {today:%d %b %Y}"
    return f"{start:%d %b %Y} – {today:%d %b %Y} ({d} days)"


def _page_summary(data):
    u, now, t = data['user'], data['now'], data['totals']
    fig = _new_page('Training report', f"{u['name']}  ·  {_period_label(data)}")
    ftp, weight = u['ftp'], u['weight']
    cp = data['cp2']['cp'] if data['cp2'] else None
    _tile_row(fig, 0.895, [
        ('Fitness (CTL)', _fmt(now.get('ctl'), '.0f'), '42-day load'),
        ('Fatigue (ATL)', _fmt(now.get('atl'), '.0f'), '7-day load'),
        ('Form (TSB)', _fmt(now.get('tsb'), '+.0f'), _plain(form_status(now.get('tsb')))),
        ('Ramp rate', _fmt(now.get('ramp'), '+.1f', '/wk'), _plain(ramp_status(now.get('ramp')))),
        ('Acute : chronic', _fmt(now.get('acwr'), '.2f'), _plain(acwr_status(now.get('acwr')))),
        ('FTP', f"{ftp} W", f"{ftp / weight:.2f} W/kg" if weight else ''),
        ('Critical power', _fmt(cp, '.0f', ' W'), 'last 90 days' if cp else 'no max efforts'),
        ('Monotony (7 d)', _fmt(now.get('monotony'), '.2f'), _plain(monotony_status(now.get('monotony')))),
        ('Rides', f"{t['rides']}", f"{t['hours']:.1f} h riding"),
        ('Distance', f"{t['km']:,.0f} km", f"{t['elevation_m']:,.0f} m climbing"),
    ])
    ax = _axes(fig, 0.72, 0.30)
    chart_load(ax, data['pmc'], data['today'])
    y = 0.37
    fig.text(LEFT, y, 'Notes', fontsize=10, fontweight='bold', color=INK, va='top')
    y -= 0.025
    y = _note(fig, y, f"In this period: {t['rides']} rides, {t['hours']:.1f} h, {t['km']:,.0f} km, "
                      f"{t['elevation_m']:,.0f} m of climbing, {t['kj']:,.0f} kJ of work and {t['tss']:,.0f} TSS.",
              size=8.5, color=INK)
    fc = data['forecast']
    if fc:
        y = _note(fig, y, f"With the planned sessions, form on {fc['date']:%a %d %b} would be {fc['tsb']:+.0f} "
                          f"({_plain(form_status(fc['tsb']))}) with fitness at {fc['ctl']:.0f}.", size=8.5, color=INK)
    s = data['suggestion']
    if s:
        if s['direction'] == 'up':
            y = _note(fig, y, f"Your best 20 minutes in the last {s['window_days']} days "
                              f"({s['best_20min_w']:.0f} W, {s['best_20min_date']:%d %b}) suggests an FTP of about "
                              f"{s['ftp']} W, {s['change_w']:+d} W vs the {ftp} W in your profile.", size=8.5, color=INK)
        else:
            y = _note(fig, y, f"Your best 20 minutes in the last {s['window_days']} days implies about {s['ftp']} W, "
                              f"below your profile FTP of {ftp} W. That only means no recent effort backs it up.",
                      size=8.5, color=INK)
    if data['rides_needing_update']:
        y = _note(fig, y, f"{data['rides_needing_update']} rides were saved by an older version of the app and are "
                          "missing some data (run 'Update ride history' in the User Corner to include them).",
                  size=8.5, color=INK)
    _note(fig, y - 0.01, "Load (TSS) is power-based where a power meter was used and heart-rate based otherwise. "
                         "Form = yesterday's fitness minus fatigue. Status words are rules of thumb, not medical advice.")
    return fig


def _page_volume(data):
    fig = _new_page('Volume and load progression', _period_label(data))
    chart_volume(_axes(fig, 0.855, 0.2), data['volume'], data['freq'])
    chart_ramp(_axes(fig, 0.585, 0.2), data['ramp'])
    chart_acwr(_axes(fig, 0.315, 0.2), data['acwr'])
    return fig


def _page_power(data):
    u = data['user']
    fig = _new_page('Power profile', f"{u['name']}  ·  FTP {u['ftp']} W  ·  {u['weight']:.1f} kg")
    chart_power_windows(_axes(fig, 0.855, 0.25), data['curves'], u['weight'])
    cp2, cp3 = data['cp2'], data['cp3']
    _tile_row(fig, 0.54, [
        ('CP (2-parameter)', _fmt(cp2 and cp2['cp'], '.0f', ' W'), 'best 3–20 min, 90 d'),
        ("W′", _fmt(cp2 and cp2['w_prime'] / 1000, '.1f', ' kJ'), 'work above CP'),
        ('CP (3-parameter)', _fmt(cp3 and cp3['cp'], '.0f', ' W'), 'Morton model'),
        ('Pmax (model)', _fmt(cp3 and cp3['pmax'], '.0f', ' W'), 'needs recent sprints'),
    ], cols=4)
    chart_cp_history(_axes(fig, 0.43, 0.17), data['cp_history'])
    prof = data['profile']
    fig.text(LEFT, 0.205, 'Power profile', fontsize=9.5, fontweight='bold', color=INK, va='top')
    ax = _axes(fig, 0.19, 0.15)
    rows = [[r['duration'], _fmt(r['watts'], '.0f', ' W'), _fmt(r['w_per_kg'], '.2f'), _fmt(r['x_ftp'], '.2f'),
             _fmt(r['all_time'], '.2f')] for _, r in prof.iterrows() if pd.notna(r['watts'])]
    if rows:
        _table(ax, ['Duration', 'Last 90 d', 'W/kg (90 d)', '× FTP (90 d)', 'W/kg (all time)'], rows,
               [0.2, 0.2, 0.2, 0.2, 0.2])
    else:
        _empty(ax, 'No power data in the last 90 days.')
    return fig


def _page_durability(data):
    fig = _new_page('Durability and records', _period_label(data))
    chart_durability(_axes(fig, 0.855, 0.24), data['durability'])
    tbl = data['durability']
    ax = _axes(fig, 0.575, 0.12)
    if tbl is not None and not tbl.empty:
        rows = [[f"{r['kj'] / 1000:g} MJ", dur_label(r['duration']), f"{r['fresh']:.0f} W",
                 f"{r['fatigued']:.0f} W", f"{r['retention_pct']:.0f} %"] for _, r in tbl.iterrows()]
        _table(ax, ['After', 'Effort', 'Fresh best', 'Best after', 'Kept'], rows, [0.2] * 5)
    else:
        ax.set_axis_off()
    fig.text(LEFT, 0.43, 'All-time records', fontsize=9.5, fontweight='bold', color=INK, va='top')
    rec = data['records']
    ax = _axes(fig, 0.415, 0.36)
    if rec is not None and not rec.empty:
        rec = rec[rec['duration'].isin(KEY_DURATIONS)]
        w = data['user']['weight']
        rows = [[dur_label(r['duration']), f"{r['watts']:.0f} W", f"{r['watts'] / w:.2f}" if w else '',
                 f"{r['date']:%d %b %Y}", _fmt(r.get('recent'), '.0f', ' W')] for _, r in rec.iterrows()]
        _table(ax, ['Duration', 'Best', 'W/kg', 'Set on', 'Last 90 d'], rows, [0.2] * 5, row_h_in=0.185)
    else:
        _empty(ax, 'No power records yet.')
    return fig


def _page_intensity(data):
    kind = 'power' if data['zone_kind'] == 'power' else 'heart-rate'
    fig = _new_page('Intensity distribution', f"{_period_label(data)}  ·  {kind} zones, moving time")
    sh, pi = data['shares'], data['polarization']
    if sh:
        _tile_row(fig, 0.895, [
            ('Easy (Z1–2)', f"{sh['low']:.0%}", ''), ('Moderate (Z3–4)', f"{sh['mid']:.0%}", ''),
            ('Hard (Z5+)', f"{sh['high']:.0%}", ''),
            ('Polarization index', _fmt(pi, '.2f'), tid_label(sh['low'], sh['mid'], sh['high'], pi)),
        ], cols=4)
    chart_intensity(_axes(fig, 0.79, 0.27), data['distribution'], data['freq'])
    chart_polarization(_axes(fig, 0.45, 0.2), data['distribution'])
    _note(fig, 0.17, "Zones: Coggan power zones from FTP, Friel heart-rate zones from LTHR, grouped into three: "
                     "Z1–2 easy, Z3–4 moderate, Z5+ hard. Polarization index = log10(easy / moderate × hard × 100) on "
                     "time shares (Treff et al. 2019); above 2.0 is polarized, undefined when hard time exceeds easy.")
    return fig


def _page_rides(data):
    fig = _new_page('Recent rides and plan', _period_label(data))
    stats = data['stats'].sort_values('date', ascending=False).head(22)
    fig.text(LEFT, 0.895, 'Recent rides', fontsize=9.5, fontweight='bold', color=INK, va='top')
    ax = _axes(fig, 0.885, 0.5)
    if not stats.empty:
        rows = [[f"{r['date']:%a %d %b}", f"{(r['moving_s'] or 0) / 3600:.1f}", f"{r['distance_km']:.0f}",
                 f"{r['elevation_m']:.0f}", _fmt(r['norm_power'], '.0f'), _fmt(r['tss'], '.0f'),
                 _fmt(r['vi'], '.2f'), _fmt(r['ef'], '.2f'), _fmt(r['decoupling_pct'], '.1f')]
                for _, r in stats.iterrows()]
        _table(ax, ['Date', 'Hours', 'km', 'Climb m', 'NP W', 'TSS', 'VI', 'EF', 'Decoupl. %'], rows,
               [0.17, 0.09, 0.09, 0.12, 0.1, 0.1, 0.09, 0.09, 0.15], row_h_in=0.205)
    else:
        _empty(ax, 'No rides in this period.')
    fig.text(LEFT, 0.355, 'Planned sessions', fontsize=9.5, fontweight='bold', color=INK, va='top')
    plan = data['planned']
    ax = _axes(fig, 0.345, 0.25)
    if plan is not None and not plan.empty:
        rows = []
        for _, r in plan.iterrows():
            detail = ''
            if isinstance(r.get('blocks'), str):
                try:
                    detail = wo.describe(json.loads(r['blocks']))
                except ValueError:
                    pass
            text = r['note'] if isinstance(r['note'], str) and r['note'] else detail
            rows.append([f"{r['date']:%a %d %b}", (text or '')[:58], f"{r['tss']:.0f}"])
        _table(ax, ['Date', 'Session', 'TSS'], rows, [0.17, 0.73, 0.1], aligns=['left', 'left', 'right'])
    else:
        _empty(ax, 'Nothing planned.')
    _note(fig, 0.08, "VI = NP / average power. EF = NP / average heart rate. Decoupling = drift of power:HR between the "
                     "first and second half (meaningful only for steady rides of 20+ min).")
    return fig


def build_report_pdf(data):
    """The report as PDF bytes."""
    pages = [_page_summary(data), _page_volume(data), _page_power(data), _page_durability(data),
             _page_intensity(data), _page_rides(data)]
    left = f"BikeTools Analyzer  ·  {data['user']['name']}  ·  generated {datetime.now():%d %b %Y %H:%M}"
    buf = io.BytesIO()
    with PdfPages(buf, metadata={'Title': f"Training report - {data['user']['name']}",
                                 'Author': 'BikeTools Analyzer', 'CreationDate': datetime.now()}) as pdf:
        for n, fig in enumerate(pages, 1):
            _footer(fig, left, n, len(pages))
            pdf.savefig(fig)
    return buf.getvalue()
