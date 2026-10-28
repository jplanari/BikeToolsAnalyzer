"""
Charts for the Training page.

Colours follow the validated data-viz palette (categorical slots in fixed
order, a single-hue blue ramp for ordered intensity), with separate light and
dark steps. Every chart uses one y-axis; related measures on different
scales get their own chart.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

PALETTES = {
    'light': {
        'series': ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'],
        # easy -> hard: light -> dark (easy recedes toward the light surface)
        'intensity': ['#86b6ef', '#2a78d6', '#184f95'],
        'muted': '#8a8985', 'grid': 'rgba(0,0,0,0.08)', 'pos': '#1baf7a', 'neg': '#e34948',
    },
    'dark': {
        'series': ['#3987e5', '#d95926', '#199e70', '#c98500'],
        # easy -> hard: dark -> light (easy recedes toward the dark surface)
        'intensity': ['#184f95', '#2a78d6', '#6da7ec'],
        'muted': '#8f8e88', 'grid': 'rgba(255,255,255,0.08)', 'pos': '#199e70', 'neg': '#e66767',
    },
}


def palette(mode):
    return PALETTES['dark' if mode == 'dark' else 'light']


def _layout(fig, title, y_title, pal, x_title=None, legend=True, **kw):
    kw.setdefault('hovermode', 'x unified')
    fig.update_layout(
        title=title, yaxis_title=y_title, xaxis_title=x_title,
        showlegend=legend, bargap=0.25,
        legend=dict(orientation='h', y=1.12, x=0),
        margin=dict(t=60, b=40, l=10, r=10), **kw)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=pal['grid'], zeroline=False)
    return fig


def _today_line(fig, today, pal):
    fig.add_vline(x=today, line=dict(color=pal['muted'], width=1, dash='dot'))
    fig.add_annotation(x=today, y=1, yref='paper', text='today', showarrow=False,
                       yanchor='bottom', font=dict(size=11, color=pal['muted']))


# --- Load ------------------------------------------------------------------

def plot_training_load(pmc, today, mode='light'):
    """CTL and ATL lines, TSB bars; the planned (forecast) part is dashed / faded."""
    pal = palette(mode)
    fig = go.Figure()
    if pmc is None or pmc.empty:
        return fig
    actual = pmc[~pmc['planned']]
    plan = pmc[pmc['planned']]
    # Join the forecast to the last actual day so the line is continuous
    plan_line = pd.concat([actual.tail(1), plan]) if not plan.empty else plan

    tsb_colors = np.where(pmc['TSB'] >= 0, pal['pos'], pal['neg'])
    fig.add_trace(go.Bar(x=pmc.index, y=pmc['TSB'], name='Form (TSB)',
                         marker=dict(color=tsb_colors,
                                     opacity=np.where(pmc['planned'], 0.35, 0.6).tolist()),
                         hovertemplate='%{y:.1f}'))
    for col, name, color in [('CTL', 'Fitness (CTL)', pal['series'][0]),
                             ('ATL', 'Fatigue (ATL)', pal['series'][1])]:
        fig.add_trace(go.Scatter(x=actual.index, y=actual[col], name=name, mode='lines',
                                 line=dict(color=color, width=2), hovertemplate='%{y:.1f}'))
        if not plan_line.empty:
            fig.add_trace(go.Scatter(x=plan_line.index, y=plan_line[col], name=f'{name} · plan',
                                     mode='lines', line=dict(color=color, width=2, dash='dash'),
                                     showlegend=False, hovertemplate='%{y:.1f} (plan)'))
    _today_line(fig, today, pal)
    return _layout(fig, 'Fitness, fatigue and form', 'Training load / form', pal)


def plot_ramp_rate(ramp, mode='light', aggressive=8.0):
    """Weekly CTL change; a dashed guide at the 'aggressive' level."""
    pal = palette(mode)
    fig = go.Figure()
    if ramp is None or ramp.dropna().empty:
        return fig
    weekly = ramp.resample('W-SUN').last().dropna()
    fig.add_trace(go.Bar(x=weekly.index, y=weekly.values, name='Ramp rate',
                         marker_color=pal['series'][0], hovertemplate='%{y:+.1f} CTL / week'))
    fig.add_hline(y=aggressive, line=dict(color=pal['muted'], width=1, dash='dash'),
                  annotation_text=f'+{aggressive:g}: aggressive', annotation_position='top left',
                  annotation_font=dict(size=11, color=pal['muted']))
    return _layout(fig, 'Ramp rate (fitness change per week)', 'CTL points / week', pal, legend=False)


def plot_acwr(ratio, mode='light'):
    """ACWR with the commonly cited 0.8-1.3 band shaded."""
    pal = palette(mode)
    fig = go.Figure()
    r = ratio.dropna() if ratio is not None else pd.Series(dtype=float)
    if r.empty:
        return fig
    fig.add_hrect(y0=0.8, y1=1.3, fillcolor=pal['pos'], opacity=0.10, line_width=0)
    fig.add_annotation(x=r.index[0], y=1.3, text='0.8–1.3', showarrow=False, xanchor='left',
                       yanchor='bottom', font=dict(size=11, color=pal['muted']))
    fig.add_trace(go.Scatter(x=r.index, y=r.values, name='ACWR', mode='lines',
                             line=dict(color=pal['series'][0], width=2), hovertemplate='%{y:.2f}'))
    return _layout(fig, 'Acute : chronic load (ATL / CTL)', 'ratio', pal, legend=False)


def plot_volume(vol, metric, label, mode='light'):
    pal = palette(mode)
    fig = go.Figure()
    if vol is None or vol.empty:
        return fig
    fmt = '%{y:.1f}' if metric in ('hours',) else '%{y:,.0f}'
    fig.add_trace(go.Bar(x=vol.index, y=vol[metric], name=label, marker_color=pal['series'][0],
                         hovertemplate=fmt + f' {label}<extra></extra>'))
    return _layout(fig, None, label, pal, legend=False, hovermode='closest')


# --- Power profile -----------------------------------------------------------

_TICKS = dict(tickvals=[1, 5, 15, 60, 300, 1200, 3600, 18000],
              ticktext=['1s', '5s', '15s', '1m', '5m', '20m', '1h', '5h'])


def plot_power_windows(curves, mode='light', weight=None):
    """Best power curves of several windows on one log-time axis (<= 4 series)."""
    pal = palette(mode)
    fig = go.Figure()
    for i, (label, c) in enumerate(curves.items()):
        if c is None or c.empty:
            continue
        c = c.sort_values('duration')
        custom = np.stack([c['date'].dt.strftime('%d %b %Y'),
                           (c['watts'] / weight).round(2) if weight else c['watts'] * np.nan], axis=1)
        fig.add_trace(go.Scatter(
            x=c['duration'], y=c['watts'], name=label, mode='lines',
            line=dict(color=pal['series'][i % 4], width=2.5 if i == 0 else 2),
            customdata=custom,
            hovertemplate='%{y:.0f} W · %{customdata[1]} W/kg · %{customdata[0]}'))
    fig.update_xaxes(type='log', **_TICKS)
    return _layout(fig, 'Best power by period', 'Watts', pal, x_title='Duration')


def plot_cp_history(hist, mode='light'):
    """CP and 95 % of 20-min power over time (both in watts: one axis)."""
    pal = palette(mode)
    fig = go.Figure()
    if hist is None or hist.dropna(how='all').empty:
        return fig
    for i, (col, name) in enumerate([('cp', 'Critical Power'), ('ftp_20min', '95 % of best 20 min')]):
        s = hist[col].dropna()
        if s.empty:
            continue
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode='lines+markers',
                                 line=dict(color=pal['series'][i], width=2), marker=dict(size=8),
                                 hovertemplate='%{y:.0f} W'))
    return _layout(fig, 'Threshold estimates over time (rolling 90-day best)', 'Watts', pal)


def plot_durability(table, mode='light'):
    """Retention (%) after 1/2/3 MJ, grouped by effort duration."""
    pal = palette(mode)
    fig = go.Figure()
    if table is None or table.empty:
        return fig
    labels = {5: '5 s', 60: '1 min', 300: '5 min', 1200: '20 min', 3600: '60 min'}
    for i, (kj, g) in enumerate(table.groupby('kj')):
        g = g.sort_values('duration')
        fig.add_trace(go.Bar(
            x=[labels.get(d, f'{d}s') for d in g['duration']], y=g['retention_pct'],
            name=f'after {kj / 1000:g} MJ', marker_color=pal['series'][i % 4],
            customdata=np.stack([g['fatigued'], g['fresh']], axis=1),
            hovertemplate='%{y:.0f} % (%{customdata[0]:.0f} of %{customdata[1]:.0f} W)'))
    fig.add_hline(y=100, line=dict(color=pal['muted'], width=1, dash='dot'))
    return _layout(fig, 'Power kept after accumulated work', '% of fresh best', pal,
                   barmode='group', yaxis_range=[0, 110])


# --- Intensity distribution ------------------------------------------------------

def plot_intensity_distribution(dist, mode='light'):
    """Share of time easy / moderate / hard per period (100 % stacked, ordered ramp)."""
    pal = palette(mode)
    fig = go.Figure()
    if dist is None or dist.empty:
        return fig
    names = [('low', 'Easy (Z1–2)'), ('mid', 'Moderate (Z3–4)'), ('high', 'Hard (Z5+)')]
    for (col, name), color in zip(names, pal['intensity']):
        fig.add_trace(go.Bar(x=dist.index, y=dist[col] * 100, name=name, marker_color=color,
                             marker_line=dict(width=0), customdata=dist['hours'],
                             hovertemplate='%{y:.0f} %'))
    return _layout(fig, 'Training intensity distribution', '% of riding time', pal,
                   barmode='stack', yaxis_range=[0, 100])


def plot_polarization(dist, mode='light'):
    pal = palette(mode)
    fig = go.Figure()
    s = dist['pi'].dropna() if dist is not None and not dist.empty else pd.Series(dtype=float)
    if s.empty:
        return fig
    fig.add_trace(go.Scatter(x=s.index, y=s.values, name='Polarization index', mode='lines+markers',
                             line=dict(color=pal['series'][0], width=2), marker=dict(size=8),
                             hovertemplate='%{y:.2f}'))
    fig.add_hline(y=2.0, line=dict(color=pal['muted'], width=1, dash='dash'),
                  annotation_text='> 2.0: polarized', annotation_position='top left',
                  annotation_font=dict(size=11, color=pal['muted']))
    return _layout(fig, 'Polarization index', 'a.u.', pal, legend=False)
