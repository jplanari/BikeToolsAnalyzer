import numpy as np
import pandas as pd

CTL_DAYS = 42
ATL_DAYS = 7


def compute_pmc(history, until=None, ctl_days=CTL_DAYS, atl_days=ATL_DAYS, planned=None, today=None):
    """
    Performance Management Chart from a ride history.

    history: iterable of {'date': <datetime-like>, 'tss': <number>}
    until:   last day of the chart (default: today). Days without rides get
             TSS = 0, so CTL/ATL keep decaying up to this date.

    Uses the standard impulse-response constants
        k = 1 - exp(-1/T)   (T = 42 d for CTL, 7 d for ATL)
    and TSB(day) = CTL(day-1) - ATL(day-1), i.e. the form you START the day
    with (the TrainingPeaks convention).

    planned: optional iterable of {'date', 'tss'} for days AFTER `today`
             (default: the current date). The chart then runs to the last
             planned day, forecasting CTL/ATL/TSB; those rows have
             planned=True. Planned entries on or before today are ignored.

    Returns a DataFrame indexed by day with columns tss, CTL, ATL, TSB,
    planned (empty if there is no valid data).
    """
    df = pd.DataFrame(list(history), columns=['date', 'tss'])
    if df.empty:
        return pd.DataFrame(columns=['tss', 'CTL', 'ATL', 'TSB'])

    # Parse element-wise: rows may mix ISO strings with/without timezone, and
    # vectorised parsing infers one format from the first row, dropping the rest.
    df['date'] = df['date'].map(lambda d: pd.to_datetime(d, errors='coerce', utc=True))
    df['tss'] = pd.to_numeric(df['tss'], errors='coerce')
    df = df.dropna(subset=['date', 'tss'])
    df = df[df['tss'] >= 0]
    if df.empty:
        return pd.DataFrame(columns=['tss', 'CTL', 'ATL', 'TSB'])

    df['day'] = df['date'].dt.tz_convert(None).dt.normalize()
    daily = df.groupby('day')['tss'].sum()

    today = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.today().normalize()
    end = pd.Timestamp(until).normalize() if until is not None else today
    end = max(end, daily.index.max())

    plan = _daily_plan(planned, today)
    if not plan.empty:
        end = max(end, plan.index.max())

    idx = pd.date_range(daily.index.min(), end, freq='D')
    daily = daily.reindex(idx, fill_value=0.0).astype(float)
    is_planned = pd.Series(False, index=idx)
    if not plan.empty:
        daily.loc[plan.index] = plan.to_numpy()
        is_planned.loc[plan.index] = True

    k_ctl = 1.0 - np.exp(-1.0 / ctl_days)
    k_atl = 1.0 - np.exp(-1.0 / atl_days)

    out = pd.DataFrame({'tss': daily})
    # Start from zero load before the first ride
    out['CTL'] = _ewma_from_zero(daily.to_numpy(), k_ctl)
    out['ATL'] = _ewma_from_zero(daily.to_numpy(), k_atl)
    prev_ctl = out['CTL'].shift(1, fill_value=0.0)
    prev_atl = out['ATL'].shift(1, fill_value=0.0)
    out['TSB'] = prev_ctl - prev_atl
    out['planned'] = is_planned
    out.index.name = 'date'
    return out


def _daily_plan(planned, today):
    """Planned sessions after `today`, summed per day."""
    if not planned:
        return pd.Series(dtype=float)
    df = pd.DataFrame(list(planned), columns=['date', 'tss'])
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
    df['tss'] = pd.to_numeric(df['tss'], errors='coerce')
    df = df.dropna()
    df = df[(df['date'] > today) & (df['tss'] >= 0)]
    return df.groupby('date')['tss'].sum()


def ramp_rate(pmc, days=7):
    """CTL change over the previous `days` (fitness gained per week)."""
    if pmc is None or pmc.empty:
        return pd.Series(dtype=float)
    return pmc['CTL'] - pmc['CTL'].shift(days)


def monotony_strain(pmc, window=7):
    """
    Foster's training monotony and strain over a rolling week:
        monotony = mean daily load / standard deviation of daily load
        strain   = weekly load x monotony
    Rest days count as 0. Undefined (NaN) when every day had the same load.
    """
    if pmc is None or pmc.empty:
        return pd.DataFrame(columns=['monotony', 'strain'])
    tss = pmc['tss']
    mean = tss.rolling(window, min_periods=window).mean()
    sd = tss.rolling(window, min_periods=window).std(ddof=0)
    monotony = (mean / sd).where(sd > 0)
    strain = tss.rolling(window, min_periods=window).sum() * monotony
    return pd.DataFrame({'monotony': monotony, 'strain': strain})


def acwr(pmc, min_history_days=28):
    """
    Acute:chronic workload ratio, as ATL / CTL from the exponentially
    weighted loads (EWMA-ACWR). ~0.8-1.3 is the commonly cited range.
    NaN for the first `min_history_days` of data (with almost no chronic
    load the ratio is meaningless, e.g. 5.7 on day one) and while CTL ~ 0.
    """
    if pmc is None or pmc.empty:
        return pd.Series(dtype=float)
    ratio = (pmc['ATL'] / pmc['CTL']).where(pmc['CTL'] > 1.0)
    return ratio.where(pmc.index >= pmc.index[0] + pd.Timedelta(days=min_history_days))


def volume(stats, freq='W'):
    """
    Training volume per week ('W', Monday start) or month ('M') from
    per-ride stats (columns date, moving_s, distance_km, elevation_m, kj, tss).
    """
    cols = ['rides', 'hours', 'km', 'elevation_m', 'kj', 'tss']
    if stats is None or stats.empty:
        return pd.DataFrame(columns=cols)
    df = stats.dropna(subset=['date']).copy()
    period = 'W-SUN' if freq == 'W' else 'M'
    df['period'] = df['date'].dt.to_period(period).dt.start_time
    g = df.groupby('period')
    out = pd.DataFrame({
        'rides': g.size(),
        'hours': g['moving_s'].sum(min_count=1).fillna(0) / 3600,
        'km': g['distance_km'].sum(),
        'elevation_m': g['elevation_m'].sum(),
        'kj': g['kj'].sum(),
        'tss': g['tss'].sum(),
    })
    # Include empty periods so gaps show as zero weeks
    full = pd.period_range(out.index.min(), out.index.max(), freq=period).start_time
    out = out.reindex(full, fill_value=0)
    out.index.name = 'period'
    return out


def _ewma_from_zero(values, k):
    """x_t = x_{t-1} + k * (v_t - x_{t-1}), starting from x_{-1} = 0."""
    res = np.empty(len(values), dtype=float)
    x = 0.0
    for i, v in enumerate(values):
        x += k * (v - x)
        res[i] = x
    return res
