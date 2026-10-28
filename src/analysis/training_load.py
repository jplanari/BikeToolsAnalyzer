import numpy as np
import pandas as pd

CTL_DAYS = 42
ATL_DAYS = 7


def compute_pmc(history, until=None, ctl_days=CTL_DAYS, atl_days=ATL_DAYS):
    """
    Performance Management Chart from a ride history.

    history: iterable of {'date': <datetime-like>, 'tss': <number>}
    until:   last day of the chart (default: today). Days without rides get
             TSS = 0, so CTL/ATL keep decaying up to this date.

    Uses the standard impulse-response constants
        k = 1 - exp(-1/T)   (T = 42 d for CTL, 7 d for ATL)
    and TSB(day) = CTL(day-1) - ATL(day-1), i.e. the form you START the day
    with (the TrainingPeaks convention).

    Returns a DataFrame indexed by day with columns tss, CTL, ATL, TSB
    (empty if there is no valid data).
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

    end = pd.Timestamp(until).normalize() if until is not None else pd.Timestamp.today().normalize()
    end = max(end, daily.index.max())
    idx = pd.date_range(daily.index.min(), end, freq='D')
    daily = daily.reindex(idx, fill_value=0.0).astype(float)

    k_ctl = 1.0 - np.exp(-1.0 / ctl_days)
    k_atl = 1.0 - np.exp(-1.0 / atl_days)

    out = pd.DataFrame({'tss': daily})
    # Start from zero load before the first ride
    out['CTL'] = _ewma_from_zero(daily.to_numpy(), k_ctl)
    out['ATL'] = _ewma_from_zero(daily.to_numpy(), k_atl)
    prev_ctl = out['CTL'].shift(1, fill_value=0.0)
    prev_atl = out['ATL'].shift(1, fill_value=0.0)
    out['TSB'] = prev_ctl - prev_atl
    out.index.name = 'date'
    return out


def _ewma_from_zero(values, k):
    """x_t = x_{t-1} + k * (v_t - x_{t-1}), starting from x_{-1} = 0."""
    res = np.empty(len(values), dtype=float)
    x = 0.0
    for i, v in enumerate(values):
        x += k * (v - x)
        res[i] = x
    return res
