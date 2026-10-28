"""Date parsing shared by the database layer and the analytics."""
import pandas as pd


def parse_utc(values):
    """
    Datetime-like values (ISO strings with or without an offset, Timestamps, None) ->
    naive-UTC datetime64[ns] Series. Histories have few distinct strings but many rows, so
    each distinct value is parsed once and mapped back; unparseable values become NaT.
    Parsing is per value (not inferred from the first row), so formats may be mixed.
    """
    series = values if isinstance(values, pd.Series) else pd.Series(list(values), dtype=object)
    uniq = pd.unique(series.dropna())
    if len(uniq) == 0:
        return pd.Series(pd.NaT, index=series.index, dtype='datetime64[ns]')
    try:
        parsed = pd.to_datetime(pd.Series(uniq), errors='coerce', utc=True, format='ISO8601')
    except (ValueError, TypeError):          # something that is not ISO 8601: slower, forgiving path
        parsed = pd.Series([pd.to_datetime(d, errors='coerce', utc=True) for d in uniq])
    parsed = pd.Series(pd.DatetimeIndex(parsed).tz_convert(None))
    out = series.map(dict(zip(uniq, parsed)))
    return pd.to_datetime(out).astype('datetime64[ns]')
