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


# --- Time zones ----------------------------------------------------------------------------
# Rides are stored as UTC instants. "Which day was this ride" is a question about the rider's own clock, so
# the day filters and the day buckets of the analytics use the rider's time zone (an IANA name such as
# 'Europe/Madrid'). Without one, the computer's own zone is used, and UTC when that cannot be found.

def valid_timezone(name):
    """The IANA name when it is usable on this machine, else None."""
    if not name:
        return None
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo(str(name))
        return str(name)
    except Exception:                      # unknown name, or no tz database (Windows without tzdata)
        return None


def system_timezone():
    """This computer's IANA time zone, or 'UTC' when it cannot be told."""
    import os
    for candidate in (os.environ.get("BIKETOOLS_TZ"), os.environ.get("TZ")):
        if valid_timezone((candidate or "").lstrip(":")):
            return candidate.lstrip(":")
    try:
        target = os.path.realpath("/etc/localtime")
        if "zoneinfo/" in target and valid_timezone(target.split("zoneinfo/", 1)[1]):
            return target.split("zoneinfo/", 1)[1]
        with open("/etc/timezone") as fh:
            name = fh.read().strip()
        if valid_timezone(name):
            return name
    except OSError:
        pass
    return "UTC"


def resolve_timezone(name=None):
    return valid_timezone(name) or system_timezone()


def to_utc_iso(value):
    """Any datetime-like (naive = UTC) -> 'YYYY-MM-DDTHH:MM:SS+00:00', or None when it is empty or unreadable."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == '':
        return None
    ts = pd.to_datetime(value, errors='coerce', utc=True)
    if pd.isna(ts):
        return None
    return ts.strftime('%Y-%m-%dT%H:%M:%S+00:00')


def to_local_naive(values, tz):
    """Datetime-like values (UTC instants) -> naive datetimes on the clock of time zone `tz`."""
    utc = pd.to_datetime(parse_utc(values)).dt.tz_localize('UTC')
    return utc.dt.tz_convert(resolve_timezone(tz)).dt.tz_localize(None)


def local_day_bounds_utc(start, end, tz):
    """
    Inclusive local dates -> (first_instant, end_instant) as stored UTC strings, either None when open-ended:
    start 00:00 local, and the midnight after `end` local.
    """
    zone = resolve_timezone(tz)

    def instant(day):
        ts = pd.Timestamp(day).normalize().tz_localize(zone, nonexistent='shift_forward', ambiguous=True)
        return ts.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%S+00:00')
    return (instant(start) if start is not None else None,
            instant(pd.Timestamp(end) + pd.Timedelta(days=1)) if end is not None else None)
