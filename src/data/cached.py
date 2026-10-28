"""
Cached reads for the Streamlit pages.

Streamlit re-runs the whole script on every widget click, so a page that queries the
database and recomputes its analytics on each run gets slow as the history grows. These
wrappers cache the results; the key includes db.data_stamp(), which changes whenever the
database file is written, so anything saved or updated shows up on the next run.

Use these in the UI only. Code that writes should keep calling src.data.db directly.
"""
import streamlit as st

from src.data import db
from src.analysis import training_load as tl
from src.analysis import power_profile as pp


@st.cache_data(show_spinner=False, max_entries=256)
def _read(name, stamp, args, kwargs):
    return getattr(db, name)(*args, **dict(kwargs))


def _cached(name):
    def reader(*args, **kwargs):
        return _read(name, db.data_stamp(), args, tuple(sorted(kwargs.items())))
    reader.__name__ = name
    reader.__doc__ = getattr(db, name).__doc__
    return reader


get_user_data = _cached('get_user_data')
get_user_rides = _cached('get_user_rides')
get_user_best_power = _cached('get_user_best_power')
get_user_tss_history = _cached('get_user_tss_history')
count_rides_needing_update = _cached('count_rides_needing_update')
get_hist = _cached('get_hist')
get_ride_stats = _cached('get_ride_stats')
period_coverage = _cached('period_coverage')
get_power_records = _cached('get_power_records')
get_kj_records = _cached('get_kj_records')
get_zone_rows = _cached('get_zone_rows')
get_planned_sessions = _cached('get_planned_sessions')


# Analytics that depend only on their inputs: cached on the inputs themselves.
@st.cache_data(show_spinner=False, max_entries=32)
def compute_pmc(history, today=None, planned=None):
    return tl.compute_pmc(history, today=today, planned=planned)


@st.cache_data(show_spinner=False, max_entries=32)
def cp_history(recs, today=None):
    return pp.cp_history(recs, today=today)
