"""Ride-list helpers shared by the gallery and the phone API (no Streamlit)."""
import pandas as pd

from src.data.imports.strava_import import DUPLICATE_START_TOLERANCE_S


def _when(row):
    ts = pd.to_datetime(row.get("start_local") or row.get("date_time"), errors="coerce")   # the rider's own clock
    return None if pd.isna(ts) else ts


def _source_rank(filename):
    """Which copy of the same ride to keep: the recorder's FIT file first, then other uploads, Strava's rebuilt
    streams last (lower rank wins)."""
    name = (filename or "").lower()
    return 2 if name.endswith(".json") else (0 if name.endswith(".fit") else 1)


def collapse_duplicates(rides, tolerance_s=DUPLICATE_START_TOLERANCE_S):
    """The same ride saved from two sources (Strava and Wahoo, or an upload) starts at the same moment: keep one."""
    dated = sorted((r for r in rides if _when(r) is not None), key=_when)
    undated = [r for r in rides if _when(r) is None]
    kept = []
    for r in dated:
        if kept and abs((_when(r) - _when(kept[-1])).total_seconds()) <= tolerance_s:
            if _source_rank(r.get("filename")) < _source_rank(kept[-1].get("filename")):
                kept[-1] = r
        else:
            kept.append(r)
    return kept + undated
