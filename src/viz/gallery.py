"""Start page of the ride analysis: the user's saved rides as cards; open one to analyse it."""
import io

import pandas as pd
import streamlit as st

from src.data.cached import get_user_rides
from src.analysis.ride_list import _when, _source_rank, collapse_duplicates  # noqa: F401  (re-exported)
from src.data.db import get_ride_file
from src.viz.formatting import format_duration
from src.i18n import t, fdate

PAGE_SIZE = 12
GALLERY_KEY = "gallery_file"


def ride_title(filename):
    """Readable name for a stored file: 'strava_123.json' -> 'Strava ride 123', 'morning-ride.fit' -> 'morning-ride'."""
    stem = (filename or "ride").rsplit(".", 1)[0]
    for prefix, label in (("strava_", t("Strava ride {id}")), ("wahoo_", t("Wahoo ride {id}"))):
        if stem.lower().startswith(prefix):
            return label.replace('{id}', stem[len(prefix):])
    return stem


def filter_rides(rides, query="", limit=PAGE_SIZE):
    """Newest first; keeps rides whose title or date contains every word of `query`. Returns (shown, total_matching)."""
    words = (query or "").lower().split()
    out = []
    for r in sorted(collapse_duplicates(rides), key=lambda r: str(r.get("date_time") or ""), reverse=True):
        ts = _when(r)
        hay = f"{ride_title(r.get('filename'))} {fdate(ts, '%d %b %Y %A %B')}".lower() if ts is not None \
            else ride_title(r.get('filename')).lower()
        if all(w in hay for w in words):
            out.append(r)
    return out[:limit], len(out)


def week_summary(rides, today=None):
    """Rides, hours and TSS of the last 7 days (today included)."""
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    n, secs, tss = 0, 0.0, 0.0
    for r in collapse_duplicates(rides):
        ts = _when(r)
        if ts is not None and today - pd.Timedelta(days=6) <= ts.normalize() <= today:
            n += 1
            secs += float(r.get("moving_s") or 0)
            tss += float(r.get("tss") or 0)
    return {"rides": n, "hours": secs / 3600.0, "tss": tss}


def _num(v, spec, unit=""):
    return "–" if v is None or pd.isna(v) else f"{v:{spec}}{unit}"


def _card(r):
    ts = _when(r)
    moving = r.get("moving_s")
    with st.container(border=True):
        st.markdown(f"**{ride_title(r.get('filename'))}**")
        st.caption(fdate(ts, "%a %d %b %Y · %H:%M") if ts is not None else t("date unknown"))
        duration = format_duration(int(moving)) if moving is not None and not pd.isna(moving) else "–"
        st.markdown(f"{_num(r.get('distance_km'), '.1f', ' km')} · ↑ {_num(r.get('elevation_m'), '.0f', ' m')} · {duration}")
        st.caption(f"NP {_num(r.get('norm_power'), '.0f', ' W')} · TSS {_num(r.get('tss'), '.0f')}")
        if st.button(t("Open ride"), key=f"open_ride_{r['id']}", use_container_width=True):
            name, content = get_ride_file(int(r["id"]))
            if content is None:
                st.error(t("The stored file for this ride is missing."))
            else:
                f = io.BytesIO(content)
                f.name = name
                st.session_state[GALLERY_KEY] = f
                st.rerun()


def render_gallery(user_name):
    """Cards of the latest saved rides, with search and 'show more'. Needs a selected user."""
    if not user_name:
        return
    rides = get_user_rides(user_name)
    if not rides:
        st.caption(t("Rides you analyse are saved automatically and will appear here."))
        return
    wk = week_summary(rides)
    st.subheader(t("Your rides"))
    a, b, c, d = st.columns([1, 1, 1, 2])
    a.metric(t("Last 7 days"), f"{wk['rides']} rides")
    b.metric(t("Time"), f"{wk['hours']:.1f} h")
    c.metric("TSS", f"{wk['tss']:.0f}")
    query = d.text_input(t("Search rides"), key="gallery_query", placeholder=t("name, month, weekday…"))
    shown_n = st.session_state.get("gallery_shown", PAGE_SIZE)
    shown, total = filter_rides(rides, query, shown_n)
    if not shown:
        st.info(t("No saved ride matches that search."))
        return
    for start in range(0, len(shown), 3):
        for col, r in zip(st.columns(3), shown[start:start + 3]):
            with col:
                _card(r)
    if total > len(shown) and st.button(t('Show more ({total} older)', total=total - len(shown)), key="gallery_more"):
        st.session_state["gallery_shown"] = shown_n + PAGE_SIZE
        st.rerun()
