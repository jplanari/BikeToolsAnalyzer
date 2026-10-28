"""Small display helpers shared by the pages."""
import streamlit as st


def show_centered(fig):
    c1, c2, c3 = st.columns([1, 3, 1])
    with c2: st.plotly_chart(fig, use_container_width=True)


def duration_label(seconds):
    """Compact label for power-curve durations: 5s, 1m30s, 7m, 1h30m."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m" if m else f"{h}h"
    if m:
        return f"{m}m{s:02d}s" if s else f"{m}m"
    return f"{s}s"


def format_duration(seconds):
    if not seconds: return "00:00:00"

    m, s =divmod(int(seconds), 60)
    h, m =divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02}"
