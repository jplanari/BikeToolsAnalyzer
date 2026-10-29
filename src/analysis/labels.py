"""Plain-text helpers for the Training page and the PDF report: periods, durations and the
rule-of-thumb status labels (no Streamlit, no plotting)."""
import pandas as pd

PERIODS = {'6 weeks': 42, '3 months': 91, '6 months': 182, '1 year': 365, 'All': None}


def period_start(label, today):
    days = PERIODS[label]
    return None if days is None else today - pd.Timedelta(days=days - 1)


def dur_label(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m" if m else f"{h}h"
    if m:
        return f"{m}m{s:02d}s" if s else f"{m}m"
    return f"{s}s"


# --- Status labels (rules of thumb; shown in each metric's help) ---------------

def ramp_status(v):
    if v is None or pd.isna(v):
        return None
    if v > 8:
        return "⚠️ Aggressive"
    if v >= 3:
        return "↗ Building"
    if v > -3:
        return "→ Maintaining"
    return "↘ Losing fitness"


def acwr_status(v):
    if v is None or pd.isna(v):
        return None
    if v < 0.8:
        return "↘ Low load"
    if v <= 1.3:
        return "✅ In range"
    if v <= 1.5:
        return "⚠️ High"
    return "⛔ Spike"


def form_status(v):
    if v is None or pd.isna(v):
        return None
    if v > 25:
        return "Very fresh"
    if v > 5:
        return "✅ Fresh"
    if v >= -10:
        return "Neutral"
    if v >= -30:
        return "Productive fatigue"
    return "⚠️ Very fatigued"


def monotony_status(v):
    if v is None or pd.isna(v):
        return None
    return "⚠️ High" if v > 2.0 else "✅ Varied"


def tid_label(low, mid, high, pi):
    if pi is not None and pi > 2.0:
        return "Polarized"
    if low >= mid >= high:
        return "Pyramidal"
    if mid > low and mid > high:
        return "Threshold-focused"
    return "Non-polarized"
