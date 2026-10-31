"""
Looks for the app. A look is a small CSS pack layered on Streamlit's own theme (.streamlit/config.toml), written
with colours derived from the text colour (currentColor) so it works in light and dark mode alike.

  Classic    Streamlit's default widgets, untouched.
  Dashboard  Metric cards with an accent edge, pill-style selectors, rounded panels, roomier spacing.
  Compact    The same cards, tighter: more on a small laptop screen.

Navigation is a bar at the top of the page.
"""
import streamlit as st

LOOKS = ["Compact", "Dashboard", "Classic"]
DEFAULT_LOOK = "Compact"
ACCENT = "#FF5A1F"                              # same as primaryColor in .streamlit/config.toml

_EDGE = "color-mix(in srgb, currentColor 14%, transparent)"
_FILL = "color-mix(in srgb, currentColor 5%, transparent)"

_CARDS = f"""
[data-testid="stMetric"] {{
  background: {_FILL}; border: 1px solid {_EDGE}; border-left: 4px solid {ACCENT};
  border-radius: 12px; padding: .65rem .9rem;
}}
[data-testid="stMetricLabel"] p {{ font-size: .75rem; letter-spacing: .05em; text-transform: uppercase; opacity: .72; }}
[data-testid="stMetricValue"] {{ font-weight: 650; }}
[data-testid="stExpander"] {{ border: 1px solid {_EDGE}; border-radius: 12px; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 12px; }}
.stButton > button, .stDownloadButton > button, [data-testid="stLinkButton"] a {{ border-radius: 10px; }}
[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}
[data-testid="stSidebar"] [data-testid="stMetric"] {{ border-left-width: 3px; }}
/* selectors in the page body (sections, periods, navigation) look like pills */
[data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] {{ gap: .4rem; flex-wrap: wrap; }}
[data-testid="stMain"] [data-testid="stRadio"] label {{
  border: 1px solid {_EDGE}; border-radius: 999px; padding: .2rem .85rem; background: {_FILL}; cursor: pointer;
}}
[data-testid="stMain"] [data-testid="stRadio"] label > div:first-child {{ display: none; }}
[data-testid="stMain"] [data-testid="stRadio"] label:has(input:checked) {{
  background: {ACCENT}; border-color: {ACCENT}; color: #fff;
}}
[data-testid="stMain"] [data-testid="stRadio"] label:has(input:checked) p {{ color: #fff; font-weight: 600; }}
.bt-header {{ display: flex; align-items: baseline; gap: .8rem; margin: 0 0 .6rem; }}
.bt-header h1 {{ margin: 0; padding: 0; font-size: 1.9rem; letter-spacing: -.02em; }}
.bt-header span {{ opacity: .6; font-size: .9rem; }}
"""

_DASHBOARD = _CARDS + """
[data-testid="stMainBlockContainer"] { padding-top: 2.2rem; max-width: 1500px; }
h2 { margin-top: 1.4rem; padding-bottom: .25rem; border-bottom: 2px solid """ + ACCENT + """; display: inline-block; }
footer { visibility: hidden; }
"""

_COMPACT = _CARDS + """
[data-testid="stMainBlockContainer"] { padding-top: 1rem; padding-bottom: 2rem; max-width: 1700px; }
[data-testid="stMetric"] { padding: .4rem .65rem; }
[data-testid="stMetricValue"] { font-size: 1.3rem; }
[data-testid="stVerticalBlock"] { gap: .6rem; }
h2 { font-size: 1.35rem; margin-top: .8rem; } h3 { font-size: 1.1rem; }
.bt-header h1 { font-size: 1.4rem; }
footer { visibility: hidden; }
"""

CSS = {"Dashboard": _DASHBOARD, "Compact": _COMPACT, "Classic": ""}


def css_for(look):
    return CSS.get(look, "")


def _query_value(name, allowed, default):
    try:
        v = st.query_params.get(name)
    except Exception:                       # query params are not available everywhere (older Streamlit, tests)
        v = None
    return v if v in allowed else default


def current_look():
    """The chosen look: this session's widget value, else the ?look= of the address (so a bookmark keeps it)."""
    return st.session_state.get("look") or _query_value("look", LOOKS, DEFAULT_LOOK)


def apply_look():
    css = css_for(current_look())
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def appearance_controls():
    """Sidebar box to switch look; the choice is also kept in the page address."""
    with st.sidebar.expander("🎨 Appearance"):
        look = st.selectbox("Look", LOOKS, index=LOOKS.index(current_look()), key="look",
                            help="Compact: dense cards. Dashboard: roomier. Classic: Streamlit's default.")
        st.caption("Light/dark follows your system; change it in ⋮ → Settings.")
    try:
        st.query_params["look"] = look
    except Exception:
        pass


def page_header(subtitle=""):
    st.markdown(f'<div class="bt-header"><h1>🚴 BikeTools Analyzer</h1><span>{subtitle}</span></div>',
                unsafe_allow_html=True)


def welcome(has_user=True):
    """What to do first, when no ride is open. Without a profile the page is only the profile form."""
    if not has_user:
        _first_profile()
        return
    st.info("👈 Upload a GPX/FIT file in the sidebar or pick a ride from **Strava** or **Wahoo**.")


DEFAULT_FTP, DEFAULT_LTHR = 200, 170

FTP_HELP = ("**FTP (functional threshold power)** is the highest power, in watts, you could hold for about an "
            "hour. It sets your power zones and the TSS / IF of every ride. Quick estimate: 95 % of your best "
            "20-minute power. If unsure, leave it empty: the app starts from "
            f"{DEFAULT_FTP} W and then estimates it from your rides (best 20 minutes) as soon as it has some.")
LTHR_HELP = ("**LTHR (lactate threshold heart rate)** is the average heart rate, in bpm, you can sustain for "
             "about an hour of hard effort. It sets your heart-rate zones and the HR-based load for rides "
             "without a power meter. Quick estimate: your average HR over the last 20 minutes of a 30-minute "
             f"all-out effort. If unsure, leave it empty: the app starts from {DEFAULT_LTHR} bpm and then estimates it "
             "from the best 20-minute heart rate in your rides.")


DEFAULT_W_PRIME_KJ = 20

W_PRIME_HELP = ("**W′ (W prime)** is the amount of work, in kilojoules, you can do above your threshold power before "
                "you are exhausted: your anaerobic battery. It drives the W′ balance chart of every ride. Typical "
                f"values are 15 to 30 kJ. If unsure, leave it empty: the app starts from {DEFAULT_W_PRIME_KJ} kJ and, "
                "when a ride uses more than that, offers to raise it.")


def w_prime_from_kj(kj):
    """Joules from the kilojoule field of a form; None when it was left empty."""
    return float(kj) * 1000.0 if kj else None


def _first_profile():
    """The one thing to do before anything else: create the rider's profile (zones, TSS and W/kg use it)."""
    from src.data.db import save_user
    st.subheader("Welcome — create your profile")
    st.caption("Only the name is required. You can fill in or change the rest later in the sidebar.")
    with st.expander("What are FTP, LTHR and W′?", expanded=True):
        st.markdown(FTP_HELP)
        st.markdown(LTHR_HELP)
        st.markdown(W_PRIME_HELP)
    with st.form("first_profile_form"):
        name = st.text_input("Name", key="first_profile_name")
        c1, c2, c3, c4, c5 = st.columns(5)
        ftp = c1.number_input("FTP (Watts) · optional", value=None, min_value=1, step=5, key="first_profile_ftp",
                              placeholder="estimated from your rides", help=FTP_HELP)
        lthr = c2.number_input("LTHR (bpm) · optional", value=None, min_value=1, step=1, key="first_profile_lthr",
                               placeholder="estimated from your rides", help=LTHR_HELP)
        weight = c3.number_input("Weight (kg)", value=70.0)
        height = c4.number_input("Height (cm)", value=175)
        wprime = c5.number_input("W′ (kJ) · optional", value=None, min_value=1.0, step=0.5, key="first_profile_wprime",
                                 placeholder=f"{DEFAULT_W_PRIME_KJ} kJ", help=W_PRIME_HELP)
        if st.form_submit_button("Create profile", type="primary"):
            if not name.strip():
                st.error("Enter a name first.")
            else:
                ok, msg = save_user(name.strip(), int(ftp) if ftp else None, int(lthr) if lthr else None, weight, height,
                                 w_prime_from_kj(wprime))
                if ok:
                    st.rerun()
                else:
                    st.error(msg)
