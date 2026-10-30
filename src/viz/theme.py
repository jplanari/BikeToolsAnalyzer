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


def _first_profile():
    """The one thing to do before anything else: create the rider's profile (zones, TSS and W/kg use it)."""
    from src.data.db import save_user
    st.subheader("Welcome — create your profile")
    st.caption("Zones, TSS and W/kg are computed from these values. You can edit them later in the sidebar.")
    with st.form("first_profile_form"):
        name = st.text_input("Name", key="first_profile_name")
        c1, c2, c3, c4 = st.columns(4)
        ftp = c1.number_input("FTP (Watts)", value=200)
        lthr = c2.number_input("LTHR (bpm)", value=170)
        weight = c3.number_input("Weight (kg)", value=70.0)
        height = c4.number_input("Height (cm)", value=175)
        if st.form_submit_button("Create profile", type="primary"):
            if not name.strip():
                st.error("Enter a name first.")
            else:
                ok, msg = save_user(name.strip(), ftp, lthr, weight, height)
                if ok:
                    st.rerun()
                else:
                    st.error(msg)
