"""
The Streamlit side of the language: pick it once per browser session, apply it at the start of every run, and let
the rider change it.

Order: what the rider saved in their profile > the browser's language > this computer's language > English.
The picker only shows when at least one language besides English is released (see catalog.py).
"""
import streamlit as st

from . import DEFAULT, available_languages, language_name, set_language, t
from .detect import detect_language

KEY = "lang"                  # the language of this browser session
SOURCE_KEY = "lang_source"    # where it came from: 'profile', 'chosen', 'browser', 'system', 'default'
APPLIED_USER_KEY = "lang_user"


def browser_language_header():
    """The Accept-Language header of this session's browser (Streamlit 1.37+); None if it cannot be read."""
    try:
        return st.context.headers.get("Accept-Language")
    except Exception:
        return None


def init_session():
    """Call first thing in every run: detects the language once, then applies the session's language."""
    if KEY not in st.session_state:
        code, source = detect_language(available_languages(), browser_language_header())
        st.session_state[KEY], st.session_state[SOURCE_KEY] = code, source
    if st.session_state[KEY] not in available_languages():     # a language that is no longer offered
        st.session_state[KEY], st.session_state[SOURCE_KEY] = DEFAULT, "default"
    return set_language(st.session_state[KEY])


def apply_user_language(user_name, saved):
    """When a rider is selected, their saved language wins, once per rider. Returns True when the run must repeat
    (the sidebar was drawn in the old language)."""
    if not user_name or st.session_state.get(APPLIED_USER_KEY) == user_name:
        return False
    st.session_state[APPLIED_USER_KEY] = user_name
    if saved and saved in available_languages() and saved != st.session_state.get(KEY):
        st.session_state[KEY], st.session_state[SOURCE_KEY] = saved, "profile"
        return True
    return False


def picker(user_name=None):
    """The language selector (sidebar). Saves the choice in the rider's profile when one is selected. Draws nothing
    while English is the only language on offer."""
    languages = available_languages()
    if len(languages) < 2:
        return
    current = st.session_state.get(KEY, DEFAULT)

    def changed():
        code = st.session_state["lang_select"]
        st.session_state[KEY], st.session_state[SOURCE_KEY] = code, "chosen"
        if user_name:
            from src.data.db import set_user_language
            set_user_language(user_name, code)

    st.sidebar.selectbox(t("🌐 Language"), languages, index=languages.index(current) if current in languages else 0,
                         format_func=language_name, key="lang_select", on_change=changed)
