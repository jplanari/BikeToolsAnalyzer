"""
Tab-like sections that only render the selected one.

st.tabs() runs every tab's code on every interaction and ships every tab's charts to the
browser. A horizontal radio plus an `if` renders just the visible section. Streamlit forgets
the value of a widget that isn't drawn in a run, so widgets inside sections go through
remembered() to keep their choice while another section is showing.
"""
import streamlit as st


def section_picker(options, key):
    """Horizontal selector that looks like a tab bar. Returns the chosen option."""
    return st.radio("Section", options, horizontal=True, key=key, label_visibility="collapsed")


def remembered(widget, key, default, *args, **kwargs):
    """
    Create `widget(*args, key=key, **kwargs)` so its value survives the widget not being drawn
    for a while (e.g. another section is selected). `default` is the starting value; do not pass
    index=/value= to the widget, the value comes from session state.
    """
    saved = f"_saved_{key}"
    if key not in st.session_state:
        st.session_state[key] = st.session_state.get(saved, default)
    value = widget(*args, key=key, **kwargs)
    st.session_state[saved] = value
    return value
