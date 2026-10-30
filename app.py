import streamlit as st
from src.data.db import init_db
from src.viz.dashboard import (render_sidebar, process_and_display_analysis, render_history,
                               render_user_corner, handle_strava_callback)
from src.viz.training_page import render_training_page
from src.viz import theme
from src.viz.gallery import render_gallery

# Initialize Database
init_db()

# Page Config
st.set_page_config(page_title="BikeTools Analyzer", page_icon="🚴", layout="wide")
theme.apply_look()
theme.page_header("v1.4")

# Initialize Session State
if 'history_file' not in st.session_state:
    st.session_state['history_file'] = None

# Finish a Strava OAuth round trip if we were just redirected back
handle_strava_callback()

# --- 1. RENDER SIDEBAR ---
# The sidebar handles inputs, file uploads, and navigation selection.
app_mode, selected_user, current_file, settings = render_sidebar()

# --- 2. MAIN PAGE LOGIC ---
if app_mode == "Analyze Upload":
    opened = current_file if current_file is not None else st.session_state.get("gallery_file")
    if opened is not None:
        if current_file is None and st.button("← All rides", key="back_to_gallery"):
            st.session_state["gallery_file"] = None
            st.rerun()
        else:
            process_and_display_analysis(opened, selected_user, settings)
    else:
        theme.welcome(has_user=bool(selected_user))
        if selected_user:
            render_gallery(selected_user)

elif app_mode == "Training":
    render_training_page(selected_user)

elif app_mode == "User Corner":
    render_user_corner(selected_user)
    render_history(selected_user)
