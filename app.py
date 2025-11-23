import streamlit as st
from src.data.db import init_db
from src.viz.dashboard import render_sidebar, process_and_display_analysis, render_history

# Initialize Database
init_db()

# Page Config
st.set_page_config(page_title="BikeTools Analyzer", layout="wide")
st.title("🚴 BikeTools Analysis v1.4")

# Initialize Session State
if 'history_file' not in st.session_state:
    st.session_state['history_file'] = None

# --- 1. RENDER SIDEBAR ---
# The sidebar handles inputs, file uploads, and navigation selection.
app_mode, selected_user, current_file, settings = render_sidebar()

# --- 2. MAIN PAGE LOGIC ---
if app_mode == "Analyze Upload":
    if current_file is not None:
        process_and_display_analysis(current_file, selected_user, settings)
    else:
        st.info("👈 Please upload a GPX file in the Sidebar, or select a ride from **Ride History**.")

elif app_mode == "Ride History":
    render_history(selected_user)
