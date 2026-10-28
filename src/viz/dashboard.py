"""
Page entry points used by app.py. The pages live in their own modules:
sidebar.py, ride_analysis.py, user_corner.py (and training_page.py); formatting.py has shared helpers.
"""
from src.viz.sidebar import render_sidebar, handle_strava_callback
from src.viz.ride_analysis import process_and_display_analysis
from src.viz.user_corner import render_user_corner, render_history

__all__ = ["render_sidebar", "handle_strava_callback", "process_and_display_analysis",
           "render_user_corner", "render_history"]
