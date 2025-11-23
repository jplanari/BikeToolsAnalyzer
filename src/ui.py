import streamlit as st
import pandas as pd
import io
import os
import hashlib
from streamlit_folium import st_folium

# Imports from your existing modules
from src.utilities import parse_gpx, compute_distance_and_ascent, resample_to_seconds
from src.graphical import plot_elevation, plot_power_time, plot_power_curve, plot_zone_distribution, plot_climbs, plot_detailed_climb
from src.power import NP, power_curve, time_in_zones, coggan_zones
from src.hr import estimate_hr_threshold, time_in_hr_zones
from src.db import save_user, get_all_users, get_user_data, delete_user, save_ride, get_user_rides
from src.climbs import detect_climbs, get_climb_segments
from src.map_view import create_route_map

def render_sidebar():
    """Renders the sidebar and returns all user configuration settings."""
    st.sidebar.header("Navigation")
    
    app_mode = st.sidebar.radio("Go to", ["Analyze Upload", "Ride History"])

    st.sidebar.markdown("---")
    st.sidebar.header("👤 User Profile")

    # User Management
    tab_select, tab_create = st.sidebar.tabs(["Select User", "Create/Edit"])
    selected_user_data = None
    selected_user_name = None

    with tab_select:
        users = get_all_users()
        if users:
            # Simple selectbox
            selected_user_name = st.selectbox("Active User", users)
            if selected_user_name:
                selected_user_data = get_user_data(selected_user_name)
                if st.button("Delete User", type="primary"):
                    delete_user(selected_user_name)
                    st.rerun()
        else:
            st.info("No users found. Create one!")

    with tab_create:
        with st.form("user_form"):
            new_name = st.text_input("Name")
            new_ftp = st.number_input("FTP", value=200)
            new_lthr = st.number_input("LTHR", value=170)
            new_weight = st.number_input("Weight (kg)", value=70.0)
            submitted = st.form_submit_button("Save Profile")
            
            if submitted and new_name:
                success, msg = save_user(new_name, new_ftp, new_lthr, new_weight)
                if success:
                    st.success(f"Saved {new_name}!")
                    st.rerun()
                else:
                    st.error(f"Error: {msg}")

    st.sidebar.markdown("---")

    # --- SIMPLIFIED FILE UPLOAD ---
    # We allow re-uploading the same file by not using any state keys that block it.
    uploaded_file = st.sidebar.file_uploader("Upload GPX File", type=["gpx"])

    # Analysis Settings
    st.sidebar.header("📊 Analysis Settings")
    default_ftp = selected_user_data['ftp'] if selected_user_data else 250
    default_lthr = selected_user_data['lthr'] if selected_user_data else 0

    settings = {
        "ftp": st.sidebar.number_input("FTP (Watts)", min_value=0, value=default_ftp, step=5),
        "lthr": st.sidebar.number_input("Threshold HR (bpm)", min_value=0, value=default_lthr, step=1),
        "show_map": st.sidebar.checkbox("Route Map", value=True),
        "show_ele": st.sidebar.checkbox("Elevation Profile", value=True),
        "show_power": st.sidebar.checkbox("Power vs Time", value=True),
        "show_curve": st.sidebar.checkbox("Power Curve", value=True),
        "show_zones": st.sidebar.checkbox("Zone Distributions", value=True),
        "show_climbs": st.sidebar.checkbox("Climb Analysis", value=False)
    }

    return app_mode, selected_user_name, uploaded_file, settings

def process_and_display_analysis(file_obj, user_name, settings):
    """Main logic to parse GPX, calculate stats, and render plots."""
    
    # We REMOVED the check that prevents processing if hash exists.
    # Analysis happens every time.

    file_bytes = file_obj.getvalue()
    
    # We still calculate hash for DB saving logic later
    file_hash = hashlib.md5(file_bytes).hexdigest()

    with open("temp.gpx", "wb") as f:
        f.write(file_bytes)

    with st.spinner(f"Processing {file_obj.name}..."):
        try:
            df = parse_gpx("temp.gpx")
            df, total_dist, total_ascent = compute_distance_and_ascent(df)
            df = resample_to_seconds(df)
            df, total_dist, total_ascent = compute_distance_and_ascent(df)
        except Exception as e:
            st.error(f"Error parsing GPX: {e}")
            if os.path.exists("temp.gpx"): os.remove("temp.gpx")
            return

        # Basic Stats
        duration_s = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds()
        avg_speed = (total_dist / duration_s) * 3.6 if duration_s > 0 else 0
        norm_power = NP(df['power']) if 'power' in df.columns else 0

        # Auto-Save Logic
        # We try to save. If it's a duplicate, the DB function returns False,
        # but that's fine - we just catch it and don't show an error.
        if user_name:
             _save_ride_to_db(user_name, file_obj.name, file_bytes, df, total_dist, total_ascent, avg_speed, norm_power)

    # Detect Climbs
    detected_climbs = []
    if settings['show_climbs'] or settings['show_map']:
         detected_climbs = detect_climbs(df, min_length=1000, join_dist=1000)

    # Dashboard
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Distance", f"{total_dist/1000:.2f} km")
    c2.metric("Elevation", f"{int(total_ascent)} m")
    c3.metric("Avg Speed", f"{avg_speed:.1f} km/h")
    c4.metric("Norm Power", f"{int(norm_power)} W" if norm_power else "N/A")

    st.markdown("---")

    # Visualizations
    _render_plots(df, settings, detected_climbs)
    
    if os.path.exists("temp.gpx"): os.remove("temp.gpx")

def _save_ride_to_db(user_name, filename, file_bytes, df, dist, ele, speed, np_val):
    """Helper to handle database saving logic."""
    def safe_mean(series):
        if series.empty: return 0
        m = series.mean()
        return int(m) if pd.notna(m) else 0

    stats = {
        'date_time': df['time'].iloc[0],
        'dist_km': dist / 1000,
        'ele_m': ele,
        'speed': speed,
        'np': int(np_val) if pd.notna(np_val) else 0,
        'avg_p': safe_mean(df['power']) if 'power' in df.columns else 0,
        'avg_hr': safe_mean(df['hr']) if 'hr' in df.columns else 0
    }
    
    # We use a session state set to avoid showing the Toast notification 
    # every single time you refresh the page, but we allow the save attempt.
    if 'saved_hashes' not in st.session_state:
        st.session_state['saved_hashes'] = set()
        
    file_hash = hashlib.md5(file_bytes).hexdigest()
    
    # Only try to write to DB if we haven't successfully done so in this session
    if file_hash not in st.session_state['saved_hashes']:
        success, msg = save_ride(user_name, filename, file_bytes, stats)
        
        if success:
            st.toast(f"✅ Saved to History: {filename}")
            st.session_state['saved_hashes'].add(file_hash)
        elif "duplicate" in msg.lower():
            # It's in the DB already. That's fine! 
            # We just don't toast, and we mark it as 'handled' for this session.
            st.session_state['saved_hashes'].add(file_hash)
        else:
            st.error(f"Save Error: {msg}")

def _render_plots(df, settings, detected_climbs):
    def show_centered(fig):
        c1, c2, c3 = st.columns([1, 3, 1])
        with c2: st.pyplot(fig)

    if settings['show_map']:
        st.subheader("Route Map")
        map_obj = create_route_map(df, detected_climbs)
        if map_obj: st_folium(map_obj, width=1000, height=500)
        else: st.warning("No GPS coordinates found.")

    if settings['show_ele']:
        st.subheader("Elevation Profile")
        show_centered(plot_elevation(df))

    if settings['show_power'] and 'power' in df.columns:
        st.subheader("Power vs Time")
        show_centered(plot_power_time(df))

    if settings['show_curve'] and 'power' in df.columns:
        st.subheader("Power Curve")
        show_centered(plot_power_curve(power_curve(df['power'])))

    if settings['show_zones']:
        st.subheader("Zone Distribution")
        c1, c2 = st.columns(2)
        if 'power' in df.columns and settings['ftp']:
            zones = coggan_zones(settings['ftp'])
            times = time_in_zones(df['power'], zones)
            fig = plot_zone_distribution(times, "Power Zones")
            if fig: c1.pyplot(fig)
        
        active_lthr = settings['lthr']
        if not active_lthr and 'hr' in df.columns:
             est, _ = estimate_hr_threshold(df['hr'])
             active_lthr = est
        
        if 'hr' in df.columns and active_lthr:
            times = time_in_hr_zones(df['hr'], active_lthr)
            fig = plot_zone_distribution(times, "Heart Rate Zones")
            if fig: c2.pyplot(fig)

    if settings['show_climbs']:
        _render_climb_details(df, detected_climbs)

def _render_climb_details(df, climbs):
    st.subheader("Climb Analysis")
    if not climbs:
        st.info("No significant climbs detected.")
        return

    fig = plot_climbs(df, climbs)
    c1, c2, c3 = st.columns([1, 3, 1])
    with c2: st.pyplot(fig)

    data = []
    for i, c in enumerate(climbs):
        data.append({
            "ID": i,
            "Climb": f"#{i+1}",
            "Len (km)": round(c['length_m']/1000, 2),
            "Gain (m)": int(c['elev_gain_m']),
            "Grad (%)": round(c['avg_gradient_pct'], 1),
            "VAM": int(c['vam_mph'])
        })
    
    df_display = pd.DataFrame(data)
    st.info("👇 Select a climb to see detailed segments.")
    
    sel = st.dataframe(
        df_display, 
        column_config={"ID": None}, 
        hide_index=True, 
        on_select="rerun", 
        selection_mode="single-row"
    )

    if sel.selection.rows:
        idx = sel.selection.rows[0]
        c = climbs[idx]
        st.markdown(f"**Detailed Profile: Climb #{idx+1}**")
        segs = get_climb_segments(df, c['start_idx'], c['end_idx'])
        fig_det = plot_detailed_climb(df, c['start_idx'], c['end_idx'], segs)
        st.pyplot(fig_det)

def render_history(user_name):
    """Renders the history table purely for viewing stats."""
    st.header("📜 Ride History")
    
    if not user_name:
        st.warning("Please select a user in the sidebar to view their history.")
        return

    history = get_user_rides(user_name)
    if not history:
        st.info("No rides saved yet.")
        return

    hist_df = pd.DataFrame(history)
    st.info("Here is a log of your past rides. (Loading historical charts is disabled).")

    st.dataframe(
        hist_df[['date_time', 'filename', 'distance_km', 'elevation_m', 'norm_power']],
        hide_index=True,
    )
