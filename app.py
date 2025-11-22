import streamlit as st
import pandas as pd
import os
import io
from streamlit_folium import st_folium

# --- IMPORTS ---
from src.utilities import parse_gpx, compute_distance_and_ascent, resample_to_seconds
from src.graphical import plot_elevation, plot_power_time, plot_power_curve, plot_zone_distribution, plot_climbs, plot_detailed_climb
from src.power import NP, power_curve, time_in_zones, coggan_zones
from src.hr import estimate_hr_threshold, time_in_hr_zones
from src.db import init_db, save_user, get_all_users, get_user_data, delete_user, save_ride, get_user_rides
from src.climbs import detect_climbs, get_climb_segments
from src.map_view import create_route_map

# Initialize DB
init_db()

st.set_page_config(page_title="BikeTools Analyzer", layout="wide")
st.title("🚴 BikeTools Analysis v1.3")

# --- STATE MANAGEMENT ---
if 'history_file' not in st.session_state:
    st.session_state['history_file'] = None

# --- SIDEBAR ---
st.sidebar.header("Navigation")
# We use a radio button instead of tabs so we can programmatically switch views
app_mode = st.sidebar.radio(
    "Go to", 
    ["Analyze Upload", "Ride History"], 
    key="navigation_radio"
)

st.sidebar.markdown("---")
st.sidebar.header("👤 User Profile")

# User Selection Logic
tab_select, tab_create = st.sidebar.tabs(["Select User", "Create/Edit"])
selected_user_data = None
selected_user_name = None

with tab_select:
    users = get_all_users()
    if users:
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

# --- FILE HANDLING LOGIC ---
# We resolve which file to use BEFORE rendering the pages.
uploaded_widget = st.sidebar.file_uploader("Upload GPX File", type=["gpx"])
current_file = None

# 1. Priority: Widget (New Upload)
if uploaded_widget is not None:
    # If user uploads a new file, clear any history file to avoid confusion
    st.session_state['history_file'] = None
    current_file = uploaded_widget

# 2. Fallback: History File
elif st.session_state['history_file'] is not None:
    current_file = st.session_state['history_file']
    st.sidebar.success(f"📂 **Loaded:** {current_file.name}")
    if st.sidebar.button("❌ Clear File"):
        st.session_state['history_file'] = None
        st.rerun()

# --- RIDE SETTINGS ---
st.sidebar.header("📊 Analysis Settings")
default_ftp = selected_user_data['ftp'] if selected_user_data else 250
default_lthr = selected_user_data['lthr'] if selected_user_data else 0

ftp_input = st.sidebar.number_input("FTP (Watts)", min_value=0, value=default_ftp, step=5)
lthr_input = st.sidebar.number_input("Threshold HR (bpm)", min_value=0, value=default_lthr, step=1)
user_override_lthr = lthr_input if lthr_input > 0 else None

show_map = st.sidebar.checkbox("Route Map", value=True)
show_ele = st.sidebar.checkbox("Elevation Profile", value=True)
show_power = st.sidebar.checkbox("Power vs Time", value=True)
show_curve = st.sidebar.checkbox("Power Curve", value=True)
show_zones = st.sidebar.checkbox("Zone Distributions", value=True)
show_climbs = st.sidebar.checkbox("Climb Analysis", value=False)

# --- HELPER FUNCTION ---
def show_centered_plot(fig):
    col_left, col_center, col_right = st.columns([1, 3, 1])
    with col_center:
        st.pyplot(fig)

# =========================================================
# PAGE 1: ANALYZE UPLOAD
# =========================================================
if app_mode == "Analyze Upload":
    if current_file is not None:
        # READ BYTES ONCE
        file_bytes = current_file.getvalue()

        # Write temp file
        with open("temp.gpx", "wb") as f:
            f.write(file_bytes)

        with st.spinner(f"Processing {current_file.name}..."):
            df = parse_gpx("temp.gpx")
            df, total_dist, total_ascent = compute_distance_and_ascent(df)
            df = resample_to_seconds(df)
            df, total_dist, total_ascent = compute_distance_and_ascent(df)
            
            duration_s = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds()
            avg_speed = (total_dist / duration_s) * 3.6 if duration_s > 0 else 0
            norm_power = NP(df['power']) if 'power' in df.columns else 0

            # --- AUTO-SAVE LOGIC ---
            # Only save if it's a NEW upload (from widget), not a history reload
            is_history_view = (st.session_state['history_file'] is not None)
            
            if selected_user_name and not is_history_view:
                def safe_mean(series):
                    if series.empty: return 0
                    m = series.mean()
                    return int(m) if pd.notna(m) else 0

                ride_stats = {
                    'date_time': df['time'].iloc[0],
                    'dist_km': total_dist / 1000,
                    'ele_m': total_ascent,
                    'speed': avg_speed,
                    'np': int(norm_power) if pd.notna(norm_power) else 0,
                    'avg_p': safe_mean(df['power']) if 'power' in df.columns else 0,
                    'avg_hr': safe_mean(df['hr']) if 'hr' in df.columns else 0
                }
                
                success, msg = save_ride(selected_user_name, current_file.name, file_bytes, ride_stats)
                if success:
                    st.toast(f"✅ {msg}")
                elif "duplicate" in msg:
                    pass # Don't spam user on refresh
                else:
                    st.error(f"Save Error: {msg}")

        # Detect Climbs
        detected_climbs_list = []
        if show_climbs or show_map:
             detected_climbs_list = detect_climbs(df, min_length=1000, join_dist=500)

        # --- DASHBOARD ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Distance", f"{total_dist/1000:.2f} km")
        col2.metric("Elevation", f"{int(total_ascent)} m")
        col3.metric("Avg Speed", f"{avg_speed:.1f} km/h")
        col4.metric("Norm Power", f"{int(norm_power)} W" if norm_power else "N/A")

        # LTHR Info
        active_lthr = None
        if 'hr' in df.columns:
            est_lthr, peak_20m_hr = estimate_hr_threshold(df['hr'])
            if user_override_lthr:
                active_lthr = user_override_lthr
            elif est_lthr:
                active_lthr = est_lthr

        st.markdown("---")

        # --- VISUALIZATIONS ---
        if show_map:
            st.subheader("Route Map")
            map_obj = create_route_map(df, detected_climbs_list)
            if map_obj:
                st_folium(map_obj, width=1000, height=500)
            else:
                st.warning("No GPS coordinates found.")

        if show_ele:
            st.subheader("Elevation Profile")
            fig = plot_elevation(df, outpath=None)
            show_centered_plot(fig)

        if show_power and 'power' in df.columns:
            st.subheader("Power vs Time")
            fig = plot_power_time(df, outpath=None)
            show_centered_plot(fig)

        if show_curve and 'power' in df.columns:
            st.subheader("Power Curve")
            pcurve = power_curve(df['power'])
            fig = plot_power_curve(pcurve, outpath=None)
            show_centered_plot(fig)

        if show_zones:
            st.subheader("Zone Distribution")
            col1, col2 = st.columns(2)
            if 'power' in df.columns and ftp_input:
                p_zones = coggan_zones(ftp_input)
                p_zone_times = time_in_zones(df['power'], p_zones)
                fig_p = plot_zone_distribution(p_zone_times, "Power Zones",  None)
                if fig_p: col1.pyplot(fig_p)
            if 'hr' in df.columns and active_lthr:
                h_zone_times = time_in_hr_zones(df['hr'], active_lthr)
                fig_h = plot_zone_distribution(h_zone_times, "Heart Rate Zones", None)
                if fig_h: col2.pyplot(fig_h)

        if show_climbs:
            st.subheader("Climb Analysis")
            if detected_climbs_list:
                fig_climbs = plot_climbs(df, detected_climbs_list, outpath=None)
                show_centered_plot(fig_climbs)

                # Interactive Table
                climb_display_data = []
                for i, c in enumerate(detected_climbs_list):
                    climb_display_data.append({
                        "ID": i,
                        "Climb": f"#{i + 1}",
                        "Len (km)": round(c['length_m']/1000, 2),
                        "Gain (m)": int(c['elev_gain_m']),
                        "Grad (%)": round(c['avg_gradient_pct'], 1),
                        "VAM": int(c['vam_mph'])
                    })
                
                df_display = pd.DataFrame(climb_display_data)
                st.info("👇 Select a climb to see detailed segments.")
                
                selection = st.dataframe(
                    df_display,
                    column_config={"ID": None},
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row"
                )

                if selection.selection.rows:
                    row_idx = selection.selection.rows[0]
                    sel_climb = detected_climbs_list[row_idx]
                    
                    st.markdown(f"**Detailed Profile: Climb #{row_idx+1}**")
                    segments = get_climb_segments(df, sel_climb['start_idx'], sel_climb['end_idx'])
                    fig_det = plot_detailed_climb(df, sel_climb['start_idx'], sel_climb['end_idx'], segments)
                    st.pyplot(fig_det)
            else:
                st.info("No significant climbs detected.")
        
        os.remove("temp.gpx")

    else:
        # No file loaded
        st.info("👈 Please upload a GPX file in the Sidebar, or select a ride from **Ride History**.")

# =========================================================
# PAGE 2: RIDE HISTORY
# =========================================================
elif app_mode == "Ride History":
    st.header("📜 Ride History")
    
    if selected_user_name:
        history = get_user_rides(selected_user_name)
        
        if history:
            hist_df = pd.DataFrame(history)
            st.info("👇 Click on a row to load the ride.")

            # Interactive History Table
            event = st.dataframe(
                hist_df[['id', 'date_time', 'filename', 'distance_km', 'elevation_m']],
                selection_mode="single-row",
                on_select="rerun",
                hide_index=True,
                use_container_width=True
            )
            
            if event.selection.rows:
                idx = event.selection.rows[0]
                ride_id = hist_df.iloc[idx]['id']
                
                from src.db import get_ride_file
                fname, fbytes = get_ride_file(ride_id)
                
                if fbytes:
                    # Create Virtual File
                    virtual_file = io.BytesIO(fbytes)
                    virtual_file.name = fname
                    
                    # 1. Set the file in session state
                    st.session_state['history_file'] = virtual_file
                    
                    # 2. FORCE THE VIEW TO SWITCH
                    st.session_state['navigation_radio'] = "Analyze Upload"
                    
                    st.toast(f"Loading {fname}...")
                    st.rerun()
        else:
            st.warning("No rides found for this user.")
    else:
        st.warning("Please select a user in the sidebar to view their history.")
