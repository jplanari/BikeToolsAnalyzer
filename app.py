import streamlit as st
import pandas as pd
import os
from streamlit_folium import st_folium 

# --- IMPORTS ---
from src.utilities import parse_gpx, compute_distance_and_ascent, resample_to_seconds
from src.graphical import plot_elevation, plot_power_time, plot_power_curve, plot_zone_distribution, plot_climbs, plot_detailed_climb
from src.power import NP, power_curve, time_in_zones, coggan_zones
from src.hr import estimate_hr_threshold, time_in_hr_zones
from src.db import init_db, save_user, get_all_users, get_user_data, delete_user
from src.climbs import detect_climbs, get_climb_segments
from src.map_view import create_route_map

# Initialize DB
init_db()

st.set_page_config(page_title="BikeTools Analyzer", layout="wide")
st.title("🚴 BikeTools Analysis v1.1")

# --- SIDEBAR: User Profile Management ---
st.sidebar.header("👤 User Profile")
tab_select, tab_create = st.sidebar.tabs(["Select User", "Create/Edit"])

selected_user_data = None

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

# --- SIDEBAR: Ride Settings ---
st.sidebar.header("📊 Ride Settings")
uploaded_file = st.sidebar.file_uploader("Upload GPX File", type=["gpx"])

default_ftp = selected_user_data['ftp'] if selected_user_data else 250
default_lthr = selected_user_data['lthr'] if selected_user_data else 0

ftp_input = st.sidebar.number_input("FTP (Watts)", min_value=0, value=default_ftp, step=5)
lthr_input = st.sidebar.number_input("Threshold Heart Rate (bpm)", min_value=0, value=default_lthr, step=1)
user_override_lthr = lthr_input if lthr_input > 0 else None

st.sidebar.subheader("Plots to Show")
show_map = st.sidebar.checkbox("Route Map", value=True)
show_ele = st.sidebar.checkbox("Elevation Profile", value=True)
show_power = st.sidebar.checkbox("Power vs Time", value=True)
show_curve = st.sidebar.checkbox("Power Curve", value=True)
show_zones = st.sidebar.checkbox("Zone Distributions", value=True)
show_climbs = st.sidebar.checkbox("Climb Analysis", value=False)

def show_centered_plot(fig):
    col_left, col_center, col_right = st.columns([1, 3, 1])
    with col_center:
        st.pyplot(fig)

# --- Main App Logic ---
if uploaded_file is not None:
    with open("temp.gpx", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing GPX..."):
        df = parse_gpx("temp.gpx")
        df, total_dist, total_ascent = compute_distance_and_ascent(df)
        df = resample_to_seconds(df)
        df, total_dist, total_ascent = compute_distance_and_ascent(df)
        
        duration_s = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds()
        avg_speed = (total_dist / duration_s) * 3.6 if duration_s > 0 else 0
        norm_power = NP(df['power']) if 'power' in df.columns else 0

    # Detect Climbs (Used for Map and Analysis)
    detected_climbs_list = []
    if show_climbs or show_map:
         detected_climbs_list = detect_climbs(df, min_length=1000, join_dist=500)

    # Display Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Distance", f"{total_dist/1000:.2f} km")
    col2.metric("Elevation", f"{int(total_ascent)} m")
    col3.metric("Avg Speed", f"{avg_speed:.1f} km/h")
    col4.metric("Normalized Power", f"{int(norm_power)} W" if norm_power else "N/A")

    # LTHR Logic
    active_lthr = None
    if 'hr' in df.columns:
        est_lthr, peak_20m_hr = estimate_hr_threshold(df['hr'])
        status_box = st.container()

        if user_override_lthr:
            active_lthr = user_override_lthr
            status_box.success(f"✅ Using **Custom/Profile** LTHR: **{int(active_lthr)} bpm**")
            if est_lthr:
                status_box.caption(f"(Note: File estimate was {int(est_lthr)} bpm)")     
        elif est_lthr:
            active_lthr = est_lthr
            status_box.info(f"💡 Using **Estimated** LTHR from file: **{int(active_lthr)} bpm**")
            status_box.caption(f"Derived from Peak 20m HR ({int(peak_20m_hr)} bpm).")
        else:
            st.warning("Could not estimate LTHR. Please enter a value manually.")

    st.markdown("---")

    # --- MAP SECTION ---
    if show_map:
        st.subheader("Route Map")
        map_obj = create_route_map(df, detected_climbs_list)
        if map_obj:
            st_folium(map_obj, width=1000, height=500)
        else:
            st.warning("No GPS coordinates found in file.")

    # --- STANDARD PLOTS ---
    if show_ele:
        st.subheader("Elevation Profile")
        fig = plot_elevation(df, outpath=None)
        show_centered_plot(fig)

    if show_power and 'power' in df.columns:
        st.subheader("Power & Elevation")
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
        
        # Power Zones
        if 'power' in df.columns and ftp_input:
            p_zones = coggan_zones(ftp_input)
            p_zone_times = time_in_zones(df['power'], p_zones)
            fig_p = plot_zone_distribution(p_zone_times, "Power Zones",  None)
            if fig_p:
                col1.pyplot(fig_p)
            else:
                col1.info("Insufficient Power data.")
            
        # HR Zones
        if 'hr' in df.columns and active_lthr:
            h_zone_times = time_in_hr_zones(df['hr'], active_lthr)
            fig_h = plot_zone_distribution(h_zone_times, "Heart Rate Zones", None)
            if fig_h:
                col2.pyplot(fig_h)
            else:
                col2.info("Insufficient Heart Rate data.")

    # --- INTERACTIVE CLIMB ANALYSIS ---
    if show_climbs:
        st.subheader("Detected Climbs Analysis")
        if detected_climbs_list:
            # 1. Overview Plot
            fig_climbs = plot_climbs(df, detected_climbs_list, outpath=None)
            show_centered_plot(fig_climbs)

            # 2. Prepare Data for Interactive Table
            climb_display_data = []
            for i, c in enumerate(detected_climbs_list):
                climb_display_data.append({
                    "ID": i,
                    "Climb #": f"#{i + 1}",
                    "Length (km)": round(c['length_m']/1000, 2),
                    "Elev Gain (m)": int(c['elev_gain_m']),
                    "Avg Grad (%)": round(c['avg_gradient_pct'], 1),
                    "Avg Speed (km/h)": round(c['avg_speed_kph'], 1),
                    "VAM (m/h)": int(c['vam_mph'])
                })
            
            df_display = pd.DataFrame(climb_display_data)
            
            st.info("👇 Click on a row below to see the detailed gradient profile for that climb.")
            
            # 3. Interactive Table
            selection = st.dataframe(
                df_display,
                column_config={"ID": None}, # Hide ID
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row"
            )

            # 4. Handle Selection
            if selection.selection.rows:
                selected_row_idx = selection.selection.rows[0]
                selected_climb = detected_climbs_list[selected_row_idx]
                
                st.markdown(f"### Detailed Profile: Climb #{selected_row_idx + 1}")
                
                segments = get_climb_segments(df, selected_climb['start_idx'], selected_climb['end_idx'])
                fig_detailed = plot_detailed_climb(df, selected_climb['start_idx'], selected_climb['end_idx'], segments)
                show_centered_plot(fig_detailed)

        else:
            st.info("No significant climbs detected.")

    os.remove("temp.gpx")
else:
    st.info("Please upload a GPX file to begin.")
