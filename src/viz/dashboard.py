import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import hashlib
from streamlit_folium import st_folium

# Imports from your existing modules
from src.data.gpx import parse_gpx, compute_distance_and_ascent, resample_to_seconds, calculate_bearing, compute_speed, compute_grade
from src.data.fit import parse_fit
from src.viz.plots import (
        plot_elevation, plot_x_time, plot_power_curve, plot_zone_distribution, 
        plot_climbs, plot_detailed_climb, plot_power_budget, plot_w_prime_balance,
        plot_pmc
    )
from src.analysis.power import NP, IF, TSS, power_curve, time_in_zones, coggan_zones, calculate_w_prime_balance, calculate_ride_kJ
from src.analysis.hr import estimate_hr_threshold, time_in_hr_zones
from src.data.db import (
        save_user, get_all_users, get_user_data, delete_user, 
        save_ride, get_user_rides, get_user_best_power, get_user_tss_history
    )
from src.analysis.climbs import detect_climbs, get_climb_segments
from src.viz.maps import create_route_map
from src.physics.aerodyn import calculate_CdA, get_avg_cda, calculate_power_components
from src.data.weather import fetch_ride_weather  # Ensure this is imported
from src.data.strava import fetch_gpx_from_strava  # Ensure this is imported

def show_centered(fig):
    c1, c2, c3 = st.columns([1, 3, 1])
    with c2: st.pyplot(fig)

def format_duration(seconds):
    if not seconds: return "00:00:00"

    m, s =divmod(int(seconds), 60)
    h, m =divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02}"

def render_sidebar():
    """Renders the sidebar and returns all user configuration settings."""
    st.sidebar.header("Navigation")
    
    app_mode = st.sidebar.radio("Go to", ["Analyze Upload", "User Corner"])

    st.sidebar.markdown("---")
    st.sidebar.header("👤 User Profile")

    # User Management
    tab_select, tab_create = st.sidebar.tabs(["Select User", "Create/Edit"])
    
    selected_user_name = None
    
    with tab_select:
        users = get_all_users()
        if users:
            selected_user_name = st.selectbox("Current User", users)
            if selected_user_name:
                u_data = get_user_data(selected_user_name)
                if u_data:
                    user_ftp = u_data['ftp']
                    user_lthr = u_data['lthr']
                    user_weight = u_data['weight']


            if st.button("Delete User", type="primary"):
                delete_user(selected_user_name)
                st.rerun()
        else:
            st.info("No users found. Create one!")

    with tab_create:
        with st.form("user_form"):
            new_name = st.text_input("Name")
            new_ftp = st.number_input("FTP (Watts)", value=200)
            new_lthr = st.number_input("LTHR (bpm)", value=170)
            new_weight = st.number_input("Weight (kg)", value=70.0)
            submitted = st.form_submit_button("Save User")
            if submitted and new_name:
                success, msg = save_user(new_name, new_ftp, new_lthr, new_weight)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    st.sidebar.markdown("---")
    st.sidebar.header("📂 Data Import")
    
    # 1. Initialize current_file to None by default
    current_file = None
    
    import_method = st.sidebar.radio("Source", ["Upload Ride", "Strava URL"])

    if import_method == "Upload Ride":
        uploaded_file = st.sidebar.file_uploader("Upload file ride", type=["gpx", "fit"])
        # If user uploaded a file, assign it to current_file
        if uploaded_file is not None:
            current_file = uploaded_file

    elif import_method == "Strava URL":
        st.sidebar.info("Requires your '_strava4_session' cookie from your browser.")
        
        strava_url = st.sidebar.text_input("Strava Activity URL")
        strava_cookie = st.sidebar.text_input("Strava Session Cookie", type="password")
        
        if st.sidebar.button("Fetch from Strava"):
            if strava_url and strava_cookie:
                with st.spinner("Downloading GPX from Strava..."):
                    file_obj, error = fetch_gpx_from_strava(strava_url, strava_cookie)
                    
                    if file_obj:
                        st.sidebar.success("Download successful!")
                        # Extract ID from URL for the filename
                        activity_id = strava_url.split('/activities/')[-1].split('/')[0]
                        file_obj.name = f"strava_{activity_id}.gpx"
                        
                        # Save to session state so it persists after rerun
                        st.session_state['temp_strava_file'] = file_obj
                        current_file = file_obj
                    else:
                        st.sidebar.error(error)
            else:
                st.sidebar.warning("URL and Cookie are required.")
        
        # Restore from session state if available (so the file stays loaded)
        if current_file is None and 'temp_strava_file' in st.session_state:
            current_file = st.session_state['temp_strava_file']

    # Settings
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Settings")
    settings = {
        'show_map': st.sidebar.checkbox("Show Map", value=True),
        'show_curve': st.sidebar.checkbox("Show Power Curve", value=True),
        'show_zones': st.sidebar.checkbox("Show Zones", value=True),
        'show_ele': st.sidebar.checkbox("Show Elevation Profile", value=True),
        'show_power': st.sidebar.checkbox("Show Power & Metrics", value=True),
        'show_climbs': st.sidebar.checkbox("Show Climb Analysis", value=True),
        'show_aero': st.sidebar.checkbox("Show Aerodynamic Analysis (Beta)", value=True),

        'ftp': user_ftp,
        'lthr': user_lthr,
        'weight': user_weight
    }

    # RETURN 'current_file' (which works for both logic paths)
    return app_mode, selected_user_name, current_file, settings

def process_and_display_analysis(file_obj, user_name, settings):
    """Main logic to parse GPX, calculate stats, and render plots."""
    
    file_bytes = file_obj.getvalue()
    filename = file_obj.name.lower()
    
    with open("temp.gpx", "wb") as f:
        f.write(file_bytes)

    with st.spinner(f"Processing {file_obj.name} and fetching weather..."):
        try:
            if filename.endswith(".fit"):
                df = parse_fit(file_obj)
                df, total_dist, total_ascent = compute_distance_and_ascent(df)
                df = resample_to_seconds(df)
                df, total_dist, total_ascent = compute_distance_and_ascent(df)
                df = compute_grade(df)
            # 2. Physics Prep
                df['bearing'] = calculate_bearing(df)

            # 3. AUTOMATIC WEATHER FETCH
            # We attempt to fetch it immediately. If it fails, df remains unchanged.
                df = fetch_ride_weather(df)

                moving_thresh = 2.0 #m/s
                is_moving = df['speed'] >= moving_thresh

                moving_time_sec = is_moving.sum()
                elapsed_time_sec = len(df) #Assuming 1Hz resampled

            else:
                # 1. Standard Parsing
                df = parse_gpx("temp.gpx")
                df, total_dist, total_ascent = compute_distance_and_ascent(df)
                df = resample_to_seconds(df)
                df, total_dist, total_ascent = compute_distance_and_ascent(df)
                df = compute_speed(df)
                df = compute_grade(df)
    
            # 2. Physics Prep
                df['bearing'] = calculate_bearing(df)

            # 3. AUTOMATIC WEATHER FETCH
            # We attempt to fetch it immediately. If it fails, df remains unchanged.
                df = fetch_ride_weather(df)

                moving_thresh = 2.0 #m/s
                is_moving = df['speed'] >= moving_thresh

                moving_time_sec = is_moving.sum()
                elapsed_time_sec = len(df) #Assuming 1Hz resampled

        except Exception as e:
            st.error(f"Error parsing GPX: {e}")
            if os.path.exists("temp.gpx"): os.remove("temp.gpx")
            return

        # Basic Stats
        duration_s = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds()
        if moving_time_sec > 0:
            avg_speed = df.loc[is_moving, 'speed'].mean() * 3.6
        else:
            avg_speed = 0

        avg_power = df.loc[is_moving, 'power'].mean() if 'power' in df.columns and moving_time_sec > 0 else 0
        avg_hr = df.loc[is_moving, 'hr'].mean() if 'hr' in df.columns and moving_time_sec > 0 else 0
        norm_power = NP(df['power']) if 'power' in df.columns else 0
        current_ftp = settings['ftp']
        ride_if = IF(norm_power, current_ftp)
        ride_tss = TSS(norm_power, duration_s, current_ftp)
        ride_kj = calculate_ride_kJ(df['power']) if 'power' in df.columns else 0

        current_curve = {}
        if 'power' in df.columns:
            current_curve = power_curve(df['power'])
        settings['w_prime_cap'] = 20000.0
        #W' Balance Calculation (if power data exists)
        if 'power' in df.columns and current_ftp > 0:
            df['w_prime_balance'] = calculate_w_prime_balance(df['power'], current_ftp)

        # Auto-Save
        if user_name:
             _save_ride_to_db(user_name, file_obj.name, file_bytes, df, total_dist, total_ascent, avg_speed, norm_power, ride_tss, ride_if, current_curve)

    # Detect Climbs
    detected_climbs = []
    if settings['show_climbs'] or settings['show_map']:
         detected_climbs = detect_climbs(df, min_length=1000, join_dist=1000)

    # Dashboard
    st.subheader("Ride Summary")
    
    # Display Weather Badge if data exists
    if 'wind_speed' in df.columns:
        avg_w = df['wind_speed'].mean() * 3.6
        avg_t = df['temp_c'].mean()
        st.caption(f"🌤️ Weather Data Integrated: {avg_t:.1f}°C, Wind ~{avg_w:.1f} km/h")
    
    dHHMMSS = format_duration(duration_s)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Distance", f"{total_dist/1000:.2f} km")
    c2.metric("Elevation", f"{int(total_ascent)} m")
    c3.metric("Avg Speed", f"{avg_speed:.1f} km/h")
    c4.metric("Avg Power", f"{int(avg_power)} W" if avg_power else "N/A")
    c5.metric("Duration", f"{dHHMMSS}")
    
    st.subheader("Performance Metrics")
    c6, c7, c8, c9 = st.columns(4)
    c6.metric("Norm Power", f"{int(norm_power)} W" if norm_power else "N/A")
    c7.metric("Intensity Factor", f"{ride_if:.2f}" if norm_power else "N/A")
    c8.metric("TSS", f"{int(ride_tss)}" if norm_power else "N/A")
    c9.metric("Work", f"{int(ride_kj)} kJ" if norm_power else "N/A")
    
    st.markdown("---")

    # Visualizations
    _render_plots(df, settings, detected_climbs, current_curve, user_name)
    
    if os.path.exists("temp.gpx"): os.remove("temp.gpx")

def _save_ride_to_db(user_name, filename, file_bytes, df, dist, ele, speed, np_val, tss, if_val, power_curve_dict):
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
        'tss': int(tss),
        'if': if_val,
        'avg_hr': safe_mean(df['hr']) if 'hr' in df.columns else 0
    }
    
    if 'saved_hashes' not in st.session_state:
        st.session_state['saved_hashes'] = set()
        
    file_hash = hashlib.md5(file_bytes).hexdigest()
    
    if file_hash not in st.session_state['saved_hashes']:
        success, msg = save_ride(user_name, filename, file_bytes, stats, power_curve_dict)
        if success:
            st.toast(f"✅ Saved to History: {filename}")
            st.session_state['saved_hashes'].add(file_hash)
        elif "duplicate" in msg.lower():
            st.session_state['saved_hashes'].add(file_hash)
        else:
            st.error(f"Save Error: {msg}")

def _render_plots(df, settings, detected_climbs, current_curve, user_name):

    if settings['show_map']:
        st.subheader("Route Map")
        map_obj = create_route_map(df, detected_climbs)
        if map_obj: st_folium(map_obj, width=1000, height=500)
        else: st.warning("No GPS coordinates found.")

    if settings['show_ele']:
        st.subheader("Elevation Profile")
        show_centered(plot_elevation(df))

    # --- CdA SECTION ---
    if settings.get('show_aero', True) and 'power' in df.columns and 'speed' in df.columns:
        st.subheader("💨 Aerodynamic Analysis & Power Budget")
        
        # Calculation happens automatically here using available columns
        df['cda'], df['p_aero'] = calculate_CdA(df, rider_mass=settings.get('weight', 70))
        avg_cda = get_avg_cda(df['cda'])
        avg_p_aero = df['p_aero'].mean() if 'p_aero' in df.columns else None
        if pd.notna(avg_cda):
            c1, c2 = st.columns(2)        
            c1.metric("Est. CdA", f"{avg_cda:.3f} m²" if pd.notna(avg_cda) else "N/A")
            c2.metric("Avg Aero Power", f"{int(avg_p_aero)} W" if pd.notna(avg_p_aero) else "N/A")
        else:
            st.warning("Not enough stable data points (Speed > 18km/h required).")

        st.markdown("**Power Budget: Where did your watts go?**")
        with st.spinner("Calculating power components..."):
            df['p_grav'], df['p_roll'], df['p_accel']  = calculate_power_components(df, settings['weight'])
            if not df.empty:
                fig_budget = plot_power_budget(df)
                show_centered(fig_budget)
                
    if settings['show_power']:
        st.subheader("Speed")
        show_centered(plot_x_time(df, 'speed_kmh', 'Speed (km/h)'))

        st.subheader("Power")
        if 'power' not in df.columns:
            st.info("No power data available.")
        else:
            show_centered(plot_x_time(df, 'power', 'Power (W)'))

        st.subheader("Cadence")
        if 'cadence' in df.columns:
            show_centered(plot_x_time(df, 'cadence', 'Cadence (rpm)'))
        else:
            st.info("No cadence data available.")

        st.subheader("Heart Rate")
        if 'hr' in df.columns:
            show_centered(plot_x_time(df, 'hr', 'Heart Rate (bpm)'))
        else:
            st.info("No heart rate data available.")

        st.subheader("CdA")
        if 'cda' in df.columns:
            show_centered(plot_x_time(df, 'cda', 'CdA (m²)'))
        else:
            st.info("No CdA data available.")

        st.subheader("W' Balance")
        if 'w_prime_balance' in df.columns:
            show_centered(plot_w_prime_balance(df,df['w_prime_balance'],settings['w_prime_cap']))
        else:
            st.info("No W' Balance data available.")

    if settings['show_curve'] and 'power' in df.columns:
        st.subheader("Power Curve")
        best_curve = None
        if user_name:
            best_curve = get_user_best_power(user_name)

        show_centered(plot_power_curve(power_curve(df['power']), best_curve))

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
        _render_climb_details(df, detected_climbs, settings)

def _render_climb_details(df, climbs, settings):
    st.subheader("Climb Analysis")
    if not climbs:
        st.info("No significant climbs detected.")
        return

    fig = plot_climbs(df, climbs)
    c1, c2, c3 = st.columns([1, 3, 1])
    with c2: st.pyplot(fig)
    
    user_weight = settings.get('weight', 70)
    data = []
    for i, c in enumerate(climbs):

        avg_p_climb = 0
        w_kg_climb = 0

        if 'power' in df.columns:
            seg_pwr = df['power'].iloc[c['start_idx']:c['end_idx']+1]
            if not seg_pwr.empty:
                avg_p_climb = int(seg_pwr.mean())
                w_kg_climb = round(avg_p_climb / user_weight, 2)

        vam = c['vam_mph']
        grade = c['avg_gradient_pct']

        est_w_kg = 0.0
        den = 100* (2 + grade/10.0) # Rough empirical formula (Ferrari formula)
        if den > 0:
            est_w_kg = round(vam / den, 2)

        data.append({
            "ID": i,
            "Climb": f"#{i+1}",
            #            "Category": c.get('type', 'Unknown'),
            "Len (km)": round(c['length_m']/1000, 2),
            "Gain (m)": int(c['elev_gain_m']),
            "Grad (%)": round(c['avg_gradient_pct'], 1),
            "VAM": int(c['vam_mph']),
            "Power (W)": avg_p_climb if avg_p_climb > 0 else "-",
            "W/kg": w_kg_climb if w_kg_climb > 0 else "-",
            "Est W/kg": est_w_kg#,
            #            "Error W/kg (%)": round(100*np.abs(w_kg_climb - est_w_kg)/w_kg_climb, 2) if w_kg_climb > 0 else "-"
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
        fig_det = plot_detailed_climb(df, c['start_idx'], c['end_idx'], segs, settings['weight'])
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
    st.info("Here is a log of your past rides.")

    st.dataframe(
        hist_df[['date_time', 'filename', 'distance_km', 'elevation_m', 'norm_power','tss']],
        hide_index=True,
    )

def render_user_corner(user_name):
    st.header("👤 User Corner")

    if not user_name:
        st.warning("Please select a user in the sidebar to view profile details.")
        return
        
    user_data = get_user_data(user_name)
    if not user_data:
        st.error("User data not found.")
        return

    # 1. Profile Summary
    c1, c2, c3 = st.columns(3)
    c1.metric("FTP", f"{user_data['ftp']} W")
    c2.metric("LTHR", f"{user_data['lthr']} bpm")
    c3.metric("Weight", f"{user_data['weight']} kg")
    
    st.markdown("---")

    # 2. Performance Management Chart (PMC)
    st.subheader("📈 Fitness & Form (PMC)")
    
    # Fetch Data
    history_data = get_user_tss_history(user_name)
    
    # --- CHANGE: Allow chart if we have ANY data (> 0) ---
    if len(history_data) > 0:
        # Create DataFrame
        pmc_df = pd.DataFrame(history_data)
        pmc_df['date'] = pd.to_datetime(pmc_df['date'])
        
        # Sort just in case
        pmc_df = pmc_df.sort_values('date')
        
        # Set index to date
        pmc_df = pmc_df.set_index('date')
        
        # Resample to Daily (Sum TSS if multiple rides per day)
        # We assume 0 TSS for days with no rides between start and end
        if len(pmc_df) > 1:
            full_idx = pd.date_range(start=pmc_df.index.min(), end=pmc_df.index.max(), freq='D')
            pmc_df = pmc_df.resample('D').sum().reindex(full_idx, fill_value=0)
        else:
            # If only 1 day of data, just keep it as is (no resampling needed yet)
            pass
        
        # Calculate Metrics (Exponential Weighted Moving Averages)
        # CTL = 42-day time constant
        # ATL = 7-day time constant
        # We use min_periods=1 so it calculates even with 1 data point
        pmc_df['CTL'] = pmc_df['tss'].ewm(alpha=1/42, adjust=False, min_periods=1).mean()
        pmc_df['ATL'] = pmc_df['tss'].ewm(alpha=1/7, adjust=False, min_periods=1).mean()
        
        # TSB (Form) = CTL - ATL
        pmc_df['TSB'] = pmc_df['CTL'] - pmc_df['ATL']
        
        # Show Plot
#        st.pyplot(plot_pmc(pmc_df))
        
        # Show Current Status
        curr = pmc_df.iloc[-1]
#        k1, k2, k3 = st.columns(3)
#        k1.metric("Fitness (CTL)", f"{curr['CTL']:.1f}", help="Chronic Training Load (42-day avg)")
#        k2.metric("Fatigue (ATL)", f"{curr['ATL']:.1f}", help="Acute Training Load (7-day avg)")
#        k3.metric("Form (TSB)", f"{curr['TSB']:.1f}", 
#                  delta=f"{curr['TSB']:.1f}", delta_color="normal",
#                  help="Training Stress Balance (Fitness - Fatigue)")
                  
        # DEBUG: Show raw data if user wants to check
 #       with st.expander("View Raw PMC Data"):
 #           st.dataframe(pmc_df)
                  
    else:
        st.info("No TSS history data available. Upload a ride to see your chart!")

    st.markdown("---")

    # 3. Best Power Records
    best_power = get_user_best_power(user_name)
    if best_power:
        st.subheader("🏆 Best Power Records")
        data = []
        for duration, watts in sorted(best_power.items()):
            dur_label = f"{duration}s"
            if duration >= 60:
                dur_label = f"{int(duration/60)}m"
            if duration >= 3600:
                dur_label = f"{int(duration/3600)}h"
            
            data.append({
                "Duration": dur_label, 
                "Watts": int(watts), 
                "W/kg": round(watts / user_data['weight'], 2)
            })
        
        st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)   

