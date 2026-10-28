import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import hashlib
from streamlit_folium import st_folium

import plotly.graph_objects as go

# Imports from your existing modules
from src.data.gpx import calculate_bearing
from src.data.fit import get_fit_laps
from src.data.history import recompute_history
from src.data.pipeline import prepare_ride, summarize_ride, ride_kind, metrics_for_storage
from src.viz.plots import (
        plot_elevation, plot_x_time, plot_power_curve, plot_zone_distribution, 
        plot_climbs, plot_detailed_climb, plot_power_budget, plot_w_prime_balance,
        plot_pmc, plot_joint_heatmap, plot_y_by_power, plot_trend, plot_quadrants,
        plot_cadence_by_zone
    )
from src.analysis.power import (
        NP, IF, TSS, power_curve, time_in_zones, coggan_zones, 
        calculate_w_prime_balance, calculate_ride_kJ,
        estimate_critical_power
    )
from src.analysis.hr import estimate_hr_threshold, time_in_hr_zones
from src.analysis.training_load import compute_pmc
from src.analysis import correlations as co
from src.analysis import power_profile as pp
from src.data.db import (
        save_user, get_all_users, delete_user, save_ride, build_ride_metadata, update_user_crank,
        get_strava_auth, save_strava_auth, delete_strava_auth, create_oauth_state, consume_oauth_state
    )
from src.data.cached import (get_user_data, get_user_rides, get_user_best_power, get_user_tss_history,
                             count_rides_needing_update, get_hist, get_ride_stats, period_coverage,
                             get_power_records)
from src.analysis.climbs import detect_climbs, get_climb_segments
from src.viz.maps import create_route_map
from src.viz.sections import section_picker
from src.physics.aerodyn import calculate_CdA, get_avg_cda, calculate_power_components
from src.data.weather import fetch_ride_weather  # Ensure this is imported
from src.data import strava as sv
from src.data.strava_import import (get_valid_auth, list_recent_rides, fetch_ride_file,
                                    import_history)
from src.physics.thermal import calculate_thermal_profile, CLOTHING_CATALOG

@st.cache_data(show_spinner=False)
def run_heavy_analysis(df, bike_mass, crr, drivetrain_loss, rider_mass, ftp, w_prime_cap_j=20000):
    """
    Bundles all heavy math into one cached function.
    Returns: (df_enriched, climbs_list, power_components_df)
    """
    # 1. Physics Engine (Aerodyn)
    # Note: calculate_power_components returns a separate DF, 
    # but we might want to attach CdA to the main DF too.
    p_comps = calculate_power_components(
        df, 
        rider_mass=rider_mass, 
        bike_mass=bike_mass, 
        crr=crr, 
        drivetrain_loss=drivetrain_loss
    )
    
    # 2. Climb Detection
    climbs = detect_climbs(df, min_length=1000, join_dist=1000)
    
    # 3. W' Balance (if FTP exists)
    if ftp and 'power' in df.columns:
        df['w_prime_balance'] = calculate_w_prime_balance(df['power'], ftp, w_prime_cap_j=w_prime_cap_j)
        breakthrough = df['w_prime_balance'].min()
        if breakthrough < 0:
            st.success(f"🚀 **Fitness Breakthrough Detected!**")
            st.write(f"You depleted your modeled W' battery and kept going. You went **{abs(breakthrough)/1000:.1f} kJ** into the negative.")
    
            st.info("""
    **Interpretation:** Your actual CP or W' is higher than your current settings.
    - If this was a short, punchy ride: Increase your **W'**.
    - If this was a long, steady hard ride: Increase your **CP/FTP**.
    """)
    else:
        df['w_prime_balance'] = np.nan
        
    return df, climbs, p_comps

def show_centered(fig):
    c1, c2, c3 = st.columns([1, 3, 1])
    with c2: st.plotly_chart(fig, use_container_width=True)

def duration_label(seconds):
    """Compact label for power-curve durations: 5s, 1m30s, 7m, 1h30m."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m" if m else f"{h}h"
    if m:
        return f"{m}m{s:02d}s" if s else f"{m}m"
    return f"{s}s"

def format_duration(seconds):
    if not seconds: return "00:00:00"

    m, s =divmod(int(seconds), 60)
    h, m =divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02}"

def render_sidebar():
    """Renders the sidebar and returns all user configuration settings."""
    
    user_ftp = 200
    user_weight = 70.0
    user_lthr = 170
    user_height = 175

    st.sidebar.header("Navigation")
    
    app_mode = st.sidebar.radio("Go to", ["Analyze Upload", "Training", "User Corner"])

    st.sidebar.markdown("---")
    st.sidebar.header("👤 User Profile")

    # User Management
    tab_select, tab_create = st.sidebar.tabs(["Select User", "Create/Edit"])
    
    selected_user_name = None
    
    with tab_select:
        users = get_all_users()
        if users:
            selected_user_name = st.selectbox("Current User", users, key="current_user")
            if selected_user_name:
                u_data = get_user_data(selected_user_name)
                if u_data:
                    user_ftp = u_data['ftp']
                    user_lthr = u_data['lthr']
                    user_weight = u_data['weight']
                    user_height = u_data['height']


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
            new_height = st.number_input("Height (cm)", value=175)
            submitted = st.form_submit_button("Save User")
            if submitted and new_name:
                success, msg = save_user(new_name, new_ftp, new_lthr, new_weight, new_height)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    st.sidebar.markdown("---")
    st.sidebar.header("📂 Data Import")
    
    # 1. Initialize current_file to None by default
    current_file = None
    
    import_method = st.sidebar.radio("Source", ["Upload Ride", "Strava"])

    if import_method == "Upload Ride":
        uploaded_file = st.sidebar.file_uploader("Upload file ride", type=["gpx", "fit"])
        # If user uploaded a file, assign it to current_file
        if uploaded_file is not None:
            current_file = uploaded_file

    elif import_method == "Strava":
        current_file = _render_strava_sidebar(selected_user_name, user_ftp, user_lthr)

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
        'weight': user_weight,
        'height': user_height
    }

    # RETURN 'current_file' (which works for both logic paths)
    return app_mode, selected_user_name, current_file, settings

# --- Strava (official API) ---------------------------------------------------

def _strava_config():
    """App credentials from .streamlit/secrets.toml ([strava] section), or None."""
    try:
        sec = st.secrets["strava"]
        return sv.StravaConfig(
            client_id=str(sec["client_id"]),
            client_secret=str(sec["client_secret"]),
            redirect_uri=str(sec.get("redirect_uri", "http://localhost:8501/")),
        )
    except Exception:
        return None

def _strava_usage():
    if 'strava_usage' not in st.session_state:
        st.session_state['strava_usage'] = sv.RateUsage()
    return st.session_state['strava_usage']

def handle_strava_callback():
    """
    Completes the OAuth round trip when Strava redirects back to the app
    (?code=...&scope=...&state=...). Call before rendering the sidebar.
    """
    params = st.query_params
    if "state" not in params or not ("code" in params or "error" in params):
        return

    state, code = params.get("state"), params.get("code")
    error, scope = params.get("error"), params.get("scope")
    st.query_params.clear()  # the code is single-use; never process it twice

    user_name = consume_oauth_state(state)
    if not user_name:
        st.error("Strava connection failed: the request expired or did not come from this app. Please try again.")
        return
    st.session_state["current_user"] = user_name

    if error:
        st.warning("Strava access was not granted.")
        return
    if not sv.granted_scope_ok(scope):
        st.warning("Strava was connected without permission to read activities. "
                   "Please reconnect and keep the activity access box ticked.")
        return

    cfg = _strava_config()
    if cfg is None:
        st.error("Strava is not configured (see README).")
        return
    try:
        auth = sv.exchange_code(cfg, code)
    except sv.StravaError as e:
        st.error(f"Strava connection failed: {e}")
        return
    save_strava_auth(user_name, auth)
    st.toast(f"✅ Strava connected for {user_name}")

def _activity_label(a):
    when = (a.get("start_date_local") or a.get("start_date") or "")[:10]
    km = (a.get("distance") or 0) / 1000
    return f"{when} · {a.get('name', 'Ride')} · {km:.0f} km"

def _render_strava_sidebar(user_name, user_ftp, user_lthr=None):
    """Returns a file-like ride to analyse, or None."""
    cfg = _strava_config()
    if cfg is None:
        st.sidebar.warning("Strava isn't set up yet. Add your API app's `client_id` and "
                           "`client_secret` to `.streamlit/secrets.toml` (see README).")
        return None
    if not user_name:
        st.sidebar.info("Select or create a user profile first.")
        return None

    auth = get_strava_auth(user_name)
    if not auth:
        key = f"oauth_state_{user_name}"
        if key not in st.session_state:
            st.session_state[key] = create_oauth_state(user_name)
        st.sidebar.link_button("🔗 Connect Strava", sv.authorize_url(cfg, st.session_state[key]))
        st.sidebar.caption("Strava opens in a new tab; you'll continue there after approving.")
        return None

    who = auth.get("athlete_name") or "your account"
    st.sidebar.caption(f"Connected to Strava as **{who}**")
    usage = _strava_usage()
    current_file = None

    try:
        # Recent rides (cached per session to spare the rate limit)
        cache_key = f"strava_recent_{user_name}"
        if cache_key not in st.session_state or st.sidebar.button("↻ Refresh list"):
            with st.spinner("Loading your Strava rides..."):
                st.session_state[cache_key] = list_recent_rides(cfg, user_name, usage=usage)
        recent = st.session_state[cache_key]

        if recent:
            choice = st.sidebar.selectbox("Recent rides", recent, format_func=_activity_label)
            if st.sidebar.button("Analyze ride"):
                with st.spinner("Downloading ride from Strava..."):
                    name, payload = fetch_ride_file(cfg, user_name, choice, usage=usage)
                file_obj = io.BytesIO(payload)
                file_obj.name = name
                st.session_state[f'strava_file_{user_name}'] = file_obj
        else:
            st.sidebar.info("No recent rides found on Strava.")

        with st.sidebar.expander("📥 Import ride history"):
            st.caption("Imports every ride since the chosen date into your history "
                       "(PMC, power records). Rides already saved are skipped.")
            since = st.date_input("Since", value=pd.Timestamp.today().date() - pd.Timedelta(days=90))
            if st.button("Import history"):
                bar = st.progress(0.0, text="Listing activities...")

                def _progress(done, total, name):
                    frac = done / total if total else 1.0
                    bar.progress(frac, text=f"{done}/{total} {name}"[:60])

                res = import_history(cfg, user_name, since, user_ftp, progress=_progress, usage=usage,
                                     lthr=user_lthr)
                bar.empty()
                st.success(f"Imported {res.imported} of {res.found} rides.")
                skipped = []
                if res.already_imported: skipped.append(f"{res.already_imported} already imported")
                if res.duplicates_of_uploads: skipped.append(f"{res.duplicates_of_uploads} matched uploaded files")
                if res.no_data: skipped.append(f"{res.no_data} without recorded data")
                if skipped:
                    st.caption("Skipped: " + ", ".join(skipped) + ".")
                if res.failed:
                    st.warning(f"{len(res.failed)} failed:\n\n" + "\n".join(f"- {f}" for f in res.failed[:10]))
                if res.stopped_reason:
                    st.info(f"{res.stopped_reason} ({res.remaining} left)")

    except sv.StravaAuthError as e:
        st.sidebar.error(f"{e}")
    except sv.StravaError as e:
        st.sidebar.error(f"Strava: {e}")

    if st.sidebar.button("Disconnect Strava"):
        sv.revoke(cfg, auth["refresh_token"])
        delete_strava_auth(user_name)
        for k in [f"strava_recent_{user_name}", f"strava_file_{user_name}"]:
            st.session_state.pop(k, None)
        st.rerun()

    # Keyed per profile so a ride is never auto-saved to the wrong rider
    current_file = st.session_state.get(f'strava_file_{user_name}', current_file)
    return current_file

def process_and_display_analysis(file_obj, user_name, settings):
    """Main logic to parse GPX, calculate stats, and render plots."""
    
    file_bytes = file_obj.getvalue()
    filename = file_obj.name.lower()
    
    with st.spinner(f"Processing {file_obj.name} and fetching weather..."):
        try:
            df, total_dist, total_ascent = prepare_ride(filename, file_bytes)
            laps_df = get_fit_laps(file_bytes) if ride_kind(filename) == "fit" else None

            # Physics prep + automatic weather fetch (df unchanged if it fails)
            df['bearing'] = calculate_bearing(df)
            df = fetch_ride_weather(df)
        except Exception as e:
            st.error(f"Error parsing ride file: {e}")
            return

        current_ftp = settings['ftp']
        summary = summarize_ride(df, current_ftp, settings.get('lthr'))
        duration_s = summary['duration_s']
        avg_speed = summary['avg_speed_kph']
        avg_power = summary['avg_power']
        norm_power = summary['norm_power']

        with st.spinner("Running analysis engine..."):
            df, climbs, p_comps = run_heavy_analysis(
                df,
                bike_mass=8.0,
                crr=0.005,
                drivetrain_loss=0.03,
                rider_mass=settings['weight'],
                ftp=current_ftp,
                w_prime_cap_j=settings.get('w_prime', 20000.0)
            )

        ride_if = summary['if']
        ride_tss = summary['tss']
        ride_kj = summary['kj']
        current_curve = summary['power_curve']
        settings['w_prime_cap'] = 20000.0

        # Auto-Save
        if user_name:
             _save_ride_to_db(user_name, file_obj.name, file_bytes, df, total_dist, total_ascent, summary)

    # Detect Climbs
    detected_climbs = []
    if settings['show_climbs'] or settings['show_map']:
         detected_climbs = climbs

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
    if summary['tss_source'] == 'hr':
        c8.metric("TSS (HR)", f"{round(ride_tss)}",
                  help="No power data: estimated from heart rate vs your LTHR (hrTSS). "
                       "Less precise than power-based TSS.")
    else:
        c8.metric("TSS", f"{round(ride_tss)}" if summary['tss_source'] else "N/A")
    c9.metric("Work", f"{int(ride_kj)} kJ" if norm_power else "N/A")

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Moving Time", format_duration(summary['moving_time_s']))
    q2.metric("Variability Index", f"{summary['vi']:.2f}" if summary['vi'] else "N/A",
              help="NP / average power. ~1.00-1.05 = steady (TT, endurance); >1.15 = very variable (crits, group rides).")
    q3.metric("Efficiency Factor", f"{summary['ef']:.2f}" if summary['ef'] else "N/A",
              help="NP / average heart rate. Compare across similar endurance rides: rising EF = improving aerobic fitness.")
    dec = summary['decoupling_pct']
    q4.metric("Aerobic Decoupling", f"{dec:.1f} %" if dec is not None else "N/A",
              help="Drift of power:HR between the first and second half (Pa:HR). Under ~5 % on a long steady ride "
                   "suggests good aerobic endurance. Needs 20+ min with power and HR; not meaningful for intervals.")
    
    st.markdown("---")

    if laps_df is not None:
        # Extract laps and show summary
        st.subheader("Laps Summary")
        if not laps_df.empty:
            with st.expander("⏱️ Laps / Intervals"):
                st.dataframe(laps_df)
    
    # --- NEW: THERMAL SECTION ---
    # Only show if we have power data, as the model depends on it
    if 'power' in df.columns:
        _render_thermal_analysis(df, settings['weight'], settings['height'])    

    # Visualizations
    _render_plots(df, settings, detected_climbs, current_curve, p_comps, user_name)

def _save_ride_to_db(user_name, filename, file_bytes, df, dist, ele, summary):
    """Helper to handle database saving logic."""
    stats = build_ride_metadata(df, dist, ele, summary['avg_speed_kph'], summary['norm_power'],
                                summary['tss'], summary['if'], filename=filename,
                                metrics=metrics_for_storage(summary))
    power_curve_dict = summary['power_curve']

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

def _render_plots(df, settings, detected_climbs, current_curve, p_comps, user_name):

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
            df['p_grav'], df['p_roll'], df['p_accel']  = p_comps
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
            show_centered(plot_w_prime_balance(df,settings['w_prime_cap']))
        else:
            st.info("No W' Balance data available.")

    if settings['show_curve'] and 'power' in df.columns:
        st.subheader("Power Curve")
        best_curve, recent_curve = None, None
        if user_name:
            best_curve = get_user_best_power(user_name)
            today = pd.Timestamp.today().normalize()
            recent_curve = pp.as_dict(pp.best_curve(
                get_power_records(user_name, today - pd.Timedelta(days=89), today)))

        show_centered(plot_power_curve(power_curve(df['power']), best_curve, recent_curve))

    if settings['show_zones']:
        st.subheader("Zone Distribution")
        c1, c2 = st.columns(2)
        if 'power' in df.columns and settings['ftp']:
            zones = coggan_zones(settings['ftp'])
            times = time_in_zones(df['power'], zones)
            fig = plot_zone_distribution(times, "Power Zones")
            if fig: c1.plotly_chart(fig, use_container_width=True)
        
        active_lthr = settings['lthr']
        if not active_lthr and 'hr' in df.columns:
             est, _ = estimate_hr_threshold(df['hr'])
             active_lthr = est
        
        if 'hr' in df.columns and active_lthr:
            times = time_in_hr_zones(df['hr'], active_lthr)
            fig = plot_zone_distribution(times, "Heart Rate Zones")
            if fig: c2.plotly_chart(fig, use_container_width=True)

    if settings['show_climbs']:
        _render_climb_details(df, detected_climbs, settings)

def _render_climb_details(df, climbs, settings):
    if climbs:
        st.subheader(f"⛰️ Detected Climbs ({len(climbs)})")        
        # 1. Overview Plot (Interactive)
        fig_overview = plot_climbs(df, climbs)
        st.plotly_chart(fig_overview, use_container_width=True)
        
        # 2. Detailed Breakdown
        with st.expander("View Details for Each Climb"):
            for i, c in enumerate(climbs):
                st.markdown(f"### Climb #{i+1}: {c['type']}")
            
                # Metrics Row
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Length", f"{c['length_m']/1000:.2f} km")
                c2.metric("Avg Gradient", f"{c['avg_gradient_pct']:.1f}%")
                c3.metric("Ascent", f"{int(c['elev_gain_m'])} m")
                
                # Check if VAM exists (depends on your physics calc)
                vam = c.get('vam_mph', 0) # Vertical Ascent Meters per Hour
                c4.metric("VAM", f"{int(vam)} m/h")
            
            # 3. Interactive Detailed Plot
                segment = df.loc[c['start_idx']:c['end_idx']]
                if not segment.empty:
                    fig_climb = plot_detailed_climb(segment)
                    st.plotly_chart(fig_climb, use_container_width=True, key=f"climb_plot_{i}")
                
                st.markdown("---")
    else:
        st.info("No significant climbs detected on this route.")

def _render_thermal_analysis(df, user_weight, user_height):
    """
    Renders the Thermal Analysis interface with Distance x Elevation plot.
    """
    st.subheader("🔥 Thermal Analysis")
    
    # Unique ID for session state
    start_time_str = str(df['time'].iloc[0]) if 'time' in df.columns else "0"
    current_ride_id = f"{len(df)}_{start_time_str}"

    with st.expander("⚙️ Configure Thermal Model", expanded=False):
        # Clothing Selector
        default_kit = ["Summer Jersey", "None (Shorts only)"]
        selected_clothing = st.multiselect(
            "Select Clothing Layers Worn:",
            options=list(CLOTHING_CATALOG.keys()),
            default=[k for k in default_kit if k in CLOTHING_CATALOG],
            key="thermal_clothing_multiselect"
        )
        
        total_clo = sum([CLOTHING_CATALOG[item] for item in selected_clothing])
        st.caption(f"Total Insulation: **{total_clo:.2f} Clo**")
        
        # Trigger Button
        if st.button("Run Thermal Simulation"):
            with st.spinner("Simulating thermodynamics..."):
                thermal_df = calculate_thermal_profile(
                    df, 
                    rider_weight_kg=user_weight, 
                    rider_height_cm=user_height, 
                    clothing_items=selected_clothing
                )
                
                # Save to session state
                st.session_state['thermal_results'] = {
                    'id': current_ride_id,
                    'df': thermal_df
                }

    # Plotting Logic
    if 'thermal_results' in st.session_state:
        stored_result = st.session_state['thermal_results']
        
        if stored_result['id'] == current_ride_id:
            thermal_df = stored_result['df']
            
            # --- PREPARE DATA ---
            # X-Axis: Distance in km
            if 'dist_m' in df.columns:
                x_data = df['dist_m'] / 1000.0
                x_label = "Distance (km)"
            else:
                x_data = thermal_df.index
                x_label = "Index"

            # Y-Axis 2: Elevation (Handle naming variations)
            if 'ele' in thermal_df.columns:
                ele_data = thermal_df['ele']
            elif 'elevation' in thermal_df.columns:
                ele_data = thermal_df['elevation']
            elif 'altitude' in thermal_df.columns:
                ele_data = thermal_df['altitude']
            else:
                ele_data = [0] * len(thermal_df)

            # --- PLOT ---
            fig = go.Figure()
            
            # 1. Elevation Profile (Background Area) 

            fig.add_trace(go.Scatter(
                x=x_data,
                y=ele_data,
                name="Elevation (m)",
                line=dict(width=1, color='gray'),
                fill='tozeroy',
                fillcolor='rgba(128, 128, 128, 0.2)', # Transparent gray
                yaxis='y2',
                hoverinfo='y'
            ))

            # 2. Optimal & Heat Stress Zones (Background Bands)
            # We add these first so the line draws ON TOP
            fig.add_hrect(y0=36, y1=37, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Optimal")
            fig.add_hrect(y0=38.5, y1=42.0, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Heat Stress")

            # 3. Core Temperature (Main Line)
            fig.add_trace(go.Scatter(
                x=x_data, 
                y=thermal_df['core_temp'],
                name="Core Temp (°C)",
                line=dict(color='firebrick', width=3),
                yaxis='y1'
            ))
            
            # Layout
            fig.update_layout(
                title="Body Core Temperature",
                xaxis=dict(title=x_label),
                # Left Axis: Temperature
                yaxis=dict(
                    title="Core Temperature (°C)", 
                    range=[35.5, 40.0],
                    showgrid=True
                ),
                # Right Axis: Elevation
                yaxis2=dict(
                    title="Elevation (m)", 
                    overlaying='y', 
                    side='right', 
                    showgrid=False
                ),
                hovermode="x unified",
                template="plotly_white",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # --- METRICS ---
            max_core = thermal_df['core_temp'].max()
            avg_loss = thermal_df['heat_lost_W'].mean()
            avg_core = thermal_df['core_temp'].mean()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg Core Temp", f"{avg_core:.1f}°C", delta=f"{avg_core-36.5:.1f}°C")
            m2.metric("Peak Core Temp", f"{max_core:.1f}°C", delta=f"{max_core-36.5:.1f}°C")
            m3.metric("Avg Heat Loss", f"{int(avg_loss)} W")
            
            if max_core > 38.5:
                m4.error("⚠️ High Heat Strain")
            elif max_core < 36.2:
                m4.warning("❄️ Cooling Detected")
            else:
                m4.success("✅ Thermally Balanced")

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

    cols = ['date_time', 'filename', 'distance_km', 'elevation_m', 'moving_s', 'norm_power',
            'tss', 'tss_source', 'vi', 'ef', 'decoupling_pct']
    view = hist_df[[c for c in cols if c in hist_df.columns]].copy()
    if 'moving_s' in view.columns:
        view['moving_s'] = view['moving_s'].map(lambda x: format_duration(x) if pd.notna(x) else "")
    st.dataframe(
        view.rename(columns={'date_time': 'Date', 'filename': 'File', 'distance_km': 'km',
                             'elevation_m': 'Elev (m)', 'moving_s': 'Moving', 'norm_power': 'NP',
                             'tss': 'TSS', 'tss_source': 'TSS from', 'vi': 'VI', 'ef': 'EF',
                             'decoupling_pct': 'Decoupling %'}),
        hide_index=True,
        column_config={"km": st.column_config.NumberColumn(format="%.1f"),
                       "VI": st.column_config.NumberColumn(format="%.2f"),
                       "EF": st.column_config.NumberColumn(format="%.2f"),
                       "Decoupling %": st.column_config.NumberColumn(format="%.1f")},
    )

def _render_history_update(user_name, user_data):
    """Offers to rebuild outdated rides from their stored files."""
    n = count_rides_needing_update(user_name)
    if n == 0:
        return
    with st.expander(f"🛠️ Update ride history ({n} rides need updating)"):
        st.caption("These rides were saved by an older version of the app, so they are missing "
                   "newer metrics (moving time, VI, EF, decoupling, zone times, the full power "
                   "curve) or their date. This re-reads the stored files and recomputes them. "
                   "Each ride keeps the FTP it was originally scored with; rides without power "
                   "get an HR-based TSS using your LTHR.")
        if st.button("Update ride history"):
            bar = st.progress(0.0, text="Recomputing rides...")
            updated, failed = recompute_history(
                user_name, user_data['ftp'], user_data['lthr'],
                progress=lambda done, total: bar.progress(done / total if total else 1.0,
                                                          text=f"{done}/{total} rides"))
            bar.empty()
            if failed:
                st.warning(f"Updated {updated} rides; {len(failed)} could not be read:\n\n"
                           + "\n".join(f"- {f}" for f in failed[:10]))
            else:
                st.toast(f"✅ Updated {updated} rides.")
                st.rerun()

# --- Deeper analytics --------------------------------------------------------

def _fmt_delta(a, b, fmt="{:+.1f}"):
    return fmt.format(a - b) if a is not None and b is not None else None

def _render_deeper_analytics(user_name, user_data):
    st.subheader("🔬 Deeper Analytics")

    today = pd.Timestamp.today().normalize()
    c1, c2 = st.columns([2, 1])
    period = c1.date_input("Period", value=((today - pd.Timedelta(days=90)).date(), today.date()),
                           key="da_period")
    if not isinstance(period, (tuple, list)) or len(period) != 2:
        st.info("Pick a start and an end date.")
        return
    a0, a1 = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    compare = c2.checkbox("Compare with previous period", value=True, key="da_compare",
                          help="The same number of days immediately before the selected period.")
    span = (a1 - a0) + pd.Timedelta(days=1)
    b0, b1 = a0 - span, a0 - pd.Timedelta(days=1)
    lab_a = f"{a0:%d %b} – {a1:%d %b %Y}"
    lab_b = f"{b0:%d %b} – {b1:%d %b %Y}"

    ftp, lthr = user_data['ftp'], user_data['lthr']
    hr_a = get_hist(user_name, 'p_hr', a0, a1)
    cad_a = get_hist(user_name, 'p_cad', a0, a1)
    hr_b = get_hist(user_name, 'p_hr', b0, b1) if compare else None
    cad_b = get_hist(user_name, 'p_cad', b0, b1) if compare else None
    stats = get_ride_stats(user_name, b0 if compare else a0, a1)

    cov = period_coverage(user_name, a0, a1)
    if cov['needs_update']:
        st.warning(f"{cov['needs_update']} of the {cov['rides']} rides in this period were saved by an older "
                   "version and aren't included here yet. Run **Update ride history** above to add them.")

    def _empty_reason(what):
        if cov['rides'] == 0:
            return "No rides in this period."
        if cov['needs_update'] == cov['rides']:
            return f"None of the {cov['rides']} rides in this period has been updated yet (see above)."
        return (f"None of the {cov['rides'] - cov['needs_update']} up-to-date rides in this period "
                f"has both power and {what} while pedalling.")

    sections = ["❤️ Power ↔ Heart rate", "🦵 Power ↔ Cadence", "🔁 Recovery & Efficiency"]
    section = section_picker(sections, key="da_section")

    # ---- Power vs HR ----
    if section == sections[0]:
        if hr_a.empty:
            st.info(_empty_reason("heart rate"))
        else:
            fit_a = co.linear_fit(hr_a)
            fit_b = co.linear_fit(hr_b) if compare else None
            ref_p = int(round(0.65 * ftp / 5) * 5) if ftp else 150
            hr_ref_a = co.y_at_power(hr_a, ref_p)
            hr_ref_b = co.y_at_power(hr_b, ref_p) if compare else None

            m1, m2, m3 = st.columns(3)
            m1.metric("Correlation (r)", f"{fit_a['r']:.2f}" if fit_a else "N/A",
                      delta=_fmt_delta(fit_a and fit_a['r'], fit_b and fit_b['r'], "{:+.2f}"),
                      help="How tightly heart rate follows power (30 s averages, moving time). "
                           "Close to 1 = HR is a good proxy for effort; lower values usually mean "
                           "lots of short efforts, heat, fatigue or HR sensor noise.")
            m2.metric("HR rise per +10 W", f"{fit_a['slope'] * 10:.1f} bpm" if fit_a else "N/A",
                      delta=_fmt_delta(fit_a and fit_a['slope'] * 10, fit_b and fit_b['slope'] * 10),
                      delta_color="inverse",
                      help="Slope of the HR-vs-power line. A flatter slope means each extra watt costs less heart rate.")
            m3.metric(f"HR at {ref_p} W", f"{hr_ref_a:.0f} bpm" if hr_ref_a else "N/A",
                      delta=_fmt_delta(hr_ref_a, hr_ref_b, "{:+.0f} bpm"), delta_color="inverse",
                      help=f"Average HR while riding at {ref_p} W ±15 W (65 % of FTP). "
                           "Lower for the same power = better aerobic fitness (or more fatigue/heat if it rises).")
            if compare:
                st.caption(f"Deltas compare with {lab_b}.")

            curve_a = co.mean_y_by_power(hr_a)
            st.plotly_chart(plot_joint_heatmap(co.aggregate(hr_a), 5, "Heart rate (bpm, 30 s avg)",
                                               f"Time at power × heart rate · {lab_a}", overlay=curve_a),
                            use_container_width=True)
            curves = {lab_a: curve_a}
            if compare and hr_b is not None and not hr_b.empty:
                curves[lab_b] = co.mean_y_by_power(hr_b)
            st.plotly_chart(plot_y_by_power(curves, "Average HR (bpm)", "Heart rate at each power"),
                            use_container_width=True)
            st.caption("If this curve moves **down** between periods, you are producing the same power "
                       "at a lower heart rate: the clearest sign of improving aerobic fitness.")

            hr_all = pd.concat([x for x in [hr_b, hr_a] if x is not None and not x.empty])
            weekly_hr = co.weekly(hr_all, lambda g: co.y_at_power(g, ref_p, min_seconds=180))
            if len(weekly_hr) >= 2:
                st.plotly_chart(plot_trend({f"HR at {ref_p} W": weekly_hr}, "bpm",
                                           f"Weekly heart rate at {ref_p} W"), use_container_width=True)

    # ---- Power vs cadence ----
    if section == sections[1]:
        if cad_a.empty:
            st.info(_empty_reason("cadence"))
        else:
            crank_mm = float(user_data.get('crank_mm') or 172.5)
            new_crank = st.number_input("Crank length (mm)", min_value=150.0, max_value=200.0,
                                        value=crank_mm, step=2.5, key="da_crank",
                                        help="Only affects the force/velocity axes of the quadrant chart.")
            if new_crank != crank_mm:
                update_user_crank(user_name, new_crank)
                crank_mm = new_crank

            ref_cad = co.weighted_median_y(cad_a)
            ref_cad_b = co.weighted_median_y(cad_b) if compare else None
            quads = co.quadrants(cad_a, ftp, ref_cad)
            quads_b = co.quadrants(cad_b, ftp, ref_cad) if compare else None

            m1, m2, m3 = st.columns(3)
            m1.metric("Median cadence", f"{ref_cad:.0f} rpm" if ref_cad else "N/A",
                      delta=_fmt_delta(ref_cad, ref_cad_b, "{:+.0f} rpm"), delta_color="off",
                      help="Your typical cadence while pedalling in this period. Used as the quadrant reference.")
            m2.metric("High-force time (QI + QII)",
                      f"{(quads['QI'] + quads['QII']):.0%}" if quads else "N/A",
                      delta=(f"{(quads['QI'] + quads['QII']) - (quads_b['QI'] + quads_b['QII']):+.0%}"
                             if quads and quads_b else None), delta_color="off",
                      help="Share of pedalling time pushing harder than the force that produces FTP at your "
                           "median cadence: climbing, accelerations, hard efforts.")
            fit_c = co.linear_fit(cad_a, y_bin=5)
            m3.metric("Cadence rise per +50 W", f"{fit_c['slope'] * 50:.1f} rpm" if fit_c else "N/A",
                      help="How much you spin faster as power goes up. Near zero = you mostly push harder "
                           "at the same cadence (often seen on climbs or with a big gear).")

            cc1, cc2 = st.columns(2)
            cc1.plotly_chart(plot_joint_heatmap(co.aggregate(cad_a), 5, "Cadence (rpm)",
                                                f"Time at power × cadence · {lab_a}",
                                                overlay=co.mean_y_by_power(cad_a, y_bin=5)),
                             use_container_width=True)
            cc2.plotly_chart(plot_cadence_by_zone(co.cadence_by_zone(cad_a, ftp)), use_container_width=True)

            if quads:
                st.plotly_chart(plot_quadrants(co.aggregate(cad_a), ftp, ref_cad, crank_mm / 1000),
                                use_container_width=True)
                rows = [{"Quadrant": q, "Meaning": co.QUADRANT_NAMES[q], lab_a: f"{quads[q]:.0%}"} for q in quads]
                if quads_b:
                    for r in rows:
                        r[lab_b] = f"{quads_b[r['Quadrant']]:.0%}"
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                st.caption(f"Lines at your median cadence ({ref_cad:.0f} rpm) and at the pedal force that gives "
                           f"your FTP ({ftp} W) at that cadence; the red curve is FTP. QII-heavy riding builds "
                           "muscular force (climbing, low-cadence work); QIV is fast, light spinning.")

            cad_all = pd.concat([x for x in [cad_b, cad_a] if x is not None and not x.empty])
            weekly_cad = co.weekly(cad_all, lambda g: co.weighted_median_y(g) if g['seconds'].sum() >= 600 else None)
            if len(weekly_cad) >= 2:
                st.plotly_chart(plot_trend({"Median cadence": weekly_cad}, "rpm", "Weekly median cadence"),
                                use_container_width=True)

    # ---- Recovery & efficiency ----
    if section == sections[2]:
        in_a = (stats['date'] >= a0) & (stats['date'] < a1 + pd.Timedelta(days=1)) if not stats.empty else None
        hrr = stats.dropna(subset=['hrr60_bpm']) if not stats.empty else stats
        endu = co.endurance_rides(stats)

        m1, m2, m3 = st.columns(3)
        def med(df, col, mask_a=True):
            if df is None or df.empty:
                return None
            sel = df[col][in_a.reindex(df.index, fill_value=False)] if mask_a else \
                  df[col][~in_a.reindex(df.index, fill_value=False)]
            sel = sel.dropna()
            return float(sel.median()) if len(sel) else None
        hrr_a, hrr_b = med(hrr, 'hrr60_bpm'), (med(hrr, 'hrr60_bpm', False) if compare else None)
        ef_a, ef_b = med(endu, 'ef'), (med(endu, 'ef', False) if compare else None)
        dec_a, dec_b = med(endu, 'decoupling_pct'), (med(endu, 'decoupling_pct', False) if compare else None)
        m1.metric("HR recovery (60 s)", f"{hrr_a:.0f} bpm" if hrr_a else "N/A",
                  delta=_fmt_delta(hrr_a, hrr_b, "{:+.0f} bpm"),
                  help="Median heart-rate drop in the minute after hard efforts (≥ 1 min above 105 % FTP "
                       "followed by easy riding). Faster recovery generally reflects fitness; a sudden drop "
                       "can be a sign of accumulated fatigue.")
        m2.metric("EF on endurance rides", f"{ef_a:.2f}" if ef_a else "N/A",
                  delta=_fmt_delta(ef_a, ef_b, "{:+.2f}"),
                  help="Median NP / HR on rides of 60+ min with VI ≤ 1.10, so only comparable steady rides count.")
        m3.metric("Decoupling on endurance rides", f"{dec_a:.1f} %" if dec_a is not None else "N/A",
                  delta=_fmt_delta(dec_a, dec_b, "{:+.1f} pp"), delta_color="inverse",
                  help="Median power:HR drift on the same rides. Below ~5 % suggests solid aerobic endurance.")

        if hrr is not None and not hrr.empty:
            st.plotly_chart(plot_trend({"HRR60": hrr.set_index('date')['hrr60_bpm']}, "bpm drop in 60 s",
                                       "Heart-rate recovery after hard efforts", rolling=5),
                            use_container_width=True)
        else:
            st.info("No hard efforts followed by easy riding found (needs power + HR).")
        if endu is not None and not endu.empty:
            st.plotly_chart(plot_trend({"EF": endu.set_index('date')['ef']}, "NP / HR",
                                       "Efficiency factor on endurance rides", rolling=5),
                            use_container_width=True)
        else:
            st.info("No steady endurance rides (60+ min, VI ≤ 1.10, power + HR) in this period.")


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
    
    pmc_df = compute_pmc(history_data)
    if not history_data:
        st.info("No TSS history data available. Upload a ride to see your chart!")
    elif pmc_df.empty:
        st.warning("Found rides, but none with a valid date and TSS. "
                   "Use **Update ride history** below to rebuild them from the stored files.")
    else:
        st.plotly_chart(plot_pmc(pmc_df), use_container_width=True)

        # Current status (the PMC runs up to today, so this is "now")
        curr = pmc_df.iloc[-1]
        k1, k2, k3 = st.columns(3)
        k1.metric("Fitness (CTL)", f"{curr['CTL']:.1f}", help="Chronic Training Load (42-day time constant), as of today")
        k2.metric("Fatigue (ATL)", f"{curr['ATL']:.1f}", help="Acute Training Load (7-day time constant), as of today")
        k3.metric("Form (TSB)", f"{curr['TSB']:.1f}",
                  help="Training Stress Balance: yesterday's CTL - ATL (form going into today)")

        with st.expander("View Raw PMC Data"):
            st.dataframe(pmc_df)

    _render_history_update(user_name, user_data)

    st.markdown("---")

    _render_deeper_analytics(user_name, user_data)

    st.markdown("---")

    # 3. Best Power Records
    best_power = get_user_best_power(user_name)
    if best_power:

        st.subheader("🏆 User All-Time Power Curve")

        show_centered(plot_power_curve(best_power, None))


        cp_results = estimate_critical_power(best_power)
        if cp_results:
            st.markdown("### ⚡ Physiological Model (Critical Power)")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Critical Power (CP)", 
                    f"{cp_results['cp']:.0f} W",
                    help="The theoretical power you can sustain 'indefinitely' (aerobic ceiling). Usually slightly higher than FTP."
                )
            
            with col2:
                w_prime_kj = cp_results['w_prime'] / 1000
                st.metric(
                    "W' (Anaerobic Capacity)", 
                    f"{w_prime_kj:.1f} kJ",
                    help="Battery size above CP. Typical values: 10kJ (Sprinter) to 25kJ (Pursuiter)."
                )
                
            with col3:
                st.metric(
                    "Model Fit (R²)", 
                    f"{cp_results['r_squared']:.3f}",
                    help="How well your data fits the model. >0.95 is excellent. If low, you might need more all-out efforts between 3 and 12 mins."
                )
                
            # Suggestion logic based on W'
            if cp_results['w_prime'] > 20000:
                st.caption("💡 Insight: You have a high anaerobic capacity. You are likely a **Puncheur** or **Sprinter**.")
            elif cp_results['w_prime'] < 12000:
                st.caption("💡 Insight: Your engine is more diesel. You are likely a **Time Trialist** or **Climber** (or you need to do more intervals!).")
            
            st.markdown("---")
            st.caption("👇 **Action:** Update your profile automatically with these calculated values.")
            
            # Layout for buttons
            b_col1, b_col2 = st.columns([1, 2])
            
            with b_col1:
                # OPTION 1: Update Physics Only
                if st.button("💾 Save CP & W'"):
                    # Import the function inside to avoid circular imports if necessary
                    from src.data.db import update_user_physics
                    
                    update_user_physics(
                        user_name, 
                        cp=cp_results['cp'], 
                        w_prime=cp_results['w_prime']
                    )
                    st.success(f"Updated {user_name}: CP={cp_results['cp']:.0f}W, W'={cp_results['w_prime']/1000:.1f}kJ")
                    st.rerun() # Refresh page to show new settings

            with b_col2:
                # OPTION 2: Update FTP too
                # CP is effectively your "Aerobic Ceiling". Using it as FTP is scientifically sound for TSS.
                if st.button("💾 Save CP & W' AND set FTP = CP"):
                    from src.data.db import update_user_physics
                    
                    update_user_physics(
                        user_name, 
                        cp=cp_results['cp'], 
                        w_prime=cp_results['w_prime'],
                        update_ftp=True
                    )
                    st.success(f"Full Update! FTP is now {cp_results['cp']:.0f}W.")
                    st.rerun()
        else:
            st.info("Not enough data to model Critical Power. Upload maximal efforts between 3 and 20 minutes.")            

        st.subheader("🏆 Best Power Records")
        data = []
        for duration, watts in sorted(best_power.items()):
            data.append({
                "Duration": duration_label(duration), 
                "Watts": int(watts), 
                "W/kg": round(watts / user_data['weight'], 2)
            })
        
        st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)  


