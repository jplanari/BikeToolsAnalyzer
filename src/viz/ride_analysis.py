"""Analyze Upload page: parse a ride, compute its metrics and draw the analysis."""
import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from streamlit_folium import st_folium
import plotly.graph_objects as go
from src.data.gpx import calculate_bearing
from src.data.fit import get_fit_laps
from src.data.pipeline import prepare_ride, summarize_ride, ride_kind, metrics_for_storage
from src.viz.plots import (plot_elevation,
                           plot_x_time,
                           plot_power_curve,
                           plot_zone_distribution,
                           plot_climbs,
                           plot_detailed_climb,
                           plot_power_budget,
                           plot_w_prime_balance)
from src.analysis.power import power_curve, time_in_zones, coggan_zones, calculate_w_prime_balance
from src.analysis.hr import estimate_hr_threshold, time_in_hr_zones
from src.analysis import power_profile as pp
from src.data.db import save_ride, build_ride_metadata
from src.data.cached import get_user_best_power, get_power_records
from src.analysis.climbs import detect_climbs
from src.viz.maps import create_route_map
from src.physics.aerodyn import calculate_CdA, estimate_cda, calculate_power_components
from src.data.weather import fetch_ride_weather
from src.physics.thermal import calculate_thermal_profile, CLOTHING_CATALOG
from src.viz.formatting import format_duration, show_centered
from src.viz.climb_estimator import render_climb_estimator


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
    
    fixed = summary.get('cleaned') or {}
    if sum(fixed.values()):
        parts = [f"{n} {name}" for name, n in (("power", fixed.get('power', 0)),
                 ("heart-rate", fixed.get('hr', 0)), ("cadence", fixed.get('cadence', 0))) if n]
        st.caption("🧹 Removed impossible sensor readings before scoring: " + ", ".join(parts) + " samples.")

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
        _render_thermal_analysis(df, settings['weight'], settings['height'], settings.get('ftp'))    

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

    st.subheader("Route Map")
    map_obj = create_route_map(df, detected_climbs)
    if map_obj: st_folium(map_obj, width=1000, height=500)
    else: st.warning("No GPS coordinates found.")

    st.subheader("Elevation Profile")
    show_centered(plot_elevation(df))

    # --- CdA SECTION ---
    if 'power' in df.columns and 'speed' in df.columns:
        st.subheader("💨 Aerodynamic Analysis & Power Budget")
        
        # Calculation happens automatically here using available columns
        df['cda'], df['p_aero'] = calculate_CdA(df, rider_mass=settings.get('weight', 70))
        fit = estimate_cda(df, rider_mass=settings.get('weight', 70))
        avg_p_aero = df['p_aero'].mean() if 'p_aero' in df.columns else None
        df.attrs['cda_est'] = fit['cda'] if fit else None          # used by the climb estimator
        if fit:
            c1, c2 = st.columns(2)
            c1.metric("Est. CdA", f"{fit['cda']:.3f} m²",
                      help=f"Least-squares fit over {fit['seconds'] // 60} min faster than 18 km/h. Treat it as "
                           "±0.03 m²: it absorbs any error in weight, rolling resistance, wind and gradient.")
            c2.metric("Avg Aero Power", f"{int(avg_p_aero)} W" if pd.notna(avg_p_aero) else "N/A")
        else:
            st.warning("Not enough stable data points (Speed > 18km/h required).")

        st.markdown("**Power Budget: Where did your watts go?**")
        with st.spinner("Calculating power components..."):
            df['p_grav'], df['p_roll'], df['p_accel']  = p_comps
            if not df.empty:
                fig_budget = plot_power_budget(df)
                show_centered(fig_budget)
                
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

    if 'power' in df.columns:
        st.subheader("Power Curve")
        best_curve, recent_curve = None, None
        if user_name:
            best_curve = get_user_best_power(user_name)
            today = pd.Timestamp.today().normalize()
            recent_curve = pp.as_dict(pp.best_curve(
                get_power_records(user_name, today - pd.Timedelta(days=89), today)))

        show_centered(plot_power_curve(power_curve(df['power']), best_curve, recent_curve))

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

    _render_climb_details(df, detected_climbs, settings)


def _render_climb_details(df, climbs, settings):
    ride_id = f"{len(df)}_{df['time'].iloc[0]}" if 'time' in df.columns and len(df) else "0"
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
                    render_climb_estimator(df, c, i, settings, ride_id)
                
                st.markdown("---")
    else:
        st.info("No significant climbs detected on this route.")


def _render_thermal_analysis(df, user_weight, user_height, ftp=None):
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
                    clothing_items=selected_clothing,
                    ftp=ftp
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
