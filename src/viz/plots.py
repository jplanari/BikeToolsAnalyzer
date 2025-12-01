# src/graphical.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

def plot_elevation(df, outpath=None):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(df['dist_m']/1000, df['ele'], color='blue', linewidth=2)
    plt.fill_between(df['dist_m']/1000, df['ele'], color='lightblue', alpha=0.5)
    plt.title('Elevation Profile')
    plt.xlabel('Distance (km)')
    plt.ylabel('Elevation (m)')
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
        plt.close()
    return fig

def plot_x_time(df, col, ylabel, outpath=None):
    x = df[col].rolling(window=100, min_periods=1).mean()
    fig, ax1 = plt.subplots(figsize=(10,6))

    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel(ylabel, color='blue')
    ax1.plot(df['dist_m']/1000, x, color='blue', linewidth=1, alpha=0.8)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Elevation (m)', color='gray')
    ax2.plot(df['dist_m']/1000, df['ele'], color='gray', linewidth=0.5, alpha=0.3)
    ax2.fill_between(df['dist_m']/1000, df['ele'], color='gray', alpha=0.1)

    plt.title(f'{ylabel}')
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
        plt.close()
    return fig

def plot_power_curve(current_curve, best_curve=None):
    """
    Plots the power curve. 
    current_curve: dict {duration_sec: watts}
    best_curve: dict {duration_sec: watts} (Optional, for comparison)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare Data
    durations = sorted(current_curve.keys())
    # Filter only durations that exist in current_curve and are valid numbers
    x_labels = [d for d in durations if pd.notna(current_curve[d])]
    y_values = [current_curve[d] for d in x_labels]

    # Create X-axis points (equidistant for readability, not linear time scale)
    x_pos = np.arange(len(x_labels))

    # 1. Plot User All-Time Best (Background)
    if best_curve:
        # Ensure we match the x_labels of the current view
        y_best = [best_curve.get(d, 0) for d in x_labels]
        ax.plot(x_pos, y_best, color='lightgray', linestyle='--', linewidth=2, label="All-Time Best", marker='o', alpha=0.7)
        ax.fill_between(x_pos, y_best, 0, color='lightgray', alpha=0.1)

    # 2. Plot Current Ride (Foreground)
    ax.plot(x_pos, y_values, marker='o', linestyle='-', color='#FF4B4B', linewidth=2, label="This Ride")
    ax.fill_between(x_pos, y_values, 0, color='#FF4B4B', alpha=0.1)

    # Formatting
    ax.set_xticks(x_pos)
    
    # Function to format seconds into 5s, 1m, 20m, etc.
    def fmt_dur(s):
        if s < 60: return f"{s}s"
        elif s < 3600: return f"{int(s/60)}m"
        else: return f"{int(s/3600)}h"

    ax.set_xticklabels([fmt_dur(x) for x in x_labels])
    
    ax.set_title("Power Duration Curve")
    ax.set_ylabel("Power (Watts)")
    ax.set_xlabel("Duration")
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend()

    return fig

def plot_zone_distribution(zone_times, title, outpath=None):
    zones = list(zone_times.keys())
    sum_times = sum(zone_times.values())
    
    # --- FIX: Handle empty data ---
    if sum_times == 0:
        return None
    
    times = [zone_times[z] / sum_times for z in zones]  # Convert to percentages
    max_time = max(times)
    fig = plt.figure(figsize=(10,6))
    bars = plt.bar(zones, times, color='purple', alpha=0.7)
    plt.title(title)
    plt.xlabel('Zones')
    plt.ylim(0, max_time * 1.3)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2%}', ha='center', va='bottom')
    if outpath:
        plt.savefig(outpath)
        plt.close()

    return fig

def plot_climbs(df, climbs, outpath=None):
    fig, ax = plt.subplots(figsize=(10,6))
    # Plot full profile background
    ax.plot(df['dist_m']/1000, df['ele'], color='gray', alpha=0.3, linewidth=1)
    ax.fill_between(df['dist_m']/1000, df['ele'], color='gray', alpha=0.1)
    
    # Highlight climbs
    for i, c in enumerate(climbs):
        sub = df.loc[c['start_idx']:c['end_idx']]
        ax.plot(sub['dist_m']/1000, sub['ele'], color='red', linewidth=2)
        
        # Label
        mid_x = sub['dist_m'].mean()/1000
        max_y = sub['ele'].max()
        ax.text(mid_x, max_y + 10, f"#{i+1}", color='red', fontweight='bold', ha='center')

    ax.set_title("Detected Climbs")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Elevation (m)")
    plt.tight_layout()
    
    if outpath:
        plt.savefig(outpath)
        plt.close()
    return fig

# --- NEW DETAILED CLIMB PLOTTING ---

def get_gradient_color(gradient):
    if gradient < 3: return "#90EE90"  # Light Green
    if gradient < 6: return "#FFD700"  # Gold
    if gradient < 9: return "#FFA500"  # Orange
    if gradient < 12: return "#FF4500" # Red
    return "#8B0000"                   # Dark Red/Black

def plot_detailed_climb(df, start_idx, end_idx, segments, rider_weight=70):
    """
    Plots a single climb with colored segments based on gradient.
    """
    climb_df = df.loc[start_idx:end_idx]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Normalize distance to start at 0km
    base_dist = climb_df['dist_m'].iloc[0]
    x_axis = (climb_df['dist_m'] - base_dist) / 1000
    
    ax.plot(x_axis, climb_df['ele'], color='black', linewidth=1, alpha=0.5)
    
    max_ele = climb_df['ele'].max()
    min_ele = climb_df['ele'].min()

    ele_range = max_ele - min_ele
    
    for seg in segments:
        mask = (climb_df['dist_m'] >= seg['start_dist']) & (climb_df['dist_m'] <= seg['end_dist'])
        sub_df = climb_df.loc[mask]

        if sub_df.empty:
            continue

        # --- Calculations ---

        t_start = sub_df['time'].iloc[0]
        t_end = sub_df['time'].iloc[-1]
        duration = (t_end - t_start).total_seconds()
        d_ele = seg['end_ele'] - seg['start_ele']

        vam=0
        if duration > 1:
            vam = (d_ele / duration) * 3600

        avg_kph = 0.0
        distance_m = seg['end_dist'] - seg['start_dist']
        if duration > 0:
            avg_kph = (distance_m / duration) * 3.6

        avg_watts = 0
        if 'power' in sub_df.columns:
            avg_watts = sub_df['power'].mean()

        wkg = 0.0
        if rider_weight > 0 and pd.notna(avg_watts):
            wkg = avg_watts / rider_weight
        
        seg_x = (sub_df['dist_m'] - base_dist) / 1000
        seg_y = sub_df['ele']
        
        color = get_gradient_color(seg['avg_grad'])
        
        ax.fill_between(seg_x, seg_y, min_ele - 10, color=color, alpha=0.8)
        
        # Add text label
        mid_x = (seg['start_dist'] + seg['end_dist']) / 2
        mid_x_norm = (mid_x - base_dist) / 1000
        mid_y = (seg['start_ele'] + seg['end_ele']) / 2
        
        label_txt = f"{seg['avg_grad']:.1f}%"
        if vam > 0:
            label_txt += f"\n{int(vam)} VAM"
        if wkg > 0:
            label_txt += f"\n{wkg:.1f} W/kg"
        if avg_kph > 0:
            label_txt += f"\n{avg_kph:.1f} km/h"

        seg_len_km = (seg['end_dist'] - seg['start_dist']) / 1000
        rotation = 90 if seg_len_km < 0.5 else 0

        ax.text(mid_x_norm, mid_y + (ele_range * 0.1), label_txt, ha='center', va='bottom', fontsize=8, fontweight='bold', color='black', rotation=rotation)

    ax.set_title(f"Climb Detail (Length: {(x_axis.iloc[-1]):.2f} km)")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Elevation (m)")
    ax.grid(True, alpha=0.2)
    
    # Create Legend
    patches = [
        mpatches.Patch(color="#90EE90", label="< 3%"),
        mpatches.Patch(color="#FFD700", label="3-6%"),
        mpatches.Patch(color="#FFA500", label="6-9%"),
        mpatches.Patch(color="#FF4500", label="9-12%"),
        mpatches.Patch(color="#8B0000", label="> 12%")
    ]
    ax.legend(handles=patches, loc='upper left', fontsize='small')

    plt.tight_layout()
    return fig

def plot_power_budget(df):
    """
    Plots the breakdown of power components.
    comps_df: DataFrame with columns p_aero, p_grav, p_roll, p_accel, p_loss
    """
    comps_df = df[['p_aero', 'p_grav', 'p_roll', 'p_accel','ele','dist_m']].copy()
    
    # --- FIX: Ensure DatetimeIndex ---
    # If comps_df doesn't have a DatetimeIndex, try to recover it from the original df
    if not isinstance(comps_df.index, pd.DatetimeIndex):
        if 'time' in df.columns:
            comps_df = comps_df.set_index(pd.to_datetime(df['time']))
        elif isinstance(df.index, pd.DatetimeIndex):
             comps_df.index = df.index
        else:
            # Fallback: Create a dummy index if real time is missing (start at 0, 1s steps)
            start = pd.Timestamp.now()
            comps_df.index = pd.date_range(start=start, periods=len(comps_df), freq='1s')
    # 1. Resample and Fill NaNs
    # We use .mean() to smooth, then .fillna(0) to ensure stackplot doesn't break on gaps
    resampled = comps_df.resample('30s').mean().fillna(0)
    
    # 2. Setup Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    x = resampled['dist_m']/1000
    
    # 3. Prepare Data Components (Absolute)
    # Ensure no negative values for the stacked chart logic (though gravity/accel can be negative)
    # For a "Budget" chart, we typically visualize the *Cost* (positive loads).
    y_aero = resampled['p_aero'].clip(lower=0) 
    y_roll = resampled['p_roll'].clip(lower=0)
    y_grav = resampled['p_grav'].clip(lower=0) 
    y_accel = resampled['p_accel'].clip(lower=0)
    
    labels = ["Rolling Res.", "Acceleration", "Gravity (Climb)", "Aerodynamic"]
    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#1f77b4"] 
    
    # 4. Plot Absolute (Stackplot)
    ax1.stackplot(x, y_roll, y_accel, y_grav, y_aero, labels=labels, colors=colors, alpha=0.8)
    ax1.set_ylabel("Power (W)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax3 = ax1.twinx()
    ax3.plot(x, resampled['ele'], linewidth=2.5, color='black')
    ax3.set_ylabel("Elevation (m)")

    # 5. Prepare Relative Data (Percentage)
    total_cost = y_roll + y_accel + y_grav + y_aero
    
    # Avoid division by zero: only calculate ratio where total power is significant (>10W)
    mask = total_cost > 10
    
    # Helper to safe-divide
    def get_pct(series):
        return (series / total_cost).where(mask, 0) * 100

    y_roll_pct = get_pct(y_roll)
    y_accel_pct = get_pct(y_accel)
    y_grav_pct = get_pct(y_grav)
    y_aero_pct = get_pct(y_aero)
    
    # 6. Plot Relative (Stackplot)
    ax2.stackplot(x, y_roll_pct, y_accel_pct, y_grav_pct, y_aero_pct, labels=labels, colors=colors, alpha=0.8)
    
    ax2.set_ylabel("Ratio of Total Power (%)")
    ax2.set_xlabel("Distance (km)")
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
   
    ax4 = ax2.twinx()
    ax4.plot(x, resampled['ele'], linewidth=2.5, color='black')
    ax4.set_ylabel("Elevation (m)")

    
    plt.tight_layout()
    return fig

def plot_w_prime_balance(df, w_val_series, w_prime_cap):
    fig, ax = plt.subplots(figsize=(10,5))

    #x = df.index if isinstance(df.index, pd.Datetimeindex) else df['time']
    x = df['dist_m'] / 1000.0  # Distance in km
    y = w_val_series / 1000.0

    cap_kj = w_prime_cap / 1000.0

    ax.plot(x, y, color='red' , linewidth=1.5, label="W' Balance (kJ)")
    ax.axhline(cap_kj, color='black', linestyle='--', linewidth=1, label="W' Capacity (kJ)")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("W' Balance (kJ)")
    ax.set_title("Anaerobic Battery (W' Balance)")
    ax.set_ylim(0, cap_kj * 1.1)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(x, df['ele'], color='gray', linewidth=0.5, alpha=0.5)
    ax2.fill_between(x, df['ele'], color='gray', alpha=0.1)
    ax2.set_ylabel("Elevation (m)")

    return fig

def plot_pmc(pmc_df):
    fig, ax1 = plt.subplots(figsize=(10,6))

    col_ctl = '#1f77b4'  # Blue
    col_atl = '#e377c2'  # Pink
    col_tsb_pos = '#ff7f0e'  # Yellow
    col_tsb_neg = '#d62728'  # Red

    x = pmc_df.index

    ax2 = ax1.twinx()

    tsb_colors = [col_tsb_pos if v >= 0 else col_tsb_neg for v in pmc_df['TSB']]
    ax2.bar(x, pmc_df['TSB'], color=tsb_colors, alpha=0.3, width=1, label='TSB (Form)')
    ax2.set_ylabel("TSB (Form)", color='gray')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)

    ax1.plot(x, pmc_df['CTL'], color=col_ctl, linewidth=2, label='CTL (Fitness)')
    ax1.fill_between(x, pmc_df['CTL'], 0, color=col_ctl, alpha=0.1)
    ax1.plot(x, pmc_df['ATL'], color=col_atl, linewidth=2, label='ATL (Fatigue)')

    ax1.set_title("Performance Management Chart (PMC)")
    ax1.set_ylabel("Training Load")
    ax1.grid(True, alpha=0.2)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.tight_layout()

    return fig

