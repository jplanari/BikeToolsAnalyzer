# src/graphical.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

def plot_power_time(df, outpath=None):
    # Using rolling average for smoother power graph
    x = df['power'].rolling(window=15, min_periods=1).mean()
    fig, ax1 = plt.subplots(figsize=(10,6))
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Power (W)', color='green')
    ax1.plot(df['time'], x, color='green', linewidth=1, alpha=0.8)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Elevation (m)', color='gray')
    ax2.plot(df['time'], df['ele'], color='gray', linewidth=0.5, alpha=0.3)
    ax2.fill_between(df['time'], df['ele'], color='gray', alpha=0.1)
    
    plt.title('Power vs Time')
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
        plt.close()
    return fig

def plot_power_curve(pcurve_dict, outpath=None):
    durations = sorted(pcurve_dict.keys())
    values = [pcurve_dict[d] for d in durations]
    
    # Convert duration to minutes for labels if needed, or keep seconds log scale
    fig = plt.figure(figsize=(10,6))
    plt.semilogx(durations, values, marker='o', linestyle='-', color='firebrick')
    
    # Set specific ticks
    ticks = [5, 60, 300, 1200, 3600]
    labels = ["5s", "1m", "5m", "20m", "60m"]
    plt.xticks(ticks, labels)
    
    plt.grid(True, which="both", ls="--")
    plt.title("Power Duration Curve")
    plt.ylabel("Watts")
    plt.xlabel("Duration (Log Scale)")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
        plt.close()
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

def plot_detailed_climb(df, start_idx, end_idx, segments):
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
    
    for seg in segments:
        sub_df = df.loc[seg['segment_idx_start']:seg['segment_idx_end']]
        
        seg_x = (sub_df['dist_m'] - base_dist) / 1000
        seg_y = sub_df['ele']
        
        color = get_gradient_color(seg['avg_grad'])
        
        ax.fill_between(seg_x, seg_y, min_ele - 10, color=color, alpha=0.8)
        
        # Add text label
        mid_x = (seg['start_dist'] + seg['end_dist']) / 2
        mid_x_norm = (mid_x - base_dist) / 1000
        mid_y = (seg['start_ele'] + seg['end_ele']) / 2
        
        ax.text(mid_x_norm, mid_y + (max_ele-min_ele)*0.05, f"{seg['avg_grad']:.1f}%", 
                ha='center', fontsize=9, fontweight='bold', color='black', rotation=45)

    ax.set_title(f"Climb Detail (Length: {(x_axis.iloc[-1]):.2f} km)")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Elevation (m)")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
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
