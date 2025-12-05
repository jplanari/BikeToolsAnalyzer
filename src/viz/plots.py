# src/graphical.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def _add_elevation_background(fig, df, secondary_y=True):
    """Helper to add elevation profile in the background of other charts."""
    if 'ele' in df.columns and 'dist_m' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['dist_m'] / 1000,
                y=df['ele'],
                name="Elevation",
                line=dict(color='gray', width=1),
                fill='tozeroy',
                opacity=0.2,
                hoverinfo='skip' # Don't clutter hover tooltip
            ),
            secondary_y=secondary_y
        )

def plot_elevation(df):
    fig = go.Figure()
    
    # Main Elevation Line
    fig.add_trace(go.Scatter(
        x=df['dist_m'] / 1000, 
        y=df['ele'],
        mode='lines',
        name='Elevation',
        fill='tozeroy',
        line=dict(color='#1f77b4', width=2)
    ))

    fig.update_layout(
        title="Elevation Profile",
        xaxis_title="Distance (km)",
        yaxis_title="Elevation (m)",
        hovermode="x unified",
        template="plotly_white",
        height=400
    )
    return fig

def plot_x_time(df, col, ylabel, color='blue', title=None):
    """
    Plots a metric (Power, HR, Speed) vs Distance, with Elevation in background.
    """
    from plotly.subplots import make_subplots
    
    # Create Dual-Axis Plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 1. Add Background Elevation (Right Axis)
    _add_elevation_background(fig, df, secondary_y=True)
    
    # 2. Add Smoothed Metric (Left Axis)
    # Smooth the data for readability (10s rolling avg)
    y_smooth = df[col].rolling(window=10, min_periods=1).mean()
    
    fig.add_trace(
        go.Scatter(
            x=df['dist_m'] / 1000,
            y=y_smooth,
            name=ylabel,
            line=dict(color=color, width=1.5)
        ),
        secondary_y=False
    )

    fig.update_layout(
        title=title or f"{ylabel} vs Distance",
        xaxis_title="Distance (km)",
        yaxis_title=ylabel,
        yaxis2_title="Elevation (m)",
        hovermode="x unified",
        template="plotly_white",
        showlegend=False,
        height=400
    )
    return fig

def plot_power_curve(power_curve_dict, best_curve_dict=None):
    fig = go.Figure()
    
    # 1. Current Ride Curve
    durations = sorted(power_curve_dict.keys())
    watts = [power_curve_dict[d] for d in durations]
    
    fig.add_trace(go.Scatter(
        x=durations, 
        y=watts,
        mode='lines+markers',
        name='This Ride',
        line=dict(color='blue', width=3)
    ))
    
    # 2. All-Time Best Curve (Ghost Line)
    if best_curve_dict:
        # Align durations
        b_durations = sorted(best_curve_dict.keys())
        b_watts = [best_curve_dict[d] for d in b_durations]
        
        fig.add_trace(go.Scatter(
            x=b_durations,
            y=b_watts,
            mode='lines',
            name='All-Time Best',
            line=dict(color='gray', width=2, dash='dash'),
            opacity=0.7
        ))

    # Log Scale for Time (Standard for Power Curves)
    fig.update_layout(
        title="Mean Maximal Power Curve",
        xaxis_title="Duration (seconds)",
        yaxis_title="Watts",
        xaxis_type="log",
        hovermode="x unified",
        template="plotly_white",
        xaxis=dict(
            tickvals=[5, 60, 300, 1200, 3600],
            ticktext=["5s", "1m", "5m", "20m", "1h"]
        )
    )
    return fig

def plot_climbs(df, climbs):
    """
    Plots the full elevation profile with climbs highlighted in Red.
    """
    fig = go.Figure()

    # 1. Full Elevation (Gray Background)
    fig.add_trace(go.Scatter(
        x=df['dist_m'] / 1000,
        y=df['ele'],
        mode='lines',
        name='Full Route',
        line=dict(color='lightgray', width=2),
        fill='tozeroy',
        fillcolor='rgba(200,200,200,0.2)',
        hoverinfo='skip'
    ))
    
    y_min = df['ele'].min() * 0.95
    y_max = df['ele'].max() * 1.05

   # 2. Highlight Each Climb
    for i, c in enumerate(climbs):
        segment = df.loc[c['start_idx']:c['end_idx']]
        if segment.empty: continue
        
        # Hover text for the climb segment
        hover_text = (
            f"Climb #{i+1} - {c['type']}<br>"
            f"Len: {c['length_m']/1000:.1f}km, Avg: {c['avg_gradient_pct']:.1f}%"
        )
        
        fig.add_trace(go.Scatter(
            x=segment['dist_m'] / 1000,
            y=segment['ele'],
            mode='lines',
            name=f"Climb #{i+1}",
            line=dict(color='#d62728', width=3), # Red
            hoveron='fills+points',
            text=hover_text,
            hoverinfo='text+y'
        ))

    fig.update_layout(
        title="Route Elevation & Climbs Detected",
        xaxis_title="Distance (km)",
        yaxis_title="Elevation (m)",
        template="plotly_white",
        showlegend=False,
        height=300
    )
        # Apply calculated limits
    fig.update_yaxes(
        range=[y_min, y_max]
    )

    return fig

# --- NEW DETAILED CLIMB PLOTTING ---

def plot_detailed_climb(segment):
    """
    Classic 'Grand Tour' style climb profile.
    - Visuals: Continuous Elevation Profile (Filled Area).
    - Coloring: The area is split into segments, each colored by its avg gradient.
    - Annotations: Gradient % aligned to the bottom of the plot.
    """
    if segment.empty:
        return go.Figure()

    # 1. Setup Data
    start_dist = segment['dist_m'].iloc[0]
    df = segment.copy()
    df['local_dist_m'] = df['dist_m'] - start_dist
    
    # 2. Calculate Global Axis Limits (Pre-calculation)
    min_ele = df['ele'].min()
    max_ele = df['ele'].max()
    
    # Y-Axis range: 95% of min to 105% of max
    # We use this y_min to anchor the text at the bottom
    y_min = min_ele * 0.95
    y_max = max_ele * 1.05

    # Initialize Figure
    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 3. Iterate through segments
    max_dist = df['local_dist_m'].max()

    if max_dist < 1000:
        segment_length_m = 200
    elif max_dist < 7000:
        segment_length_m = 500
    elif max_dist < 15000:
        segment_length_m = 1000
    else:
        segment_length_m = 2000
    
    for start in np.arange(0, max_dist, segment_length_m):
        end = start + segment_length_m
        
        # Get the chunk
        chunk = df[(df['local_dist_m'] >= start) & (df['local_dist_m'] <= end)]
        
        if chunk.empty:
            continue
            
        # Connect to previous point
        if start > 0:
            prev_point = df[df['local_dist_m'] <= start].iloc[-1]
            chunk = pd.concat([prev_point.to_frame().T, chunk])

        # Calculate Avg Gradient
        dist_delta = chunk['dist_m'].max() - chunk['dist_m'].min()
        ele_delta = chunk['ele'].max() - chunk['ele'].min()
        
        avg_grade = 0
        if dist_delta > 0:
            avg_grade = (ele_delta / dist_delta) * 100

        # Determine Color
        fill_color = '#aaaaaa' # Default Gray
        if avg_grade >= 3:  fill_color = '#2ca02c' # Green
        if avg_grade >= 6:  fill_color = '#ff7f0e' # Orange
        if avg_grade >= 9:  fill_color = '#d62728' # Red
        if avg_grade >= 12: fill_color = '#000000' # Black

        # Add the Segment Trace
        fig.add_trace(
            go.Scatter(
                x=chunk['local_dist_m'] / 1000,
                y=chunk['ele'],
                mode='lines',
                line=dict(color='black', width=1), 
                fill='tozeroy',
                fillcolor=fill_color,
                showlegend=False,
                hovertemplate=f"<b>Seg:</b> {avg_grade:.1f}%<br><b>Elev:</b> %{{y:.0f}}m<extra></extra>"
            ),
            secondary_y=False
        )
        
        # Add Annotation (Aligned to Bottom)
        mid_x = (chunk['local_dist_m'].min() + chunk['local_dist_m'].max()) / 2000 # km
        
        if avg_grade > 3:
            fig.add_annotation(
                x=mid_x, 
                y=y_min, # Anchor to the bottom of the axis
                text=f"{avg_grade:.0f}%",
                showarrow=False,
                yshift=15, # Lift slightly off the bottom spine
                font=dict(color='black', size=11, family="Arial Black")
            )

    # 4. Add Power Overlay
    if 'power' in df.columns:
        p_smooth = df['power'].rolling(10, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=df['local_dist_m'] / 1000,
                y=p_smooth,
                name='Power (10s)',
                mode='lines',
                line=dict(color='purple', width=2),
                opacity=0.6 
            ),
            secondary_y=True
        )

    # 5. Final Layout
    fig.update_layout(
        title=f"Climb Profile",
        xaxis_title="Distance (km)",
        yaxis_title="Elevation (m)",
        yaxis2_title="Power (W)",
        hovermode="x unified",
        template="plotly_white",
        height=400,
        showlegend=False,
        shapes=[] 
    )
    
    # Apply calculated limits
    fig.update_yaxes(
        secondary_y=False, 
        range=[y_min, y_max]
    )
    
    fig.update_yaxes(
        secondary_y=True, 
        range=[0.5*df['power'].min(), df['power'].max() * 1.2], 
        showgrid=False
    )
    
    return fig

def plot_power_budget(df):
    """
    100% Stacked Area chart showing the relative contribution of each 
    power component (Aero, Gravity, Rolling, Accel) to the total power cost.
    """
    from plotly.subplots import make_subplots
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Smoothing window (seconds) to reduce noise
    w = 10
    
    # Prepare data
    # We clip negative values (like descending gravity) because this chart 
    # shows "Where the watts go" (Costs), not "Where they come from".
    roll = df['p_roll'].rolling(w).mean().clip(lower=0)
    grav = df['p_grav'].rolling(w).mean().clip(lower=0)
    aero = df['p_aero'].rolling(w).mean().clip(lower=0)
    accel = df['p_accel'].rolling(w).mean().clip(lower=0)
    
    x = df['dist_m'] / 1000 if 'dist_m' in df.columns else df.index

    # Common style for the stack
    # groupnorm='percent' tells Plotly to normalize the stack to 100 at every X point.
    stack_kw = dict(stackgroup='one', groupnorm='percent', mode='lines', line=dict(width=0))

    # 1. Rolling Resistance (Green)
    fig.add_trace(go.Scatter(
        x=x, y=roll,
        name='Rolling',
        fillcolor='rgba(44, 160, 44, 0.7)', # #2ca02c
        **stack_kw
    ), secondary_y=False)
    
    # 2. Gravity (Orange)
    fig.add_trace(go.Scatter(
        x=x, y=grav,
        name='Gravity',
        fillcolor='rgba(255, 127, 14, 0.7)', # #ff7f0e
        **stack_kw
    ), secondary_y=False)
    
    # 3. Acceleration (Red)
    fig.add_trace(go.Scatter(
        x=x, y=accel,
        name='Acceleration',
        fillcolor='rgba(214, 39, 40, 0.7)', # #d62728
        **stack_kw
    ), secondary_y=False)

    # 4. Aerodynamic (Blue)
    # Often the largest, so we put it on top or bottom. Order matters for visuals.
    fig.add_trace(go.Scatter(
        x=x, y=aero,
        name='Aero',
        fillcolor='rgba(31, 119, 180, 0.7)', # #1f77b4
        **stack_kw
    ), secondary_y=False)

    # Elevation
    fig.add_trace(go.Scatter(
        x=x, y=df['ele'],
        name="Elevation",
        line=dict(color='gray', width=2.5)),
        secondary_y=True
    )

    fig.update_layout(
        title="Power Budget (Relative Cost Breakdown)",
        xaxis_title="Distance (km)",
        yaxis_title="Contribution (%)",
        hovermode="x unified",
        template="plotly_white",
        height=400,
        yaxis=dict(range=[0, 100], fixedrange=True),
        showlegend=True,
        legend=dict(orientation="h", y=1.1)
    )
    
    return fig

def plot_w_prime_balance(df, cap_j):
    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Background Elevation
    _add_elevation_background(fig, df, secondary_y=True)
    
    # W' Balance Line
    fig.add_trace(
        go.Scatter(
            x=df['dist_m'] / 1000,
            y=df['w_prime_balance'] / 1000, # Convert J to kJ
            name="W' Balance (kJ)",
            line=dict(color='red', width=2),
            fill='tozeroy' # Fill helps visualize "tank emptying"
        ),
        secondary_y=False
    )
    
    # Add Capacity Line
    fig.add_hline(y=cap_j/1000, line_dash="dash", line_color="green", annotation_text="Full Capacity")
    fig.add_hline(y=0, line_color="black", annotation_text="Empty!")

    fig.update_layout(
        title="Anaerobic Battery (W' Balance)",
        yaxis_title="Energy Remaining (kJ)",
        hovermode="x unified",
        template="plotly_white"
    )
    return fig

def plot_pmc(pmc_df):
    """
    Performance Management Chart (CTL/ATL/TSB)
    """
    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 1. TSB (Form) as Bars on Right Axis
    # Color logic: Positive = Yellow/Orange, Negative = Red
    colors = np.where(pmc_df['TSB'] >= 0, '#ff7f0e', '#d62728')
    
    fig.add_trace(
        go.Bar(
            x=pmc_df.index, y=pmc_df['TSB'],
            name='TSB (Form)',
            marker_color=colors,
            opacity=0.4
        ),
        secondary_y=True
    )

    # 2. CTL (Fitness) Line
    fig.add_trace(
        go.Scatter(x=pmc_df.index, y=pmc_df['CTL'], name='Fitness (CTL)', line=dict(color='#1f77b4', width=3)),
        secondary_y=False
    )

    # 3. ATL (Fatigue) Line
    fig.add_trace(
        go.Scatter(x=pmc_df.index, y=pmc_df['ATL'], name='Fatigue (ATL)', line=dict(color='#e377c2', width=2)),
        secondary_y=False
    )

    fig.update_layout(
        title="Performance Management Chart (PMC)",
        yaxis_title="Training Load (TSS/day)",
        yaxis2_title="Form (TSB)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", y=1.1)
    )
    return fig

def plot_zone_distribution(zone_counts, title="Zone Distribution"):
    labels = list(zone_counts.keys())
    values = list(zone_counts.values())
    
    # Calculate %
    total = sum(values)
    if total == 0: return go.Figure()
    
    # Custom colors for Z1-Z7
    colors = ['#888888', '#3388ff', '#2ca02c', '#ffbf00', '#ff7f0e', '#d62728', '#9467bd']
    
    fig = go.Bar(
        x=labels, 
        y=values,
        marker_color=colors[:len(labels)],
        text=[f"{v/total:.1%}" for v in values],
        textposition='auto'
    )
    
    layout = go.Layout(
        title=title,
        yaxis_title="Seconds",
        template="plotly_white"
    )
    
    return go.Figure(data=[fig], layout=layout)
