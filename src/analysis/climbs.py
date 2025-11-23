# src/climbs.py
import numpy as np
import pandas as pd

def detect_climbs(df, min_gradient=1, min_length=1000, join_dist=1000):
    """
    Detect climbs in the ride data.
    """
    if df.empty or 'ele' not in df.columns or 'dist_m' not in df.columns:
        return []

    # 1. Smooth elevation slightly to avoid noise
    df['ele_smooth'] = df['ele'].rolling(window=10, min_periods=1, center=True).mean()

    climbs = []
    start_idx = None
    N = len(df)
    i = 0
    
    while i < N - 10:
        curr_dist = df.loc[i, 'dist_m']
        # Look ahead 100m to calculate gradient
        lookahead_mask = df['dist_m'] > (curr_dist + 100)
        if not lookahead_mask.any():
            break

        next_idx = lookahead_mask.idxmax()

        d_dist = df.loc[next_idx, 'dist_m'] - curr_dist
        d_elev = df.loc[next_idx, 'ele_smooth'] - df.loc[i, 'ele_smooth']

        grad = (d_elev / d_dist) * 100 if d_dist > 0 else 0

        if grad >= min_gradient:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                end_idx = i
                # Check if potential climb is long enough
                climb_len = df.loc[end_idx, 'dist_m'] - df.loc[start_idx, 'dist_m']
                if climb_len >= min_length:
                    climbs.append((start_idx, end_idx))
                start_idx = None
        
        # Increment i (skip forward slightly for performance, or just +1)
        i += 5 # mild optimization

    # Close last climb if active
    if start_idx is not None:
        end_idx = N - 1
        if (df.loc[end_idx, 'dist_m'] - df.loc[start_idx, 'dist_m']) >= min_length:
            climbs.append((start_idx, end_idx))

    # 2. Merge close climbs
    if not climbs:
        return []

    merged_climbs = []
    if climbs:
        curr_start, curr_end = climbs[0]
        for next_start, next_end in climbs[1:]:
            dist_between = df.loc[next_start, 'dist_m'] - df.loc[curr_end, 'dist_m']
            if dist_between < join_dist:
                # Merge
                curr_end = next_end
            else:
                merged_climbs.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged_climbs.append((curr_start, curr_end))

    # 3. Calculate Stats
    final_climbs = []
    for s, e in merged_climbs:
        dist = df.loc[e, 'dist_m'] - df.loc[s, 'dist_m']
        elev_gain = df.loc[e, 'ele_smooth'] - df.loc[s, 'ele_smooth']
        avg_grad = (elev_gain / dist) * 100 if dist > 0 else 0
        
        time_diff = (df.loc[e, 'time'] - df.loc[s, 'time']).total_seconds()
        avg_speed = (dist / time_diff) * 3.6 if time_diff > 0 else 0
        vam = (elev_gain / time_diff) * 3600 if time_diff > 0 else 0
        
        avg_pwr = df.loc[s:e, 'power'].mean() if 'power' in df.columns else None
        
        if dist >= min_length and avg_grad >= min_gradient:
            final_climbs.append({
                'start_idx': s,
                'end_idx': e,
                'length_m': dist,
                'elev_gain_m': elev_gain,
                'avg_gradient_pct': avg_grad,
                'avg_speed_kph': avg_speed,
                'vam_mph': vam,
                'avg_power': avg_pwr
            })

    return final_climbs

def get_climb_segments(df, start_idx, end_idx):
    """
    Segments a specific climb into 1km or 0.5km chunks and calculates
    stats for each segment.
    """
    climb_df = df.loc[start_idx:end_idx].copy()
    
    if climb_df.empty:
        return []

    start_dist = climb_df['dist_m'].iloc[0]
    total_length = climb_df['dist_m'].iloc[-1] - start_dist
    
    # Determine step size (1000m if > 2km, else 500m)
    step_size = 1000 if total_length > 4000 else 250
    
    segments = []
    current_dist_marker = start_dist
    
    while current_dist_marker < climb_df['dist_m'].iloc[-1]:
        next_dist_marker = current_dist_marker + step_size
        
        # Filter rows between current markers
        mask = (climb_df['dist_m'] >= current_dist_marker) & (climb_df['dist_m'] <= next_dist_marker)
        segment_df = climb_df.loc[mask]
        
        if segment_df.empty:
            current_dist_marker = next_dist_marker
            continue

        s_row = segment_df.iloc[0]
        e_row = segment_df.iloc[-1]
        
        d_dist = e_row['dist_m'] - s_row['dist_m']
        d_ele = e_row['ele_smooth'] - s_row['ele_smooth']
        
        if d_dist > 10: # Avoid division by zero
            grad = (d_ele / d_dist) * 100
            
            segments.append({
                'start_dist': s_row['dist_m'],
                'end_dist': e_row['dist_m'],
                'avg_grad': grad,
                'start_ele': s_row['ele'],
                'end_ele': e_row['ele'],
                'segment_idx_start': segment_df.index[0],
                'segment_idx_end': segment_df.index[-1]
            })
        
        current_dist_marker = next_dist_marker

    return segments
