# src/climbs.py
import numpy as np
import pandas as pd


def detect_climbs(df, min_gradient=1.0, min_length=500, join_dist=1000):
    """
    Detects climbs with a dynamic merge step.
    Merges climbs if the gap is < join_dist AND the elevation loss is < 10% of cumulative gain.
    """
    if df.empty or 'ele' not in df.columns or 'dist_m' not in df.columns:
        return []

    # 1. Smooth elevation
    df['ele_smooth'] = df['ele'].rolling(window=15, min_periods=1, center=True).mean()

    # --- STEP 1: DETECT CANDIDATES (Greedy) ---
    climbs = []
    SEARCH_GRADIENT_THRESHOLD = 1.0  
    
    start_idx = None
    lookahead_dist = 200 
    N = len(df)
    i = 0
    step_size = 10 
    
    while i < N - step_size:
        curr_dist = df.loc[i, 'dist_m']
        
        # Lookahead
        mask = df['dist_m'] > (curr_dist + lookahead_dist)
        if not mask.any():
            break
        
        next_idx = mask.idxmax()
        d_dist = df.loc[next_idx, 'dist_m'] - curr_dist
        d_ele = df.loc[next_idx, 'ele_smooth'] - df.loc[i, 'ele_smooth']
        grad = (d_ele / d_dist) * 100 if d_dist > 0 else 0
        
        if grad >= SEARCH_GRADIENT_THRESHOLD:
            if start_idx is None:
                start_idx = i
            i += step_size
        else:
            if start_idx is not None:
                climbs.append((start_idx, i))
                start_idx = None
            i += step_size
            
    if start_idx is not None:
        climbs.append((start_idx, N-1))

    # --- STEP 2: MERGE CLIMBS (Dynamic Logic) ---
    merged_climbs = []
    
    if climbs:
        # Initialize with the first climb
        curr_start, curr_end = climbs[0]
        
        for i in range(1, len(climbs)):
            next_start, next_end = climbs[i]
            
            # Calculate gap distance
            dist_gap = df.loc[next_start, 'dist_m'] - df.loc[curr_end, 'dist_m']
            
            # Calculate Elevation Drop (Positive means we lost height)
            ele_curr_end = df.loc[curr_end, 'ele_smooth']
            ele_next_start = df.loc[next_start, 'ele_smooth']
            drop = ele_curr_end - ele_next_start
            
            # Calculate Gains
            # Note: Gain of 'curr' uses the *original* start, so it accumulates if we keep merging
            gain_curr = ele_curr_end - df.loc[curr_start, 'ele_smooth']
            gain_next = df.loc[next_end, 'ele_smooth'] - ele_next_start
            
            cumulative_gain = gain_curr + gain_next
            
            # MERGE CONDITION:
            # 1. Distance is close enough
            # 2. Elevation drop is small relative to the size of the climbs (10% rule)
            #    We also merge if drop <= 0 (meaning the gap was actually flat or uphill)
            
            should_merge = False
            if dist_gap < join_dist:
                if drop <= 0:
                    should_merge = True
                elif drop < -(0.1 * cumulative_gain):
                    should_merge = True
            
            if should_merge:
                # Extend current end
                curr_end = next_end
            else:
                # Close current and start a new one
                merged_climbs.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        
        # Append the final holding climb
        merged_climbs.append((curr_start, curr_end))
    else:
        merged_climbs = []

    # --- STEP 3: CATEGORIZATION & FILTERING ---
    final_climbs = []
    
    for start, end in merged_climbs:
        d_start = df.loc[start, 'dist_m']
        d_end = df.loc[end, 'dist_m']
        length_m = d_end - d_start
        
        e_start = df.loc[start, 'ele_smooth']
        e_end = df.loc[end, 'ele_smooth']
        elev_gain = e_end - e_start
        
        if length_m > 0:
            avg_grad = (elev_gain / length_m) * 100
        else:
            avg_grad = 0
            
        t_start = df.loc[start, 'time']
        t_end = df.loc[end, 'time']
        duration_h = (t_end - t_start).total_seconds() / 3600
        vam = elev_gain / duration_h if duration_h > 0 else 0

        # Categorization (Longest -> Shortest)
        category = None
        
        if length_m > 5000 and avg_grad > 3.0:
            category = "Long & Steady"
        elif length_m > 3000 and avg_grad > 4.0:
            category = "Mid & Average"
        elif length_m > 1000 and avg_grad > 5.0:
            category = "Short & Steep"
        elif length_m > 500 and avg_grad > 8.0:
            category = "Punchy"
        elif length_m > 1000 and avg_grad > 2.5:
            category = "Generic Climb"
        
        if category:
            final_climbs.append({
                'start_idx': start,
                'end_idx': end,
                'length_m': length_m,
                'elev_gain_m': elev_gain,
                'avg_gradient_pct': avg_grad,
                'vam_mph': vam,
                'type': category
            })
            
    return final_climbs

def get_climb_segments(df, start_idx, end_idx):
    """
    Splits a specific climb into segments for detailed analysis.
    """
    climb_df = df.iloc[start_idx:end_idx]
    if climb_df.empty: return []

    start_dist = climb_df['dist_m'].iloc[0]
    total_length = climb_df['dist_m'].iloc[-1] - start_dist
    
    step_size = 2000
    if total_length < 10000: step_size = 1000
    if total_length < 2000: step_size = 500 
    if total_length < 1000: step_size = 250 
    
    segments = []
    curr = start_dist
    
    while curr < climb_df['dist_m'].iloc[-1]:
        next_mark = curr + step_size
        mask = (climb_df['dist_m'] >= curr) & (climb_df['dist_m'] < next_mark)
        sub = climb_df.loc[mask]
        
        if not sub.empty:
            s_ele = sub['ele_smooth'].iloc[0]
            e_ele = sub['ele_smooth'].iloc[-1]
            dist = sub['dist_m'].iloc[-1] - sub['dist_m'].iloc[0]
            grad = ((e_ele - s_ele) / dist) * 100 if dist > 0 else 0
            
            segments.append({
                'start_dist': sub['dist_m'].iloc[0],
                'end_dist': sub['dist_m'].iloc[-1],
                'start_ele': s_ele,
                'end_ele': e_ele,
                'avg_grad': grad,
                'segment_idx_start': sub.index[0],
                'segment_idx_end': sub.index[-1]
            })
        curr = next_mark
        
    return segments
