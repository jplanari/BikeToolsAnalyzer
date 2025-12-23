import pandas as pd
import numpy as np
import streamlit as st

# Constants
C_BODY = 3470  # Specific heat capacity [J/(kg·K)]
EFFICIENCY = 0.23

CLOTHING_CATALOG = {
    "None (Shorts only)": 0.05,
    "Mesh Base Layer": 0.08,
    "Summer Jersey": 0.15,
    "Arm/Leg Warmers": 0.10,
    "Gilet (Wind Vest)": 0.15,
    "Long Sleeve Jersey": 0.25,
    "Rain Jacket (Hardshell)": 0.35,
    "Winter Jacket (Softshell)": 0.45,
    "Thermal Bib Tights": 0.20,
    "Buff / Neck Warmer": 0.05
}

def du_bois_BSA(height_cm, weight_kg):
    """Calculate Body Surface Area (BSA) using Du Bois formula."""
    # Height must be in cm for this specific coefficient version usually, 
    # but standard Du Bois is 0.007184 * H^0.725 * W^0.425 where H is cm, W is kg
    return 0.007184 * (height_cm ** 0.725) * (weight_kg ** 0.425)

def R_air_boundary_layer(air_speed):
    """Calculate air boundary layer heat resistance."""
    # Avoid division by zero
    speed = max(0.1, air_speed)
    return 1 / (8.5 * (speed ** 0.84))

def calculate_vapor_pressure_kpa(Ta, rh_fraction):
    """
    Calculate vapor pressure in kPa.
    rh_fraction: 0.0 to 1.0
    """
    # Saturation Vapor Pressure (es) in kPa (Magnus formula)
    es = 0.6112 * np.exp((17.67 * Ta) / (Ta + 243.5))
    return rh_fraction * es

def calculate_thermal_profile(df, rider_weight_kg=80.0, rider_height_cm=175, clothing_items=[], weather_data=None):
    if df.empty:
        return df

    # --- 1. SETUP PARAMETERS ---
    RIDER_MASS = rider_weight_kg
    RIDER_HEIGHT = rider_height_cm
    A = du_bois_BSA(RIDER_HEIGHT, RIDER_MASS)

    # Weather Defaults (Safe Handling)
    # Check if columns exist, otherwise use defaults
    T_AIR = df['temp_c'].fillna(15.0) if 'temp_c' in df.columns else pd.Series([15.0]*len(df), index=df.index)
    
    # Handle Humidity
    if 'relative_humidity' in df.columns:
        # Assuming input is 0-100 or 0-1. Let's assume 0-100 if max > 1
        rh_raw = df['relative_humidity'].fillna(50)
        # Normalize to 0.0 - 1.0
        RH = rh_raw.apply(lambda x: x / 100.0 if x > 1.0 else x)
    else:
        RH = pd.Series([0.5]*len(df), index=df.index)

    base_clo = sum([CLOTHING_CATALOG.get(item, 0) for item in clothing_items])
    base_rsi = base_clo * 0.155 

    t_core = 36.0
    t_skin = 32.0

    core_temps = []
    skin_temps = []
    heat_gen_list = []
    heat_loss_list = []

    # --- 2. VECTORIZED PRE-CALCULATION ---
    
    # Rider Speed (v)
    # Ensure speed exists. If km/h, convert to m/s. Standard FIT/GPX parsing usually gives m/s or km/h.
    # We will assume the dataframe has 'speed' in m/s for physics. 
    # If your parser gives km/h, divide by 3.6 here. 
    # Let's assume input 'speed' is m/s (standard SI).
    if 'speed' in df.columns:
        v = df['speed'].fillna(0)
    else:
        v = pd.Series([0.0]*len(df), index=df.index)

    # Wind Logic
    if 'wind_speed' in df.columns and 'wind_deg' in df.columns and 'bearing' in df.columns:
        w_speed = df['wind_speed'].fillna(0)
        w_deg = df['wind_deg'].fillna(0)
        ride_bearing = df['bearing'].fillna(0)
        
        # Calculate Apparent Wind
        rad_diff = np.radians(w_deg - ride_bearing)
        v_hdwind = w_speed * np.cos(rad_diff)
        v_air = v + v_hdwind
        v_air = v_air.abs()
    else:
        v_air = v # Fallback if no wind data
    
    # Calculate Air Resistance
    r_a = v_air.apply(R_air_boundary_layer)
    r_total = base_rsi + r_a

    # Heat Generation (Watts)
    # Power = Metabolic_Rate * Efficiency
    if 'power' in df.columns:
        q_gen = df['power'].fillna(0) * (1/EFFICIENCY - 1)
        total_gen = df['power'].fillna(0) * (1/EFFICIENCY)
    else:
        q_gen = pd.Series([0.0]*len(df))
        total_gen = pd.Series([0.0]*len(df))

    # --- FIX: Time Delta Calculation ---
    # We use the 'time' column instead of the index
    if 'time' in df.columns:
        dt_series = pd.to_datetime(df['time']).diff().dt.total_seconds().fillna(1.0)
    else:
        # Fallback if index is datetime
        try:
            dt_series = df.index.to_series().diff().dt.total_seconds().fillna(1.0)
        except AttributeError:
            # Last resort fallback: 1 second per row
            dt_series = pd.Series([1.0]*len(df), index=df.index)

    # Convert to numpy arrays for fast looping
    q_gen_arr = q_gen.values
    total_gen_arr = total_gen.values
    r_total_arr = r_total.values
    dt_arr = dt_series.values
    T_air = T_AIR.values
    RH_arr = RH.values
    v_air_arr = v_air.values

    # --- 3. INTEGRATION LOOP ---
    for i in range(len(df)):
        dt = dt_arr[i]
        if dt > 300 or dt <= 0: dt = 1.0 # Handle pauses or errors

        H = q_gen_arr[i] 
        R = r_total_arr[i]
        
        # 1. Dry heat loss         
        Q_dry = A * (t_skin - T_air[i]) / R 
        
        # 2. Evaporative heat loss
        drive = max(0, t_core - 37.0)
        potential_sweat_cooling = drive * 1000 # suitable for average users. Professionals may be up to 1500 
        
        # Max Evap (Lewis Relation approx). 
        max_evap_capacity = 150 * (v_air_arr[i]**0.3) * (1 - RH_arr[i]) * A
        Q_evap = min(potential_sweat_cooling, max_evap_capacity)
        
        # 3. Respiratory heat loss
        factr = total_gen_arr[i] # Metabolic Rate (M)
        
        # Vapor Pressure in kPa
        vp_kpa = calculate_vapor_pressure_kpa(T_air[i], RH_arr[i])
        
        # Sensible (Warming air)
        Q_resp_sensible = 0.0014 * factr * (34 - T_air[i])
        
        # Latent (Humidifying air). 
        Q_resp_latent = 0.0173 * factr * (5.87 - vp_kpa)
        
        Q_resp = max(0, Q_resp_sensible + Q_resp_latent)

        # Total heat loss
        Q_loss = Q_dry + Q_evap + Q_resp

        # Net storage
        S = H - Q_loss 
        dS = S * dt / (RIDER_MASS * C_BODY)

        t_core += dS
        
        # Skin temp model: relax towards equilibrium between Core and Air
        t_skin = 0.8 * t_skin + 0.2 * ((t_core + T_air[i])/2) 

        core_temps.append(t_core)
        skin_temps.append(t_skin)
        heat_gen_list.append(H)
        heat_loss_list.append(Q_loss)

    # --- 4. ASSEMBLE RESULTS ---
    result_df = df.copy()
    result_df['core_temp'] = core_temps
    result_df['skin_temp'] = skin_temps
    result_df['heat_generated_W'] = heat_gen_list
    result_df['heat_lost_W'] = heat_loss_list

    return result_df
