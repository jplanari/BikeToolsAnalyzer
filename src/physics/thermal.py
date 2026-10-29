import pandas as pd
import numpy as np
import streamlit as st

from src.physics.aerodyn import apparent_air_speed

# Constants
C_BODY = 3470  # Specific heat capacity [J/(kg·K)]
EFFICIENCY = 0.23              # gross efficiency (mechanical power / total metabolic rate)
M_REST_W_PER_M2 = 58.2         # resting metabolic rate, 1 met (ISO 7730), W per m² of body surface
K_MIN_W_PER_M2K = 12.6         # core-to-skin conductance at rest: tissue 5.28 + minimal skin blood flow 7.3 (Gagge)
SWEAT_REF_SKIN_C = 33.0         # skin temperature the evaporative-capacity coefficient (150) is referred to
MIN_AIR_SPEED_MS = 0.2          # natural convection / body movement: sweat still evaporates in still air
H_CONV_COEFF = 8.5              # combined convective + radiative coefficient h = H_CONV_COEFF * v^H_CONV_EXP, W/m²K
H_CONV_EXP = 0.84
H_STILL_AIR_W_PER_M2K = 7.8     # floor in still air: radiation (~4.7) + natural convection (~3.1), ASHRAE Fundamentals ch. 9
CORE_START_C = 36.9             # core temperature at the start of the ride (warm, rested rider)
SETPOINT_GAIN_K = 1.0           # rise of the sweat and vasodilation thresholds at an intensity equal to FTP
SETPOINT_MAX_FTP_FRACTION = 1.2 # intensity above which the set point does not rise further
SETPOINT_TAU_S = 300.0          # the set point follows the (smoothed) intensity with this time constant
EVAP_COEFF_W_PER_M2 = 150.0     # evaporative capacity of dry air at 1 m/s over skin at SWEAT_REF_SKIN_C, W/m²
EVAP_EXP = 0.3
SWEAT_ONSET_CORE_C = 37.0       # core temperature at which sweating starts
SWEAT_GAIN_W_PER_K = 1000.0     # sweat latent-heat supply per K of core temperature above onset (pros up to ~1500)
VASO_ONSET_CORE_C = 36.8        # core temperature at which skin vasodilation starts
K_VASO_MAX_W_PER_M2K = 150.0    # asymptotic extra core-to-skin conductance from vasodilation, W/m²K
SKIN_MASS_FRACTION = 0.1
C_SKIN = 3500                   # J/(kg K)
MAX_SUBSTEP_S = 5.0             # explicit Euler is unstable above ~2x the skin time constant (~80 s)

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
    """Air boundary-layer resistance, m²K/W. The combined coefficient never falls below its still-air value
    (radiation and natural convection do not vanish when the air does)."""
    speed = max(0.1, air_speed)
    return 1 / max(H_STILL_AIR_W_PER_M2K, H_CONV_COEFF * (speed ** H_CONV_EXP))

def saturation_pressure_kpa(T):
    """Saturation vapour pressure (kPa) at a temperature in °C (Magnus)."""
    return 0.6112 * np.exp((17.67 * T) / (T + 243.5))


def calculate_vapor_pressure_kpa(Ta, rh_fraction):
    """
    Calculate vapor pressure in kPa.
    rh_fraction: 0.0 to 1.0
    """
    # Saturation Vapor Pressure (es) in kPa (Magnus formula)
    es = 0.6112 * np.exp((17.67 * Ta) / (Ta + 243.5))
    return rh_fraction * es

def _step(t_core, t_skin, mech_w, v_air, T_air, RH, R, A, mass, dt, shift=0.0):
    """
    One time step of the two-node model. mech_w is the mechanical (pedalling) power; shift (K) raises the sweat and vasodilation thresholds (exercise set point). Returns
    (t_core, t_skin, heat generated, heat lost) with heat in W.
    """
    # Metabolism: gross efficiency while pedalling, but never below the resting rate
    M = max(mech_w / EFFICIENCY, M_REST_W_PER_M2 * A)
    H = M - mech_w                                   # heat produced (the work leaves the body)

    # 1. Dry heat loss through clothing and boundary layer
    Q_dry = A * (t_skin - T_air) / R

    # 2. Evaporative heat loss: sweat supply (rises once the core passes 37 °C) against what the air can
    #    take up. Capacity scales with the vapour-pressure gap between the wet skin and the air, normalised
    #    to a dry-air gap at 33 °C skin (reduces to the old (1 - RH) when air and skin are at the same temperature).
    sweat_supply = max(0.0, t_core - SWEAT_ONSET_CORE_C - shift) * SWEAT_GAIN_W_PER_K
    gap = (saturation_pressure_kpa(t_skin) - calculate_vapor_pressure_kpa(T_air, RH)) / \
        saturation_pressure_kpa(SWEAT_REF_SKIN_C)
    max_evap = EVAP_COEFF_W_PER_M2 * (max(v_air, MIN_AIR_SPEED_MS) ** EVAP_EXP) * max(0.0, gap) * A
    Q_evap = min(sweat_supply, max_evap)

    # 3. Respiratory heat loss (ISO 7730, M in W)
    vp_kpa = calculate_vapor_pressure_kpa(T_air, RH)
    Q_resp = max(0.0, 0.0014 * M * (34 - T_air) + 0.0173 * M * (5.87 - vp_kpa))

    mass_skin = mass * SKIN_MASS_FRACTION
    mass_core = mass - mass_skin
    vaso = max(0.0, t_core - VASO_ONSET_CORE_C - shift)
    K_total = K_MIN_W_PER_M2K * A + K_VASO_MAX_W_PER_M2K * A * vaso / (1 + vaso)
    Q_transfer = K_total * (t_core - t_skin)

    t_core += (H - Q_resp - Q_transfer) * dt / (mass_core * C_BODY)
    t_skin += (Q_transfer - Q_dry - Q_evap) * dt / (mass_skin * C_SKIN)
    return t_core, t_skin, H, Q_dry + Q_evap + Q_resp


def _start_state(T_air, R_clothing, A):
    """
    Core and skin temperature at the start of the ride: a warm, rested rider (core CORE_START_C) whose skin
    is at the quasi-steady value for the first sample's air and clothing, i.e. the skin balance
    K (Tc - Ts) = A (Ts - Ta) / R with the basal conductance and still air.
    """
    k = K_MIN_W_PER_M2K * A
    g = A / (R_clothing + R_air_boundary_layer(MIN_AIR_SPEED_MS))
    return CORE_START_C, (k * CORE_START_C + g * T_air) / (k + g)


def calculate_thermal_profile(df, rider_weight_kg=80.0, rider_height_cm=175, clothing_items=(), weather_data=None,
                              ftp=None):
    """
    Core and skin temperature along the ride. With ftp (W) the sweat and vasodilation thresholds rise with the
    relative intensity (exercise set point, Saltin & Hermansen 1966); without it they stay at their resting values.
    """
    if df.empty:
        return df

    RIDER_MASS = rider_weight_kg
    A = du_bois_BSA(rider_height_cm, RIDER_MASS)

    # Weather (defaults when the ride has none)
    T_AIR = df['temp_c'].ffill().bfill().fillna(15.0) if 'temp_c' in df.columns \
        else pd.Series(15.0, index=df.index)
    if 'relative_humidity' in df.columns:
        rh_raw = df['relative_humidity'].fillna(50)
        RH = rh_raw.apply(lambda x: x / 100.0 if x > 1.0 else x)          # accept 0-1 or 0-100
    else:
        RH = pd.Series(0.5, index=df.index)

    base_rsi = sum(CLOTHING_CATALOG.get(item, 0) for item in clothing_items) * 0.155   # 1 clo = 0.155 m²K/W

    # Air speed over the skin: ground speed (smoothed; 1 Hz GPS speed is noisy) plus the wind at rider
    # height, cross wind included. Same wind treatment as the drag model in physics.aerodyn.
    v_air = apparent_air_speed(df)
    r_total = base_rsi + v_air.apply(R_air_boundary_layer)

    mech = df['power'].fillna(0).clip(lower=0).to_numpy() if 'power' in df.columns else np.zeros(len(df))

    if 'time' in df.columns:
        dt_series = pd.to_datetime(df['time']).diff().dt.total_seconds().fillna(1.0)
    else:
        try:
            dt_series = df.index.to_series().diff().dt.total_seconds().fillna(1.0)
        except AttributeError:
            dt_series = pd.Series(1.0, index=df.index)

    dt_arr, T_air, RH_arr = dt_series.to_numpy(), T_AIR.to_numpy(), RH.to_numpy()
    v_air_arr, r_arr = v_air.to_numpy(), r_total.to_numpy()

    t_core, t_skin = _start_state(T_air[0], base_rsi, A)
    frac = 0.0                                          # smoothed relative intensity (fraction of FTP)

    core_temps, skin_temps, heat_gen_list, heat_loss_list = [], [], [], []
    for i in range(len(df)):
        dt = dt_arr[i]
        if dt > 300 or dt <= 0:
            dt = 1.0                                   # pauses or errors
        if ftp:
            frac += (min(mech[i] / ftp, SETPOINT_MAX_FTP_FRACTION) - frac) * min(1.0, dt / SETPOINT_TAU_S)
        shift = SETPOINT_GAIN_K * frac
        n_sub = int(np.ceil(dt / MAX_SUBSTEP_S))          # keep the explicit scheme stable across long gaps
        for _ in range(n_sub):
            t_core, t_skin, H, lost = _step(t_core, t_skin, mech[i], v_air_arr[i], T_air[i], RH_arr[i],
                                            r_arr[i], A, RIDER_MASS, dt / n_sub, shift)
        core_temps.append(t_core)
        skin_temps.append(t_skin)
        heat_gen_list.append(H)
        heat_loss_list.append(lost)

    result_df = df.copy()
    result_df['core_temp'] = core_temps
    result_df['skin_temp'] = skin_temps
    result_df['heat_generated_W'] = heat_gen_list
    result_df['heat_lost_W'] = heat_loss_list
    return result_df
