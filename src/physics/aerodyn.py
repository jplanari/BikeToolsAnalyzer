import numpy as np
import pandas as pd

def get_air_density(elevation_m, temp_c=15):
    #Standard atmosphere model for air density
    P0 = 101325  # Sea level standard atmospheric pressure, Pa
    T0 = 288.15  # Sea level standard temperature, K
    L = 0.0065   # Temperature lapse rate, K/m
    R = 287.05   # Ideal gas constant for air, J/(kg·K)
    g = 9.80665  # Gravitational acceleration, m/s²

    T = T0 - L * elevation_m
    P = P0 * (1 - (L * elevation_m) / T0)
    rho = P / (R * T)
    return rho

def calculate_CdA(df, rider_mass, bike_mass=8.0, crr=0.004, rho=1.225, drivetrain_loss=0.03):
    g = 9.80665  # m/s²: gravitational acceleration
    total_mass = rider_mass + bike_mass  # kg

    if df.empty or 'speed' not in df.columns or 'power' not in df.columns:
        return pd.Series(dtype=float)

    if 'grade_smooth' in df.columns:
        slope = df['grade_smooth'] / 100.0
    else:
        slope = 0.0

    v = df['speed'] # m/s. Ground speed

    if 'wind_speed' in df.columns and 'bearing' in df.columns:
        w_speed = df['wind_speed'].fillna(0)
        w_deg = df['wind_deg'].fillna(0)
        ride_bearing = df['bearing'].fillna(0)
        rad_diff = np.radians(w_deg - ride_bearing)
        v_hdwind = w_speed * np.cos(rad_diff)
        v_air = v + v_hdwind

        v_air = v_air.abs()
    else:
        v_air = v

    p_input = df['power']

    p_wheel = p_input * (1 - drivetrain_loss)

    # Acceleration term

    accel = df['speed'].diff().fillna(0)  # m/s²
    p_accel = total_mass * v * accel

    # Rolling resistance term
    p_roll = total_mass * g * crr * v

    # Gravitational term
    p_grav = total_mass * g * slope * v

    # Aerodynamic drag term
    p_aero = p_wheel - p_accel - p_roll - p_grav

    valid_speed_mask = v > 5.0

    cda_series = p_aero / (0.5 * rho * v**2*v_air)
    cda_series = cda_series.where(valid_speed_mask, np.nan)

    cda_series = cda_series.mask(cda_series < 0, np.nan)
    cda_series = cda_series.mask(cda_series > 1.0, np.nan)

    return cda_series, p_aero

def get_avg_cda(cda_series):
    if cda_series.empty:
        return np.nan
    return cda_series.mean()

def get_avg_power(p_aero_series):
    if p_aero_series.empty:
        return np.nan
    return p_aero_series.mean()


