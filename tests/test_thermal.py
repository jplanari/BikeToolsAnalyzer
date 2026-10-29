import numpy as np
import pandas as pd
import pytest

from src.physics import aerodyn as ae
from src.physics.thermal import calculate_thermal_profile


def ride(speed=8.0, wind=0.0, wind_from=0.0, bearing=0.0, n=1800, power=220.0, temp=22.0):
    df = pd.DataFrame({'time': pd.date_range("2026-07-01 09:00", periods=n, freq="1s"),
                       'speed': speed, 'power': power, 'temp_c': temp, 'relative_humidity': 0.5})
    if wind:
        df['wind_speed'], df['wind_deg'], df['bearing'] = wind, wind_from, bearing
    return df


def final_core(**kw):
    out = calculate_thermal_profile(ride(**kw), 70, 175, ["Summer Jersey"])
    return out['core_temp'].iloc[-1], out['heat_lost_W'].mean()


# --- apparent air speed ---------------------------------------------------------------------------------

def test_wind_at_rider_height_and_direction():
    head = ae.apparent_air_speed(ride(wind=5, wind_from=0))
    tail = ae.apparent_air_speed(ride(wind=5, wind_from=180))
    assert head.iloc[100] == pytest.approx(8 + 5 * ae.WIND_HEIGHT_FACTOR)         # not the raw 10 m wind
    assert tail.iloc[100] == pytest.approx(8 - 5 * ae.WIND_HEIGHT_FACTOR)


def test_cross_wind_adds_to_the_air_speed_that_cools_the_skin():
    cross = ae.apparent_air_speed(ride(wind=6, wind_from=90)).iloc[100]
    assert cross == pytest.approx(np.hypot(8, 6 * ae.WIND_HEIGHT_FACTOR))
    assert cross > 8                                                               # the along-road part alone gives 8
    drag_only = ae._air_speed(ride(wind=6, wind_from=90), ride()['speed']).iloc[100]
    assert drag_only == pytest.approx(8)                                           # drag ignores the cross wind


def test_standing_still_feels_the_whole_wind():
    df = ride(speed=0.0, wind=5, wind_from=90, n=60)
    assert ae.apparent_air_speed(df).iloc[30] == pytest.approx(5 * ae.WIND_HEIGHT_FACTOR)


def test_no_wind_data_is_just_ground_speed():
    assert ae.apparent_air_speed(ride()).iloc[100] == pytest.approx(8.0)


# --- the model ------------------------------------------------------------------------------------------------

def test_head_wind_cools_more_than_tail_wind():
    core_head, loss_head = final_core(wind=6, wind_from=0)
    core_tail, loss_tail = final_core(wind=6, wind_from=180)
    assert loss_head > loss_tail and core_head < core_tail


def test_cross_wind_cools_more_than_none():
    _, still = final_core()
    _, cross = final_core(wind=8, wind_from=90)
    assert cross > still


def test_smoothing_stops_gps_speed_noise_from_driving_the_result():
    rng = np.random.default_rng(3)
    clean = ride()
    noisy = ride()
    noisy['speed'] = 8.0 + rng.normal(0, 1.0, len(noisy))
    a = calculate_thermal_profile(clean, 70, 175, ["Summer Jersey"])['core_temp'].iloc[-1]
    b = calculate_thermal_profile(noisy, 70, 175, ["Summer Jersey"])['core_temp'].iloc[-1]
    assert b == pytest.approx(a, abs=0.15)


def test_output_columns_unchanged():
    out = calculate_thermal_profile(ride(wind=3, wind_from=45), 70, 175, [])
    assert {'core_temp', 'skin_temp', 'heat_generated_W', 'heat_lost_W'} <= set(out.columns)
    assert out['core_temp'].between(35, 42).all()
