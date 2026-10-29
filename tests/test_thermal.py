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


def _ride(power, temp, rh=0.5, speed=8.0, n=3600):
    df = pd.DataFrame({'time': pd.date_range("2026-07-01 09:00", periods=n, freq="1s"), 'speed': speed,
                       'power': power, 'temp_c': temp, 'relative_humidity': rh})
    return calculate_thermal_profile(df, 70, 175, ["Summer Jersey"])


def test_still_air_keeps_a_heat_transfer_floor():
    from src.physics import thermal as th
    assert 1.0 / th.R_air_boundary_layer(0.0) == pytest.approx(th.H_STILL_AIR_W_PER_M2K)
    assert 1.0 / th.R_air_boundary_layer(10.0) == pytest.approx(th.H_CONV_COEFF * 10 ** th.H_CONV_EXP)


def test_start_state_is_a_warm_rider_with_ambient_dependent_skin():
    cold, warm = _ride(0, 5, speed=0, n=10), _ride(0, 30, speed=0, n=10)
    assert cold.core_temp.iloc[0] == pytest.approx(36.9, abs=0.05)
    assert 24 < cold.skin_temp.iloc[0] < warm.skin_temp.iloc[0] < 36.9


def test_resting_rider_is_stable_in_comfortable_air_and_cools_slowly_in_cold():
    warm = _ride(0, 30, speed=0)
    assert warm.core_temp.iloc[-1] == pytest.approx(warm.core_temp.iloc[0], abs=0.3)
    mild = _ride(0, 20, speed=0)
    assert 36.0 < mild.core_temp.iloc[-1] <= mild.core_temp.iloc[0]          # no shivering in the model, but no collapse


def _with_ftp(power, temp, ftp, n=3600):
    df = pd.DataFrame({'time': pd.date_range("2026-07-01 09:00", periods=n, freq="1s"), 'speed': 8.0,
                       'power': power, 'temp_c': temp, 'relative_humidity': 0.5})
    return calculate_thermal_profile(df, 70, 175, ["Summer Jersey"], ftp=ftp)


def test_exercise_raises_the_set_point_in_proportion_to_relative_intensity():
    none, hard, easy = _with_ftp(200, 20, None), _with_ftp(200, 20, 250), _with_ftp(200, 20, 400)
    end = lambda d: d.core_temp.iloc[-1]
    assert end(hard) - end(none) > 0.5                       # about 38 degC, as measured at ~80 % of FTP
    assert end(none) < end(easy) < end(hard)                 # a stronger rider is less stressed at the same power
    assert 37.5 < end(hard) < 38.6


def test_set_point_follows_the_effort_with_a_lag():
    none, hard = _with_ftp(200, 20, None), _with_ftp(200, 20, 250)
    gap = (hard.core_temp - none.core_temp)
    assert gap.iloc[60] < gap.iloc[1800]                     # builds up over minutes
    assert gap.iloc[0] == pytest.approx(0.0, abs=1e-3)


def test_without_ftp_the_thresholds_stay_at_rest_values():
    assert _with_ftp(200, 30, None).core_temp.iloc[-1] == pytest.approx(_with_ftp(200, 30, 0).core_temp.iloc[-1])


def test_humidity_raises_core_temperature_monotonically():
    ends = [_ride(220, 28, rh=rh).core_temp.iloc[-1] for rh in (0.2, 0.5, 0.8, 0.95)]
    assert ends == sorted(ends) and ends[-1] - ends[0] > 0.5


def test_cool_humid_air_still_evaporates():
    assert _ride(200, 10, rh=0.9).core_temp.iloc[-1] < 37.5


def test_energy_is_conserved():
    """Stored heat (both nodes) equals generated minus lost heat, summed over the ride."""
    from src.physics import thermal as th
    out = _ride(250, 25, speed=8.0, n=1800)
    mass, a = 70.0, th.du_bois_BSA(175, 70.0)
    skin_m = mass * th.SKIN_MASS_FRACTION
    d_store = (mass - skin_m) * th.C_BODY * (out.core_temp.iloc[-1] - out.core_temp.iloc[0]) \
        + skin_m * th.C_SKIN * (out.skin_temp.iloc[-1] - out.skin_temp.iloc[0])
    # step i integrates the heat of sample i, starting from the temperature before sample 0
    assert a > 0
    flow = ((out.heat_generated_W - out.heat_lost_W) * 1.0).iloc[1:].sum()
    assert d_store == pytest.approx(flow, rel=0.02)


def test_sparse_sampling_is_stable():
    fine = _ride(250, 30, n=3600)
    df = pd.DataFrame({'time': pd.date_range("2026-07-01 09:00", periods=18, freq="200s"), 'speed': 8.0,
                       'power': 250.0, 'temp_c': 30.0, 'relative_humidity': 0.5})
    coarse = calculate_thermal_profile(df, 70, 175, ["Summer Jersey"])
    assert coarse.skin_temp.between(25, 42).all()
    assert coarse.core_temp.iloc[-1] == pytest.approx(fine.core_temp.iloc[-1], abs=0.5)
