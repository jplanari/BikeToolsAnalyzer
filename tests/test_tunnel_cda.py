"""CdA across a tunnel: speed is zero while pedalling, then the position jumps; the average speed stands in."""
import numpy as np
import pandas as pd
import pytest

from src.physics import aerodyn as ad

RHO, CDA, MASS, CRR, LOSS = 1.2, 0.30, 78.0, 0.005, 0.03


def ride(n=400, v=10.0, tunnel=None, with_cadence=True):
    """Flat ride at constant speed with the power that CdA=0.30 needs; `tunnel=(first, last)` freezes speed and
    position over those seconds (and the position catches up with the rider at last+1)."""
    p = (0.5 * RHO * CDA * v ** 3 + MASS * ad.G * CRR * v) / (1 - LOSS)
    dist = np.arange(n) * v
    speed = np.full(n, v)
    if tunnel:
        a, b = tunnel
        speed[a:b + 1] = 0.0
        dist[a:b + 1] = dist[a - 1]
    df = pd.DataFrame({'time': pd.date_range("2026-01-01", periods=n, freq="1s"), 'dist_m': dist, 'speed': speed,
                       'power': p, 'grade_smooth': 0.0})
    if with_cadence:
        df['cadence'] = 90.0
    return df


def fit(df):
    return ad.estimate_cda(df, rider_mass=MASS - 8.0, bike_mass=8.0, crr=CRR, rho=RHO, drivetrain_loss=LOSS)


def test_a_ride_without_tunnels_is_unchanged():
    assert ad.find_gps_stalls(ride()) == []
    res = fit(ride())
    assert res['cda'] == pytest.approx(CDA, abs=0.01) and res['tunnel_seconds'] == 0


def test_tunnel_is_found_with_its_average_speed():
    df = ride(tunnel=(100, 159))                           # 60 s frozen; the position is back at sample 160
    (i0, i1, v_avg, grade), = ad.find_gps_stalls(df)
    assert (i0, i1) == (99, 160)
    assert v_avg == pytest.approx(10.0)                    # (pos_end - pos_ini) / time = 610 m / 61 s
    assert grade == 0.0


def test_cda_is_estimated_through_the_tunnel():
    df = ride(tunnel=(100, 159))
    res = fit(df)
    assert res['cda'] == pytest.approx(CDA, abs=0.02)
    assert res['tunnel_seconds'] >= 55
    cda, _ = ad.calculate_CdA(df, rider_mass=MASS - 8.0, rho=RHO)
    assert cda.iloc[110:150].notna().all()                 # the tunnel has CdA values, not gaps


def test_tunnel_does_not_poison_the_samples_next_to_it():
    # the 610 m jump in one second must not leak into the smoothed speed of the neighbouring samples
    b = ad._balance(ride(tunnel=(100, 159)), MASS - 8.0, 8.0, CRR, RHO, LOSS)
    assert b['v'].max() < 12.0 and b['v'].iloc[95:170].min() > 8.0


def test_stopped_and_not_pedalling_is_not_a_tunnel():
    df = ride(tunnel=(100, 159))
    df.loc[100:159, 'power'] = 0.0
    assert ad.find_gps_stalls(df) == []


def test_trainer_ride_with_no_movement_is_not_a_tunnel():
    df = ride()
    df['speed'] = 0.0
    df['dist_m'] = 0.0
    assert ad.find_gps_stalls(df) == []


def test_short_dropout_and_slow_average_are_ignored():
    assert ad.find_gps_stalls(ride(tunnel=(100, 105))) == []          # 6 s
    slow = ride(v=3.0, tunnel=(100, 159))
    assert ad.find_gps_stalls(slow) == []                              # 3 m/s is too slow to read drag


def test_without_a_cadence_sensor_power_alone_counts():
    assert len(ad.find_gps_stalls(ride(tunnel=(100, 159), with_cadence=False))) == 1


def test_gradient_of_the_tunnel_comes_from_the_elevation_change():
    df = ride(tunnel=(100, 159))
    df['ele'] = 100.0
    df.loc[160:, 'ele'] = 100.0 + 0.04 * 610               # 4 % over the 610 m of the stretch
    (_, _, _, grade), = ad.find_gps_stalls(df)
    assert grade == pytest.approx(4.0, abs=0.1)
