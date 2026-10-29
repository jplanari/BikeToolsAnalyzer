"""
The CdA estimate and power budget are checked against rides simulated with KNOWN physics
(speed integrated from power, mass, grade, Crr, CdA and wind), so a wrong sign, a wrong exponent or a
biased estimator shows up as CdA drifting away from the true value.
"""
import numpy as np
import pandas as pd
import pytest

from src.physics import aerodyn as ae
from src.data.gpx import calculate_bearing

G = ae.G


def simulate(cda=0.30, mass=78.0, crr=0.005, rho=1.225, loss=0.03, wind10=0.0, wind_from=0.0, bearing=0.0,
             n=2400):
    """Rolling hills and a wobbling power, riding on bearing `bearing`, wind (10 m) blowing from `wind_from`."""
    t = np.arange(n)
    grade = 0.04 * np.sin(2 * np.pi * t / 700) + 0.01 * np.sin(2 * np.pi * t / 130)
    power = 250 + 60 * np.sin(2 * np.pi * t / 300)
    v = np.zeros(n)
    v[0] = 8
    head = wind10 * ae.WIND_HEIGHT_FACTOR * np.cos(np.radians(wind_from - bearing))
    for i in range(1, n):
        th, vi = np.arctan(grade[i]), v[i - 1]
        va = vi + head
        force = mass * G * (np.sin(th) + crr * np.cos(th)) + 0.5 * rho * cda * va * abs(va)
        v[i] = max(vi + (power[i] * (1 - loss) / max(vi, 1) - force) / mass, 0.5)
    df = pd.DataFrame({'speed': v, 'power': power, 'grade_smooth': grade * 100})
    if wind10:
        df['wind_speed'], df['wind_deg'], df['bearing'] = wind10, wind_from, bearing
    return df


# --- air density ----------------------------------------------------------------------------

def test_air_density_falls_with_altitude():
    # the old formula returned 1.225 at every altitude
    assert ae.get_air_density(0, 15) == pytest.approx(1.225, abs=0.002)
    assert ae.get_air_density(1000) == pytest.approx(1.112, abs=0.003)          # standard atmosphere
    assert ae.get_air_density(2000) == pytest.approx(1.007, abs=0.003)
    assert ae.get_air_density(2000) < ae.get_air_density(1000) < ae.get_air_density(0)


def test_air_density_responds_to_temperature_and_humidity():
    cold, hot = ae.get_air_density(0, 0), ae.get_air_density(0, 30)
    assert cold == pytest.approx(1.292, abs=0.003) and hot == pytest.approx(1.165, abs=0.003)
    assert ae.get_air_density(0, 30, rel_humidity=1.0) < hot                    # moist air is lighter
    arr = ae.get_air_density(np.array([0, 1000]), np.array([15, 15]))
    assert arr.shape == (2,) and arr[1] < arr[0]


# --- CdA ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("label, kw", [
    ("no wind", {}),
    ("head wind", dict(wind10=5, wind_from=0)),
    ("tail wind", dict(wind10=5, wind_from=180)),
    ("cross wind", dict(wind10=6, wind_from=90)),
    ("oblique", dict(wind10=5, wind_from=45, bearing=270)),
])
def test_cda_is_recovered_with_and_without_wind(label, kw):
    df = simulate(**kw)
    fit = ae.estimate_cda(df, rider_mass=70, rho=1.225)
    assert fit["cda"] == pytest.approx(0.30, abs=0.004), label
    series, _ = ae.calculate_CdA(df, rider_mass=70, rho=1.225)
    assert series.mean() == pytest.approx(0.30, abs=0.01), label                 # the plotted series agrees


def test_other_true_values_are_recovered():
    for cda, mass, crr in ((0.24, 72, 0.004), (0.38, 95, 0.007)):
        df = simulate(cda=cda, mass=mass, crr=crr)
        fit = ae.estimate_cda(df, rider_mass=mass - 8, crr=crr, rho=1.225)
        assert fit["cda"] == pytest.approx(cda, abs=0.004)


def test_noise_does_not_inflate_the_estimate():
    rng = np.random.default_rng(5)
    df = simulate()
    df['speed'] += rng.normal(0, 0.3, len(df))                  # GPS-derived speed is this noisy
    df['power'] += rng.normal(0, 40, len(df))
    assert ae.estimate_cda(df, rider_mass=70, rho=1.225)["cda"] == pytest.approx(0.30, abs=0.01)


def test_wrong_inputs_show_up_in_cda_as_expected():
    df = simulate()
    low = ae.estimate_cda(df, rider_mass=70, crr=0.004, rho=1.225)["cda"]       # assumed Crr too low
    assert low > 0.30                                                           # missing rolling power -> drag
    heavy = ae.estimate_cda(df, rider_mass=90, rho=1.225)["cda"]                # assumed weight too high
    assert heavy != pytest.approx(0.30, abs=0.01)


def test_cda_none_when_there_is_nothing_to_fit():
    assert ae.estimate_cda(simulate(n=60), rider_mass=70) is None               # under two minutes
    slow = simulate(); slow['speed'] = 3.0
    assert ae.estimate_cda(slow, rider_mass=70) is None                         # never above 18 km/h
    assert ae.estimate_cda(pd.DataFrame({'speed': [8.0]}), rider_mass=70) is None   # no power channel
    s, p = ae.calculate_CdA(pd.DataFrame(), rider_mass=70)
    assert s.empty and p.empty


def test_density_uses_altitude_and_weather_when_present():
    df = simulate()
    df['ele'] = 2000.0
    assert ae.estimate_cda(df, rider_mass=70, rho=1.225)["cda"] == pytest.approx(0.30, abs=0.004)
    # with the same speeds, thinner air at 2000 m means a larger CdA for the same drag power
    thin = ae.estimate_cda(df, rider_mass=70)["cda"]
    assert thin == pytest.approx(0.30 * 1.225 / ae.get_air_density(2000), rel=0.02)


# --- power budget ---------------------------------------------------------------------------------

def test_components_add_up_to_the_wheel_power():
    df = simulate()
    grav, roll, accel = ae.calculate_power_components(df, rider_mass=70)
    _, aero = ae.calculate_CdA(df, rider_mass=70)
    total = grav + roll + accel + aero
    assert total.mean() == pytest.approx((df['power'] * (1 - ae.DRIVETRAIN_LOSS)).mean(), rel=1e-9)
    # the same rolling-resistance coefficient on both sides (used to differ: 0.005 vs 0.004)
    assert roll.mean() == pytest.approx(78 * G * ae.CRR * df['speed'].mean(), rel=0.01)


def test_acceleration_term_is_smoothed_not_noise():
    df = simulate(); rng = np.random.default_rng(1)
    df['speed'] += rng.normal(0, 0.3, len(df))
    _, _, accel = ae.calculate_power_components(df, rider_mass=70)
    assert accel.abs().mean() < 60                                              # raw 1 s differences gave ~250 W


# --- bearing -----------------------------------------------------------------------------------------

def test_bearing_is_stable_under_gps_jitter():
    n = 300
    rng = np.random.default_rng(2)
    lat = 41.0 + np.arange(n) * 8.0 / 111_000                                   # riding due north at 8 m/s
    lon = 2.0 + rng.normal(0, 2.5, n) / (111_000 * np.cos(np.radians(41)))      # 2.5 m of sideways jitter
    b = calculate_bearing(pd.DataFrame({'lat': lat, 'lon': lon}))
    off = np.minimum(b, 360 - b)
    assert off.iloc[10:-10].abs().max() < 25 and off.iloc[10:-10].abs().mean() < 8
    east = calculate_bearing(pd.DataFrame({'lat': 41.0 + 0 * np.arange(n), 'lon': 2.0 + np.arange(n) * 1e-4}))
    assert east.iloc[50] == pytest.approx(90, abs=0.5)


# --- weather ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("unit, factor", [("m/s", 1.0), ("km/h", 1 / 3.6)])
def test_wind_is_converted_to_metres_per_second(monkeypatch, unit, factor):
    from src.data import weather

    asked = {}

    class Resp:
        def json(self):
            hours = [f"2026-10-01T{h:02d}:00" for h in range(24)]
            return {"hourly_units": {"wind_speed_10m": unit},
                    "hourly": {"time": hours, "wind_speed_10m": [10.0] * 24, "wind_direction_10m": [90.0] * 24,
                               "temperature_2m": [15.0] * 24, "surface_pressure": [1000.0] * 24,
                               "relative_humidity_2m": [50.0] * 24}}

    def fake_get(url, params=None, **k):
        asked.update(params)
        return Resp()
    monkeypatch.setattr(weather.requests, "get", fake_get)
    df = pd.DataFrame({'time': pd.date_range("2026-10-01 08:00", periods=60, freq="1s"),
                       'lat': 41.0, 'lon': 2.0})
    out = weather.fetch_ride_weather(df)
    assert asked["wind_speed_unit"] == "ms"                                       # the documented name
    assert out['wind_speed'].iloc[0] == pytest.approx(10.0 * factor)
    assert out['relative_humidity'].iloc[0] == pytest.approx(0.5)
