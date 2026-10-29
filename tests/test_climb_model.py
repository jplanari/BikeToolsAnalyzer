import numpy as np
import pandas as pd
import pytest

from src.physics import climb_model as cm
from src.viz.climb_estimator import climb_baseline, BIKE_MASS_KG


def _climb(length=5000.0, grade=0.07):
    d = np.linspace(0, length, 200)
    return cm.climb_profile(d, d * grade)


def test_profile_steps_and_grade():
    lengths, grades = _climb()
    assert lengths.sum() == pytest.approx(5000) and len(lengths) == 200
    assert grades == pytest.approx(0.07, abs=1e-6)
    assert cm.climb_profile([0, 10], [0, 1])[0].size == 0           # shorter than one step
    assert cm.climb_profile([], [])[0].size == 0


def test_speed_balances_the_forces():
    g = np.array([0.07])
    v = cm.speed_for_power(280, g, 78, 0.32, 0.005, 1.2, 0.97)[0]
    th = np.arctan(0.07)
    force = 78 * cm.G * (np.sin(th) + 0.005 * np.cos(th)) + 0.5 * 1.2 * 0.32 * v ** 2
    assert v * force == pytest.approx(280 * 0.97, rel=1e-6)
    assert 3.5 < v < 5.0                                             # about 16 km/h


def test_realistic_time_for_a_known_climb():
    lengths, grades = _climb()
    r = cm.climb_time(lengths, grades, 280, 78, 0.32, 0.005)
    assert 1050 < r["time_s"] < 1200 and r["gain_m"] == pytest.approx(350)
    assert r["vam_m_h"] == pytest.approx(350 / r["time_s"] * 3600)


def test_more_power_less_weight_less_drag_less_rolling_are_all_faster():
    lengths, grades = _climb()
    base = cm.climb_time(lengths, grades, 280, 78, 0.32, 0.005)["time_s"]
    for kw in ({"power_w": 300}, {"mass_kg": 75}, {"cda": 0.28}, {"crr": 0.003}):
        args = dict(power_w=280, mass_kg=78, cda=0.32, crr=0.005)
        args.update(kw)
        assert cm.climb_time(lengths, grades, args["power_w"], args["mass_kg"], args["cda"],
                             args["crr"])["time_s"] < base
    # weight matters far more than CdA on a 7 % climb
    gain_kg = base - cm.climb_time(lengths, grades, 280, 77, 0.32, 0.005)["time_s"]
    gain_cda = base - cm.climb_time(lengths, grades, 280, 78, 0.31, 0.005)["time_s"]
    assert gain_kg > 5 * gain_cda


def test_flat_ground_is_drag_dominated():
    lengths, grades = cm.climb_profile(np.linspace(0, 5000, 100), np.zeros(100))
    v = cm.speed_for_power(250, grades, 78, 0.32, 0.005, 1.2, 0.97)[0]
    assert 9 < v < 11                                                # ~35 km/h on the flat


def test_descent_speed_is_capped():
    v = cm.speed_for_power(100, np.array([-0.10]), 78, 0.32, 0.005)[0]
    assert v <= cm.MAX_SPEED_MS + 1e-9


def test_power_for_time_inverts_climb_time():
    lengths, grades = _climb()
    t = cm.climb_time(lengths, grades, 263, 76, 0.30, 0.004)["time_s"]
    assert cm.power_for_time(lengths, grades, t, 76, 0.30, 0.004) == pytest.approx(263, abs=0.01)
    assert cm.power_for_time(lengths, grades, 60, 76, 0.30, 0.004) is None      # 5 km at 7 % in a minute
    assert cm.power_for_time([], [], 600, 76, 0.3, 0.004) is None


def test_baseline_reproduces_the_ride_from_its_samples():
    n = 1200                                                                # 20 min
    df = pd.DataFrame({
        "time": pd.date_range("2026-10-01 08:00", periods=n, freq="1s"),
        "dist_m": np.linspace(0, 5000, n), "ele": np.linspace(100, 450, n),
        "power": [280.0] * n, "cda": [0.30] * n})
    climb = {"start_idx": 0, "end_idx": n - 1}
    b = climb_baseline(df, climb, {"weight": 70.0, "ftp": 250})
    assert b["mass"] == 70 + BIKE_MASS_KG and b["power"] == pytest.approx(280) and b["cda"] == pytest.approx(0.30)
    assert b["actual_s"] == n - 1 and b["grades"].mean() == pytest.approx(0.07, abs=1e-3)
    assert 0.9 < b["rho"] < 1.3
    df2 = df.drop(columns=["power", "cda"])                                 # no power meter, no CdA
    b2 = climb_baseline(df2, climb, {"weight": 70.0, "ftp": 250})
    assert b2["power"] is None and b2["cda"] == 0.32
