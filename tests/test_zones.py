import numpy as np
import pandas as pd
import pytest

from src.analysis.power import coggan_zones, time_in_zones
from src.analysis.hr import hr_effort_zones, time_in_hr_zones


def test_power_zones_are_contiguous():
    zones = coggan_zones(250)
    bounds = list(zones.values())
    assert bounds[0][0] == 0
    for (_, hi), (lo, _) in zip(bounds, bounds[1:]):
        assert hi == lo
    assert bounds[-1][1] == np.inf


def test_power_values_in_former_gaps_are_counted():
    # 0.555*FTP and 0.755*FTP fell between zones before and were dropped
    ftp = 200
    series = pd.Series([0.555 * ftp, 0.755 * ftp, 0.905 * ftp])
    counts = time_in_zones(series, coggan_zones(ftp))
    assert sum(counts.values()) == 3
    assert counts[2] == 1 and counts[3] == 1 and counts[4] == 1


def test_power_zone_boundaries_are_half_open():
    zones = coggan_zones(200)
    # A value exactly on a lower bound belongs to that zone; very high goes to Z7
    counts = time_in_zones(pd.Series([zones[2][0], zones[5][0], 2000.0, 0.0]), zones)
    assert counts == {1: 1, 2: 1, 3: 0, 4: 0, 5: 1, 6: 0, 7: 1}


def test_time_in_zones_ignores_nan_and_empty():
    zones = coggan_zones(200)
    assert sum(time_in_zones(pd.Series([np.nan, 100.0, np.nan]), zones).values()) == 1
    assert time_in_zones(pd.Series([], dtype=float), zones) == {z: 0 for z in zones}
    assert time_in_zones(None, zones) == {z: 0 for z in zones}


def test_hr_zones_cover_every_sample():
    lthr = 170
    zones = hr_effort_zones(lthr)
    assert list(zones) == ["Z1 (Recovery)", "Z2 (Aerobic)", "Z3 (Tempo)",
                           "Z4 (Threshold)", "Z5 (VO2 Max)", "Z6 (Anaerobic)"]
    # Every integer bpm from 60 to 220 lands in exactly one zone
    hr = pd.Series(np.arange(60, 221, dtype=float))
    counts = time_in_hr_zones(hr, lthr)
    assert sum(counts.values()) == len(hr)
    # 0.895*LTHR (was in the 0.89-0.90 gap) is now Z2
    assert time_in_hr_zones(pd.Series([0.895 * lthr]), lthr)["Z2 (Aerobic)"] == 1
    # Above the old hard cap of 250 bpm still counts
    assert time_in_hr_zones(pd.Series([260.0]), lthr)["Z6 (Anaerobic)"] == 1
