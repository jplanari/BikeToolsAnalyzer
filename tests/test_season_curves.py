"""The year-by-year power curves (decoupling itself is ride_metrics.aerobic_decoupling, tested in test_ride_metrics)."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import power_profile as pp


def _records(rows):
    return pd.DataFrame(rows, columns=['duration', 'watts', 'date', 'ride_id'])


def test_season_curves_keep_each_years_own_best():
    recs = _records([
        (300, 300.0, pd.Timestamp("2024-05-01"), 1),
        (300, 280.0, pd.Timestamp("2025-05-01"), 2),
        (300, 320.0, pd.Timestamp("2026-03-01"), 3),
        (1200, 250.0, pd.Timestamp("2026-03-01"), 3),
    ])
    seasons = pp.season_curves(recs, today=pd.Timestamp("2026-10-01"), years=4)
    assert list(seasons) == [2026, 2025, 2024]              # newest first
    assert seasons[2026].set_index('duration').loc[300, 'watts'] == 320.0
    assert seasons[2025].set_index('duration').loc[300, 'watts'] == 280.0
    assert 1200 not in seasons[2025].index                  # a 2025 curve only has 2025's efforts


def test_season_curves_ignore_future_years_and_empty_input():
    recs = _records([(300, 300.0, pd.Timestamp("2027-01-05"), 1), (300, 290.0, pd.Timestamp("2026-02-01"), 2)])
    assert list(pp.season_curves(recs, today=pd.Timestamp("2026-10-01"))) == [2026]
    assert pp.season_curves(_records([]), today=pd.Timestamp("2026-10-01")) == {}
    assert pp.season_curves(None) == {}
