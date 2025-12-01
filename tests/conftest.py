# tests/conftest.py
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_power_series():
    """Returns a constant power series of 200W for 1 hour."""
    return pd.Series(np.full(3600, 200), name='power')

@pytest.fixture
def variable_power_series():
    """Returns a specific pattern: 300W for 10 min, 100W for 10 min."""
    # 600 seconds of 300W, 600 seconds of 100W
    p1 = np.full(600, 300)
    p2 = np.full(600, 100)
    return pd.Series(np.concatenate([p1, p2]), name='power')

@pytest.fixture
def synthetic_climb_df():
    """
    Creates a synthetic ride with:
    - 1km flat
    - 5km climb at 5%
    - 1km flat
    """
    # 1. Flat: 1000m, 0% grade
    dist_flat1 = np.arange(0, 1000, 10) # 10m steps
    ele_flat1 = np.full(len(dist_flat1), 100)
    
    # 2. Climb: 5000m, 5% grade (Rise = 5m per 100m = 0.5m per 10m step)
    dist_climb = np.arange(1000, 6000, 10)
    ele_climb = np.linspace(100, 350, len(dist_climb)) # Gain 250m over 5km = 5%
    
    # 3. Flat: 1000m
    dist_flat2 = np.arange(6000, 7000, 10)
    ele_flat2 = np.full(len(dist_flat2), 350)
    
    dist = np.concatenate([dist_flat1, dist_climb, dist_flat2])
    ele = np.concatenate([ele_flat1, ele_climb, ele_flat2])
    
    # Create time (assuming 10m/s speed => 1s per step for simplicity)
    time_idx = pd.date_range(start='2024-01-01', periods=len(dist), freq='1s')
    
    df = pd.DataFrame({
        'time': time_idx,
        'dist_m': dist,
        'ele': ele
    })
    return df
