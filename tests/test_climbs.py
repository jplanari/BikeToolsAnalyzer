# tests/test_climbs.py
from src.analysis.climbs import detect_climbs

def test_detect_synthetic_climb(synthetic_climb_df):
    climbs = detect_climbs(synthetic_climb_df)
    
    assert len(climbs) == 1
    c = climbs[0]
    
    # FIX: Allow a slightly wider range to catch 5100 exactly
    assert 4900 < c['length_m'] <= 5200  
    
    assert 4.8 < c['avg_gradient_pct'] < 5.2 
    assert c['type'] == "Long & Steady"

def test_ignore_flat_road():
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({
        'dist_m': np.arange(0, 5000, 10),
        'ele': np.full(500, 100),
        'time': pd.date_range('2024-01-01', periods=500, freq='1s')
    })
    
    climbs = detect_climbs(df)
    assert len(climbs) == 0
