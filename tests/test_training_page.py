import pandas as pd
import pytest

from src.data import db
from src.viz import training_page as tpg


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "test.db"))
    db.init_db()
    db.save_user("Rider", 250, 170, 70.0, 180.0)


def test_planned_sessions_roundtrip(temp_db):
    today = pd.Timestamp('2026-10-28')
    rows = pd.DataFrame({'date': ['2026-10-30', '2026-10-31', '2026-10-27', None, '2026-11-01'],
                         'tss': [120, None, 50, 80, -5], 'note': ['Long ride', None, 'past', None, 'bad']})
    assert db.replace_planned_sessions("Rider", rows, today) == 1     # only the valid future row
    plan = db.get_planned_sessions("Rider", after=today)
    assert plan.to_dict('records') == [{'date': pd.Timestamp('2026-10-30'), 'tss': 120.0, 'note': 'Long ride'}]

    # saving again replaces future sessions (no duplicates)
    db.replace_planned_sessions("Rider", pd.DataFrame({'date': ['2026-11-03'], 'tss': [90], 'note': [None]}), today)
    assert db.get_planned_sessions("Rider", after=today)['tss'].tolist() == [90.0]


def test_status_labels():
    assert tpg.ramp_status(10) == "⚠️ Aggressive" and tpg.ramp_status(0) == "→ Maintaining"
    assert tpg.acwr_status(1.0) == "✅ In range" and tpg.acwr_status(1.6) == "⛔ Spike"
    assert tpg.acwr_status(float('nan')) is None
    assert tpg.form_status(-40) == "⚠️ Very fatigued" and tpg.form_status(10) == "✅ Fresh"
    assert tpg.monotony_status(2.5) == "⚠️ High"
    assert tpg.tid_label(0.8, 0.08, 0.12, 2.08) == "Polarized"
    assert tpg.tid_label(0.7, 0.2, 0.1, 1.54) == "Pyramidal"
    assert tpg.tid_label(0.3, 0.6, 0.1, 1.0) == "Threshold-focused"
