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
    plan = db.get_planned_sessions("Rider", after=today)[['date', 'tss', 'note']]
    assert plan.to_dict('records') == [{'date': pd.Timestamp('2026-10-30'), 'tss': 120.0, 'note': 'Long ride'}]

    # saving again replaces future sessions (no duplicates)
    db.replace_planned_sessions("Rider", pd.DataFrame({'date': ['2026-11-03'], 'tss': [90], 'note': [None]}), today)
    assert db.get_planned_sessions("Rider", after=today)['tss'].tolist() == [90.0]


def test_structured_sessions_are_kept_when_quick_entries_are_replaced(temp_db):
    today = pd.Timestamp('2026-10-28')
    blocks = [{'name': 'Intervals', 'reps': 4, 'work_min': 8, 'work_w': 240, 'rest_min': 4, 'rest_w': 125}]
    sid = db.add_structured_session("Rider", '2026-10-31', blocks, 77.5, note="Threshold")
    db.replace_planned_sessions("Rider", pd.DataFrame({'date': ['2026-11-02'], 'tss': [60], 'note': [None]}), today)
    db.replace_planned_sessions("Rider", pd.DataFrame({'date': ['2026-11-03'], 'tss': [90], 'note': [None]}), today)

    assert db.get_planned_sessions("Rider", after=today)['tss'].tolist() == [77.5, 90.0]
    quick = db.get_planned_sessions("Rider", after=today, structured=False)
    assert quick['date'].tolist() == [pd.Timestamp('2026-11-03')]
    built = db.get_planned_sessions("Rider", after=today, structured=True)
    assert built['id'].tolist() == [sid] and built['note'].tolist() == ['Threshold']
    import json
    assert json.loads(built['blocks'].iloc[0]) == blocks

    assert db.delete_planned_session("Rider", sid) is True
    assert db.delete_planned_session("Rider", sid) is False
    db.save_user("Other", 200, 160, 65.0)
    other = db.add_structured_session("Other", '2026-11-05', blocks, 50)
    assert db.delete_planned_session("Rider", other) is False          # not theirs


def test_status_labels():
    assert tpg.ramp_status(10) == "⚠️ Aggressive" and tpg.ramp_status(0) == "→ Maintaining"
    assert tpg.acwr_status(1.0) == "✅ In range" and tpg.acwr_status(1.6) == "⛔ Spike"
    assert tpg.acwr_status(float('nan')) is None
    assert tpg.form_status(-40) == "⚠️ Very fatigued" and tpg.form_status(10) == "✅ Fresh"
    assert tpg.monotony_status(2.5) == "⚠️ High"
    assert tpg.tid_label(0.8, 0.08, 0.12, 2.08) == "Polarized"
    assert tpg.tid_label(0.7, 0.2, 0.1, 1.54) == "Pyramidal"
    assert tpg.tid_label(0.3, 0.6, 0.1, 1.0) == "Threshold-focused"
