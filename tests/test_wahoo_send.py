import base64
import json
from datetime import datetime, time, timezone

import pandas as pd
import pytest

from src.analysis import wahoo_plan as wp
from src.analysis import workout as wo
from src.data import db
from src.data import wahoo as wh
from src.data import wahoo_send as ws
from tests.test_strava import temp_db, FakeResponse  # noqa: F401 (fixtures)

CFG = wh.WahooConfig("cid", "secret", "http://localhost:8501/", timezone="Europe/Madrid")
BLOCKS = [
    {'name': 'Warm-up', 'reps': 1, 'work_min': 15, 'work_w': 140, 'rest_min': 0, 'rest_w': 0},
    {'name': 'Intervals', 'reps': 4, 'work_min': 8, 'work_w': 255, 'rest_min': 4, 'rest_w': 125},
    {'name': 'Cool-down', 'reps': 1, 'work_min': 10, 'work_w': 125, 'rest_min': 0, 'rest_w': 0},
]


# --- plan.json -----------------------------------------------------------------------------

def test_plan_matches_the_schema_and_the_planner_duration():
    plan = wp.plan_dict(BLOCKS, 250, "Threshold 4x8")
    h = plan["header"]
    assert h["version"] == "1.0.0" and h["workout_type_family"] == 0 and h["workout_type_location"] == 1
    assert h["ftp"] == 250
    assert wp.total_seconds(plan["intervals"]) == wo.session_stats(BLOCKS, 250)['duration_s']
    rep = plan["intervals"][1]
    assert rep["exit_trigger_type"] == "repeat" and rep["exit_trigger_value"] == 2      # 3 passes + the last work
    assert [i["name"] for i in rep["intervals"]] == ["Intervals", "Recover"]
    assert plan["intervals"][2]["name"] == "Intervals"                                  # last rep, no rest after
    for step in (plan["intervals"][0], rep["intervals"][0]):
        t = step["targets"][0]
        assert t["type"] == "watts" and t["low"] <= t["high"]
    assert rep["intervals"][0]["intensity_type"] == "ftp" and rep["intervals"][1]["intensity_type"] == "recover"


@pytest.mark.parametrize("blocks", [
    [{'name': 'a', 'reps': 2, 'work_min': 5, 'work_w': 300, 'rest_min': 3, 'rest_w': 100}],
    [{'name': 'b', 'reps': 3, 'work_min': 1, 'work_w': 400, 'rest_min': 0, 'rest_w': 0}],
    [{'name': '', 'reps': 3, 'work_min': 0, 'work_w': 0, 'rest_min': 2, 'rest_w': 90}],
    [{'name': 'c', 'reps': 1, 'work_min': 20, 'work_w': 200, 'rest_min': 5, 'rest_w': 100}],
])
def test_plan_duration_always_matches_the_planner(blocks):
    assert wp.total_seconds(wp.intervals(blocks, 250)) == wo.session_stats(blocks, 250)['duration_s']


def test_no_empty_target_arrays():
    def walk(items):
        for i in items:
            if i["exit_trigger_type"] == "repeat":
                assert "targets" not in i
                walk(i["intervals"])
            else:
                assert i["targets"]
    walk(wp.plan_dict(BLOCKS, 250, "x")["intervals"])


# --- scheduling helpers --------------------------------------------------------------------------

def test_start_is_converted_to_utc_in_the_configured_zone():
    assert ws.starts_utc("2026-10-31", time(8, 0), "Europe/Madrid") == "2026-10-31T07:00:00Z"   # CET after the change
    assert ws.starts_utc("2026-07-10", time(8, 0), "Europe/Madrid") == "2026-07-10T06:00:00Z"   # CEST
    assert ws.starts_utc("2026-10-31", time(8, 0), "Europe/Madrid", second=1).endswith("07:00:01Z")


def test_unknown_timezone_is_reported():
    with pytest.raises(ws.SendError):
        ws.starts_utc("2026-10-31", time(8, 0), "Mars/Base")


def test_sync_window():
    assert ws.in_sync_window("2026-10-29", today="2026-10-29") and ws.in_sync_window("2026-11-04", today="2026-10-29")
    assert not ws.in_sync_window("2026-11-05", today="2026-10-29")


# --- send / resend / remove ----------------------------------------------------------------------------

@pytest.fixture
def api(temp_db, monkeypatch):
    db.save_wahoo_auth("Rider", {"access_token": "A", "refresh_token": "R", "expires_at": 4_000_000_000,
                                 "wahoo_name": "Pep"})
    sid = db.add_structured_session("Rider", pd.Timestamp("2026-10-31"), BLOCKS, 90, "Threshold 4x8")
    calls = []

    def fake(method, path, token, params=None, data=None, usage=None):
        calls.append((method, path, params, data))
        if (method, path) == ("GET", "/plans"):
            return {"plans": []}
        if (method, path) == ("POST", "/plans"):
            return {"id": 700}
        if (method, path) == ("POST", "/workouts"):
            return {"id": 800}
        return {}
    monkeypatch.setattr(wh, "api_request", fake)
    return sid, calls


def _row(sid):
    df = db.get_planned_sessions("Rider", structured=True)
    return df[df['id'] == sid].iloc[0].to_dict()


def test_first_send_creates_plan_and_schedules_workout(api):
    sid, calls = api
    res = ws.send_session(CFG, "Rider", _row(sid), 250, time(8, 0), now=datetime(2026, 10, 29, tzinfo=timezone.utc))
    assert res == {'plan_id': 700, 'workout_id': 800, 'starts': '2026-10-31T07:00:00Z', 'updated': False}
    kinds = [(m, p) for m, p, *_ in calls]
    assert kinds == [("GET", "/plans"), ("POST", "/plans"), ("POST", "/workouts")]
    form = calls[1][3]
    uri = form["plan[file]"]
    assert uri.startswith("data:application/json;base64,")
    plan = json.loads(base64.b64decode(uri.split(",", 1)[1]))
    assert plan["header"]["name"] == "Threshold 4x8" and plan["intervals"]
    assert form["plan[external_id]"].startswith("biketools-") and form["plan[external_id]"].endswith(f"-{sid}")
    w = calls[2][3]
    assert (w["workout[plan_id]"], w["workout[starts]"], w["workout[minutes]"]) == (700, "2026-10-31T07:00:00Z", 69)
    assert db.get_wahoo_sent("Rider")[sid]["workout_id"] == 800


def test_resend_updates_in_place_and_shifts_the_start_a_second(api):
    sid, calls = api
    ws.send_session(CFG, "Rider", _row(sid), 250, time(8, 0))
    calls.clear()
    res = ws.send_session(CFG, "Rider", _row(sid), 250, time(8, 0))
    assert res['updated'] and res['starts'] == "2026-10-31T07:00:01Z"
    assert [(m, p) for m, p, *_ in calls] == [("PUT", "/plans/700"), ("PUT", "/workouts/800")]
    assert "plan[external_id]" not in calls[0][3]                        # cannot be changed on update
    calls.clear()
    assert ws.send_session(CFG, "Rider", _row(sid), 250, time(8, 0))['starts'] == "2026-10-31T07:00:00Z"


def test_resend_after_the_workout_was_deleted_on_wahoo(api, monkeypatch):
    sid, calls = api
    ws.send_session(CFG, "Rider", _row(sid), 250, time(8, 0))
    real = wh.api_request

    def gone(method, path, token, **k):
        if (method, path) == ("PUT", "/plans/700"):
            raise wh.WahooNotFound()
        return real(method, path, token, **k)
    monkeypatch.setattr(wh, "api_request", gone)
    res = ws.send_session(CFG, "Rider", _row(sid), 250, time(8, 0))
    assert not res['updated'] and res['plan_id'] == 700 and res['workout_id'] == 800


def test_existing_plan_with_the_same_external_id_is_reused(api, monkeypatch):
    sid, calls = api
    real = wh.api_request
    monkeypatch.setattr(wh, "api_request", lambda m, p, t, **k: {"plans": [{"id": 555}]} if (m, p) == ("GET", "/plans")
                        else real(m, p, t, **k))
    ws.send_session(CFG, "Rider", _row(sid), 250, time(8, 0))
    assert ("PUT", "/plans/555") in [(m, p) for m, p, *_ in calls]


def test_quick_sessions_cannot_be_sent(api):
    with pytest.raises(ws.SendError):
        ws.send_session(CFG, "Rider", {'id': 1, 'date': pd.Timestamp("2026-10-31"), 'note': 'x', 'blocks': None},
                        250, time(8, 0))


def test_remove_deletes_on_wahoo_and_forgets(api):
    sid, calls = api
    ws.send_session(CFG, "Rider", _row(sid), 250, time(8, 0))
    calls.clear()
    assert ws.remove_session(CFG, "Rider", sid)
    assert [(m, p) for m, p, *_ in calls] == [("DELETE", "/workouts/800"), ("DELETE", "/plans/700")]
    assert db.get_wahoo_sent("Rider") == {}
    assert not ws.remove_session(CFG, "Rider", sid)                      # nothing left to remove


def test_deleting_the_planned_session_forgets_the_link(api):
    sid, _ = api
    ws.send_session(CFG, "Rider", _row(sid), 250, time(8, 0))
    db.delete_planned_session("Rider", sid)
    assert db.get_wahoo_sent("Rider") == {}
