"""The administrator's view: server-wide figures, read-only, never part of the app's pages."""
import ast
import os
import time

import pytest

from src.data import db
from src.data.db import accounts as acc
from src.data.db import admin
from src.security import Throttle
from tests.test_db import temp_db  # noqa: F401 (fixture)
from tools import admin as cli


@pytest.fixture
def server(temp_db, monkeypatch):
    monkeypatch.setattr(acc, "COST", {'n': 2 ** 10, 'r': 8, 'p': 1})
    monkeypatch.setattr(acc, "register_throttle", Throttle(50, 3600))
    mails = []
    monkeypatch.setattr(acc, "deliver", lambda kind, email, token, name: mails.append(token))
    for name, mail in (("Ana", "ana@example.com"), ("Beni", "beni@example.com")):
        assert acc.register(name, mail, "a long password")[0]
    assert acc.verify_email(mails[0])[0]                      # Ana confirmed, Beni still waiting
    acc.login("ana@example.com", "a long password")
    today = time.strftime("%Y-%m-%d")
    for i in range(3):
        ok, msg = db.save_ride("Ana", f"a{i}.fit", b"x" * (1000 + i), {"date": f"{today}T08:00:00Z", "dist": 30}, {60: 300.0})
        assert ok, msg
    assert db.save_ride("Beni", "b.fit", b"y" * 500, {"date": "2020-01-01T08:00:00Z", "dist": 10}, {60: 200.0})[0]
    return None


def test_overview_counts_riders_rides_and_storage(server):
    ov = admin.overview()
    assert (ov["riders"], ov["riders_with_account"], ov["accounts_confirmed"], ov["accounts_waiting_for_confirmation"]) == (2, 2, 1, 1)
    assert ov["rides"] == 4 and ov["rides_last_7_days"] == 3 and ov["rides_last_30_days"] == 3
    assert ov["ride_files"] == 4 and 0 < ov["ride_files_bytes"] <= 3003 + 500
    assert ov["new_accounts_last_7_days"] == 2 and ov["sessions_active"] == 1 and ov["riders_seen_last_7_days"] == 1
    assert ov["database_bytes"] > 0 and any(t[0] == "rides" for t in ov["tables"])
    assert ov["per_rider_average"]["rides"] == 2.0


def test_riders_report_lists_each_rider_with_their_own_figures(server):
    rows = {r["name"]: r for r in admin.riders_report()}
    assert rows["Ana"]["rides"] == 3 and rows["Beni"]["rides"] == 1
    assert rows["Ana"]["state"] == "confirmed" and rows["Beni"]["state"] == "waiting for confirmation"
    assert rows["Ana"]["stored_bytes"] > rows["Beni"]["stored_bytes"] > 0
    assert rows["Ana"]["last_seen"] and rows["Beni"]["last_seen"] is None and rows["Beni"]["last_ride"] == "2020-01-01"


def test_a_desktop_rider_without_account_is_listed(temp_db):
    db.save_user("Local", 250, 170, 70.0, 175.0)
    rows = admin.riders_report()
    assert rows[0]["name"] == "Local" and rows[0]["state"].startswith("no account") and rows[0]["rides"] == 0
    ov = admin.overview()
    assert ov["riders"] == 1 and ov["riders_with_account"] == 0


def test_the_command_line_prints_all_three_views(server, capsys):
    for command in ("overview", "riders", "tables"):
        assert cli.main([command]) == 0
    out = capsys.readouterr().out
    assert "Activities (rides)" in out and "ana@example.com" in out and "ride_files" in out
    assert cli.main(["riders", "--json"]) == 0
    assert '"stored_bytes"' in capsys.readouterr().out


@pytest.mark.sqlite_only
def test_an_empty_or_missing_database_gives_a_message_not_a_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "never-created.db"))
    assert cli.main(["overview"]) == 1
    assert "start the app once" in capsys.readouterr().err


def test_size_and_age_formatting():
    assert cli.size(None) == "-" and cli.size(512) == "512 B" and cli.size(1536) == "1.5 KB"
    assert cli.size(5 * 1024 ** 3) == "5.0 GB"
    assert cli.ago(None) == "-" and cli.ago(1000, now=1000) == "today" and cli.ago(0 + 1, now=1 + 3 * 86400) == "3 d ago"


def test_no_page_of_the_app_imports_the_admin_module():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    offenders = []
    for folder in ("src", ):
        for base, _, files in os.walk(os.path.join(root, folder)):
            for f in files:
                if not f.endswith(".py") or f == "admin.py":
                    continue
                path = os.path.join(base, f)
                tree = ast.parse(open(path, encoding="utf-8").read())
                for node in ast.walk(tree):
                    names = []
                    if isinstance(node, ast.ImportFrom):
                        names = [(node.module or "") + "." + a.name for a in node.names]
                    elif isinstance(node, ast.Import):
                        names = [a.name for a in node.names]
                    if any(n.endswith("db.admin") or n.endswith("db.admin.") or ".db.admin" in n or n == "src.data.db.admin" for n in names) \
                            or (isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("data.db") and
                                any(a.name == "admin" for a in node.names)):
                        offenders.append(os.path.relpath(path, root))
    assert not offenders, "Riders' pages must never import the admin figures: " + ", ".join(offenders)
