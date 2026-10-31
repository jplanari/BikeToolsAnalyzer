import pandas as pd
import pytest

from src.data import db, mailer
from src.data import weekly_report as wr


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "test.db"))
    db.init_db()
    db.save_user("Ana", 250, 170, 70.0)
    return tmp_path


def _enable(recipient="ana@example.com"):
    db.save_smtp_settings("smtp.example.com", 587, "starttls", "ana", "pw", "ana@example.com")
    db.save_weekly_email("Ana", True, recipient)


def test_previous_week_is_monday_to_sunday():
    mon, sun = wr.previous_week("2026-10-28")            # a Wednesday
    assert (mon, sun) == (pd.Timestamp("2026-10-19"), pd.Timestamp("2026-10-25"))
    assert wr.previous_week("2026-10-26")[0] == pd.Timestamp("2026-10-19")      # Monday: the week just ended
    assert wr.previous_week("2026-10-25")[1] == pd.Timestamp("2026-10-18")      # Sunday: the week before


def test_smtp_and_weekly_settings_roundtrip(temp_db):
    assert db.get_smtp_settings() is None and db.get_weekly_email("Ana") is None
    _enable()
    assert db.get_smtp_settings()['host'] == "smtp.example.com"
    assert db.get_weekly_email("Ana")['enabled'] == 1
    assert db.users_with_weekly_email() == ["Ana"]
    db.save_weekly_email("Ana", False, "ana@example.com")
    assert db.users_with_weekly_email() == []
    db.delete_smtp_settings()
    assert db.get_smtp_settings() is None


def test_each_week_is_claimed_once(temp_db):
    _enable()
    assert db.claim_weekly_email("Ana", "2026-10-19") is True
    assert db.claim_weekly_email("Ana", "2026-10-19") is False        # a second tab / process
    assert db.claim_weekly_email("Ana", "2026-10-26") is True         # the next week is new
    db.release_weekly_email("Ana", "failed", "2026-10-19")
    assert db.claim_weekly_email("Ana", "2026-10-26") is True         # released: tried again


def test_disabled_user_is_never_claimed(temp_db):
    _enable()
    db.save_weekly_email("Ana", False, "ana@example.com")
    assert db.claim_weekly_email("Ana", "2026-10-19") is False


def test_message_has_pdf_attachment():
    msg = mailer.build_message("a@x.com", "b@x.com", "Subject", "Body", attachments=[("r.pdf", b"%PDF-1", "application/pdf")])
    parts = list(msg.iter_attachments())
    assert len(parts) == 1 and parts[0].get_filename() == "r.pdf" and parts[0].get_content_type() == "application/pdf"
    assert msg['To'] == "b@x.com"


def test_send_message_without_setup_raises(temp_db):
    with pytest.raises(mailer.MailError):
        mailer.send_message(mailer.build_message("a@x.com", "b@x.com", "s", "t"))


def test_send_if_due_sends_once_and_records(temp_db, monkeypatch):
    _enable()
    calls = []
    monkeypatch.setattr(wr, "send_weekly", lambda user, mon, sun: calls.append((user, mon, sun)))
    assert wr.send_if_due("Ana", today="2026-10-28") is True
    assert calls == [("Ana", pd.Timestamp("2026-10-19"), pd.Timestamp("2026-10-25"))]
    assert wr.send_if_due("Ana", today="2026-10-29") is False        # same week: nothing more
    assert wr.send_if_due("Ana", today="2026-11-03") is True          # a new week
    assert "sent" in db.get_weekly_email("Ana")['last_status']


def test_failed_send_is_retried_next_visit(temp_db, monkeypatch):
    _enable()
    def boom(*a):
        raise mailer.MailError("server down")
    monkeypatch.setattr(wr, "send_weekly", boom)
    assert wr.send_if_due("Ana", today="2026-10-28") is False
    row = db.get_weekly_email("Ana")
    assert "server down" in row['last_status'] and row['last_week'] is None
    monkeypatch.setattr(wr, "send_weekly", lambda *a: None)
    assert wr.send_if_due("Ana", today="2026-10-28") is True


def test_nothing_sent_without_mail_server(temp_db, monkeypatch):
    db.save_weekly_email("Ana", True, "ana@example.com")
    monkeypatch.setattr(wr, "send_weekly", lambda *a: pytest.fail("must not send"))
    assert wr.send_if_due("Ana", today="2026-10-28") is False


def test_weekly_pdf_builds_for_a_week_without_rides(temp_db):
    pdf, summary = wr.build_weekly_pdf("Ana", pd.Timestamp("2026-10-19"), pd.Timestamp("2026-10-25"))
    assert pdf.startswith(b"%PDF") and summary['rides'] == 0
    assert summary['pages'] >= 10                              # cover + 6 report pages + 3 deeper pages
    assert "No rides this week" in wr.body_text("Ana", pd.Timestamp("2026-10-19"), pd.Timestamp("2026-10-25"), summary)


def test_send_weekly_attaches_the_pdf(temp_db, monkeypatch):
    _enable()
    sent = []
    monkeypatch.setattr(mailer, "send_message", lambda msg, settings=None: sent.append(msg))
    wr.send_weekly("Ana", pd.Timestamp("2026-10-19"), pd.Timestamp("2026-10-25"))
    assert len(sent) == 1
    att = list(sent[0].iter_attachments())
    assert att[0].get_filename() == "weekly-report-Ana-2026-10-19.pdf" and sent[0]['To'] == "ana@example.com"


def test_weekly_pdf_has_a_section_per_ride(temp_db):
    from src.data import strava as sv
    from tests.test_strava import make_streams, activity
    payload = sv.build_ride_payload(activity(7, "2026-10-21T07:00:00Z"), make_streams(n=1800))
    ok, msg = db.save_ride("Ana", sv.ride_filename(7), payload,
                           {'date': "2026-10-21T07:00:00+00:00", 'tss': 40, 'distance_km': 20.0}, {60: 1.0})
    assert ok, msg
    mon, sun = pd.Timestamp("2026-10-19"), pd.Timestamp("2026-10-25")
    assert len(wr.week_rides("Ana", mon, sun)) == 1
    pdf, summary = wr.build_weekly_pdf("Ana", mon, sun)
    empty_pdf, empty = wr.build_weekly_pdf("Ana", pd.Timestamp("2026-10-12"), pd.Timestamp("2026-10-18"))
    assert summary['rides'] == 1 and summary['failures'] == []
    assert summary['pages'] == empty['pages'] + 4                  # the ride's four pages
