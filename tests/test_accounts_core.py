"""Accounts for a shared server: sign-up, e-mail confirmation, sign-in, sessions, password reset, deletion."""
import time

import pytest

from src.data import accounts_mail, db
from src.data.db import accounts as acc
from src.security import Throttle
from tests.test_db import temp_db  # noqa: F401 (fixture)


@pytest.fixture(autouse=True)
def quick(temp_db, monkeypatch):
    monkeypatch.setattr(acc, "COST", {'n': 2 ** 10, 'r': 8, 'p': 1})
    for name, args in (("login_throttle", (10, 600)), ("client_throttle", (30, 600)),
                       ("register_throttle", (8, 3600)), ("reset_throttle", (5, 3600))):
        monkeypatch.setattr(acc, name, Throttle(*args))
    mails = []
    monkeypatch.setattr(acc, "deliver", lambda kind, email, token, name: mails.append((kind, email, token, name)))
    return mails


def signed_up(mails, name="Nuria", email="nuria@example.com", password="a long password"):
    ok, msg = acc.register(name, email, password)
    assert ok, msg
    return mails[-1][2]


def verified(mails, **kw):
    assert acc.verify_email(signed_up(mails, **kw))[0]


def test_sign_up_sends_a_link_and_the_account_cannot_sign_in_until_confirmed(quick):
    token = signed_up(quick)
    assert quick[-1][:2] == ("verify", "nuria@example.com") and token
    assert acc.login("nuria@example.com", "a long password")[0] == "unverified"
    ok, msg = acc.verify_email(token)
    assert ok
    status, session, name = acc.login("NURIA@Example.com", "a long password")          # e-mail is case-insensitive
    assert (status, name) == ("ok", "Nuria") and session
    assert acc.verify_email(token)[0] is False                                         # a link works once


def test_secrets_are_not_stored_in_the_clear(quick):
    token = signed_up(quick)
    acc.verify_email(token)
    _, session, _ = acc.login("nuria@example.com", "a long password")
    with db.connect() as c:
        stored = " ".join(str(v) for t in ("users", "account_tokens", "sessions") for row in c.execute(f"SELECT * FROM {t}") for v in row)
    assert "a long password" not in stored and token not in stored and session not in stored
    assert db.get_user_data("Nuria")["has_password"] is True and "password_hash" not in db.get_user_data("Nuria")


def test_a_wrong_password_or_unknown_address_look_the_same(quick):
    verified(quick)
    assert acc.login("nuria@example.com", "nope")[0] == "wrong"
    assert acc.login("nobody@example.com", "nope")[0] == "wrong"
    assert acc.login("not an email", "x")[0] == "wrong"


def test_sign_in_is_throttled_per_address_and_per_client(quick):
    verified(quick)
    results = [acc.login("nuria@example.com", "bad", client=f"c{i}")[0] for i in range(12)]
    assert results[:10] == ["wrong"] * 10 and results[10:] == ["throttled"] * 2
    assert acc.login("nuria@example.com", "a long password", client="other")[0] == "throttled"   # even the right one waits
    flood = [acc.login(f"u{i}@example.com", "bad", client="same")[0] for i in range(32)]
    assert flood[-1] == "throttled"


def test_sign_up_checks_its_input_and_does_not_reveal_existing_addresses(quick):
    verified(quick)
    for args in (("x", "a@b.co", "long enough pw"), ("Ok Name", "not-an-email", "long enough pw"),
                 ("Ok Name", "a@b.co", "short"), ("a/b", "a@b.co", "long enough pw")):
        assert acc.register(*args)[0] is False
    quick.clear()
    ok, msg = acc.register("Someone else", "nuria@example.com", "another long pw")           # address already has an account
    assert ok and quick == [("exists", "nuria@example.com", None, "Nuria")]
    assert db.get_user_data("Someone else") is None
    first = acc.register("Third", "third@example.com", "long enough pw")[1]
    assert first == msg.replace("nuria@example.com", "third@example.com")                    # same wording either way
    assert acc.register("nuria", "other@example.com", "long enough pw") == (False, "That rider name is taken.")


def test_sign_up_attempts_are_throttled(quick):
    out = [acc.register(f"N{i}", f"n{i}@example.com", "long enough pw", client="x")[0] for i in range(10)]
    assert out == [True] * 8 + [False] * 2


def test_sessions_expire_and_can_be_ended(quick, monkeypatch):
    verified(quick)
    _, a, _ = acc.login("nuria@example.com", "a long password")
    _, b, _ = acc.login("nuria@example.com", "a long password")
    assert acc.session_user(a) == acc.session_user(b) == "Nuria" and acc.session_user("junk") is None
    acc.logout(a)
    assert acc.session_user(a) is None and acc.session_user(b) == "Nuria"
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + acc.SESSION_TTL_S + 10)
    assert acc.session_user(b) is None


def test_a_session_in_use_is_extended(quick, monkeypatch):
    verified(quick)
    _, s, _ = acc.login("nuria@example.com", "a long password")
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + acc.SESSION_TTL_S * 0.6)
    assert acc.session_user(s) == "Nuria"
    monkeypatch.setattr(time, "time", lambda: real() + acc.SESSION_TTL_S * 1.2)
    assert acc.session_user(s) == "Nuria"                                                      # it was renewed at 60 %


def test_password_reset_works_once_signs_everyone_out_and_does_not_reveal_accounts(quick):
    verified(quick)
    _, s, _ = acc.login("nuria@example.com", "a long password")
    quick.clear()
    ok1, msg1 = acc.request_password_reset("nuria@example.com")
    ok2, msg2 = acc.request_password_reset("nobody@example.com")
    assert ok1 and ok2 and msg1 == msg2 and [m[:2] for m in quick] == [("reset", "nuria@example.com")]
    token = quick[0][2]
    assert acc.reset_password(token, "short")[0] is False and acc.reset_password("junk", "a new long one")[0] is False
    assert acc.reset_password(token, "a new long one")[0]
    assert acc.reset_password(token, "again long one")[0] is False
    assert acc.session_user(s) is None
    assert acc.login("nuria@example.com", "a long password")[0] == "wrong"
    assert acc.login("nuria@example.com", "a new long one")[0] == "ok"


def test_a_newer_reset_link_replaces_the_older_and_links_expire(quick, monkeypatch):
    verified(quick)
    acc.request_password_reset("nuria@example.com")
    old = quick[-1][2]
    acc.request_password_reset("nuria@example.com")
    new = quick[-1][2]
    assert acc.reset_password(old, "a new long one")[0] is False
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + acc.RESET_TTL_S + 5)
    assert acc.reset_password(new, "a new long one")[0] is False


def test_resetting_counts_as_confirming_the_address(quick):
    signed_up(quick)
    acc.request_password_reset("nuria@example.com")
    assert acc.reset_password(quick[-1][2], "a new long one")[0]
    assert acc.login("nuria@example.com", "a new long one")[0] == "ok"


def test_resend_verification_only_for_unconfirmed_accounts(quick):
    signed_up(quick)
    quick.clear()
    acc.resend_verification("nuria@example.com")
    acc.resend_verification("nobody@example.com")
    assert [m[0] for m in quick] == ["verify"] and acc.verify_email(quick[0][2])[0]
    quick.clear()
    acc.resend_verification("nuria@example.com")
    assert quick == []                                                                         # already confirmed


def test_changing_the_password_needs_the_current_one_and_ends_other_sessions(quick):
    verified(quick)
    _, a, _ = acc.login("nuria@example.com", "a long password")
    _, b, _ = acc.login("nuria@example.com", "a long password")
    assert acc.change_password("Nuria", "wrong", "a new long one")[0] is False
    assert acc.change_password("Nuria", "a long password", "short")[0] is False
    assert acc.change_password("Nuria", "a long password", "a new long one", keep_token=a)[0]
    assert acc.session_user(a) == "Nuria" and acc.session_user(b) is None


def test_deleting_an_account_removes_the_rider_and_their_sessions(quick):
    verified(quick)
    verified(quick, name="Other", email="other@example.com")
    _, s, _ = acc.login("nuria@example.com", "a long password")
    assert acc.delete_account("Nuria", "wrong")[0] is False and db.get_user_data("Nuria")
    assert acc.delete_account("Nuria", "a long password")[0]
    assert db.get_user_data("Nuria") is None and acc.session_user(s) is None
    assert acc.account_info("Other") == {"email": "other@example.com", "verified": True}
    assert acc.register("Nuria", "nuria@example.com", "a long password")[0]                    # the address is free again


def test_riders_without_an_account_still_work_as_before(quick):
    assert db.save_user("Desktop Rider")[0]
    assert db.get_user_data("Desktop Rider")["has_password"] is False
    assert acc.login("", "")[0] == "wrong"
    assert acc.account_info("Desktop Rider") == {"email": None, "verified": False}


def test_the_mails_carry_a_link_to_the_public_address(monkeypatch):
    monkeypatch.setenv("BIKETOOLS_PUBLIC_URL", "https://bike.example.net/")
    subject, text = accounts_mail.compose("verify", "tok+en/1", "Nuria")
    assert "https://bike.example.net/?verify=tok%2Ben%2F1" in text and "Nuria" in text
    assert "?reset=" in accounts_mail.compose("reset", "abc", "N")[1]
    assert "already" in accounts_mail.compose("exists", None, "N")[1].lower()


def test_a_broken_mail_server_does_not_change_the_answer(monkeypatch):
    from src.data.reports import mailer
    monkeypatch.setattr(mailer, "send_message", lambda *a, **k: (_ for _ in ()).throw(mailer.MailError("down")))
    accounts_mail.send("verify", "a@b.co", "tok", "N")                                         # must not raise
