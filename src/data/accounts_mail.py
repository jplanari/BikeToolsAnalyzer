"""The e-mails of the account flows: confirm the address, choose a new password, "you already have an account".

They go through the SMTP account stored in the app (src/data/reports/mailer.py). The links point at the server's public
address, BIKETOOLS_PUBLIC_URL (for example https://bike.example.net), with ?verify=<token> or ?reset=<token>.
"""
import os
from urllib.parse import quote

from src.i18n import t

SUBJECTS = {
    "verify": "Confirm your e-mail address",
    "reset": "Choose a new password",
    "exists": "You already have an account",
}


def public_url():
    return os.environ.get("BIKETOOLS_PUBLIC_URL", "").strip().rstrip("/")


def link(kind, token):
    base = public_url()
    return f"{base}/?{'verify' if kind == 'verify' else 'reset'}={quote(token, safe="")}" if base else None


def compose(kind, token, name):
    """(subject, text) of the message, in the server's language."""
    if kind == "verify":
        url = link("verify", token) or token
        return (t("Confirm your e-mail address"),
                t("Hi {name},\n\nOpen this link to confirm your e-mail address:\n\n{url}\n\nIt works for 3 days. If you did "
                  "not create this account, ignore this message.\n", name=name, url=url))
    if kind == "reset":
        url = link("reset", token) or token
        return (t("Choose a new password"),
                t("Hi {name},\n\nOpen this link to choose a new password:\n\n{url}\n\nIt works for one hour and only once. "
                  "If you did not ask for it, ignore this message: your password has not changed.\n", name=name, url=url))
    return (t("You already have an account"),
            t("Hi {name},\n\nSomeone tried to create an account with this e-mail address, but you already have one. If it "
              "was you, sign in, or use \"Forgot password\" if you do not remember it. If not, ignore this message.\n",
              name=name))


def send(kind, email, token, name):
    """Sends one message. Failures are swallowed on purpose: the caller's answer must not depend on whether the mail
    server worked (or on whether the address exists), and the person can ask again."""
    from src.data.reports import mailer
    subject, text = compose(kind, token, name)
    try:
        sender = mailer.db.get_smtp_settings()
        mailer.send_message(mailer.build_message((sender or {}).get("sender", ""), email, subject, text))
    except Exception:
        pass
