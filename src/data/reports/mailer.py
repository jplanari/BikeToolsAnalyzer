"""
Sends e-mail through the SMTP account stored in the app (smtplib only: no extra dependency).
No Streamlit here, so it also runs from a background thread.
"""
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formatdate

from src.data import db

TIMEOUT_S = 30
SECURITY_CHOICES = ('starttls', 'ssl', 'none')


class MailError(Exception):
    """Anything that stops a message from being sent; the text is meant for the user."""


def is_configured():
    s = db.get_smtp_settings()
    return bool(s and s['host'] and s['sender'])


def build_message(sender, recipient, subject, text, html=None, attachments=()):
    """attachments: iterable of (filename, bytes, mime 'type/subtype')."""
    msg = EmailMessage()
    msg['From'], msg['To'], msg['Subject'] = sender, recipient, subject
    msg['Date'] = formatdate(localtime=True)
    msg.set_content(text)
    if html:
        msg.add_alternative(html, subtype='html')
    for filename, data, mime in attachments:
        maintype, _, subtype = mime.partition('/')
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)
    return msg


def send_message(msg, settings=None):
    """Sends a built message with the stored (or given) SMTP settings. Raises MailError."""
    s = settings or db.get_smtp_settings()
    if not s or not s.get('host') or not s.get('sender'):
        raise MailError("No mail server is set up yet.")
    security = s.get('security') or 'starttls'
    try:
        if security == 'ssl':
            server = smtplib.SMTP_SSL(s['host'], int(s['port']), timeout=TIMEOUT_S, context=ssl.create_default_context())
        else:
            server = smtplib.SMTP(s['host'], int(s['port']), timeout=TIMEOUT_S)
        with server:
            if security == 'starttls':
                server.starttls(context=ssl.create_default_context())
            if s.get('username'):
                server.login(s['username'], s.get('password') or '')
            server.send_message(msg)
    except smtplib.SMTPAuthenticationError:
        raise MailError("The mail server refused the user name or password "
                        "(Gmail and others need an app password, not the account password).")
    except (smtplib.SMTPException, OSError) as e:
        raise MailError(f"Could not send the e-mail: {e}")


def send_test(recipient, settings=None):
    """A short message that proves the settings work."""
    s = settings or db.get_smtp_settings()
    if not s:
        raise MailError("No mail server is set up yet.")
    send_message(build_message(s['sender'], recipient, "BikeTools Analyzer: test e-mail",
                               "This is a test from BikeTools Analyzer. Your weekly report e-mails will arrive here."),
                 settings=s)
