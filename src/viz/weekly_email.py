"""User Corner section: the weekly e-mail with last week's analytics (settings, test, send now)."""
import streamlit as st

from src.data import db, mailer
from src.data import weekly_report as wr

PRESETS = {"Gmail": ("smtp.gmail.com", 587, "starttls"), "Outlook / Office 365": ("smtp.office365.com", 587, "starttls"),
           "iCloud": ("smtp.mail.me.com", 587, "starttls"), "Other": (None, None, None)}


def _server_form():
    smtp = db.get_smtp_settings()
    with st.expander("Mail server" + (" ✅" if smtp else " (needed first)"), expanded=not smtp):
        st.caption("The app sends the e-mail through an account you own. Gmail, Outlook and iCloud need an "
                   "**app password** (created in the account's security settings), not the normal password. "
                   "It is stored in this app's database, like the other credentials.")
        preset = st.selectbox("Provider", list(PRESETS), key="smtp_preset")
        host0, port0, sec0 = PRESETS[preset]
        with st.form("smtp_form"):
            c1, c2, c3 = st.columns([3, 1, 2])
            host = c1.text_input("Server", value=(smtp or {}).get('host') if preset == "Other" and smtp else (host0 or ''))
            port = c2.number_input("Port", 1, 65535, int((smtp or {}).get('port') or port0 or 587) if preset == "Other"
                                   else int(port0 or 587))
            options = list(mailer.SECURITY_CHOICES)
            sec_default = (smtp or {}).get('security') if preset == "Other" and smtp else (sec0 or 'starttls')
            security = c3.selectbox("Security", options, index=options.index(sec_default))
            username = st.text_input("User name", value=(smtp or {}).get('username') or '')
            password = st.text_input("Password / app password", type="password",
                                     placeholder="leave empty to keep the saved one" if smtp else '')
            sender = st.text_input("Send from", value=(smtp or {}).get('sender') or '',
                                   help="Usually the same address as the user name.")
            if st.form_submit_button("Save mail server"):
                if not host.strip() or not sender.strip():
                    st.error("The server and the sender address are required.")
                else:
                    pw = password or (smtp or {}).get('password')
                    db.save_smtp_settings(host, port, security, username or sender, pw, sender)
                    st.rerun()
        if smtp and st.button("Remove mail server", key="smtp_remove"):
            db.delete_smtp_settings()
            st.rerun()


def render_weekly_email(user_name):
    st.subheader("📧 Weekly e-mail")
    st.caption("Every Monday the previous week's analytics arrive as one PDF: load, power profile, durability, "
               "intensity, all the deeper analytics, and a section for each ride. It is sent the first time the "
               "app is opened in the new week, so the app has to be running (or opened) for it to go out.")
    _server_form()
    smtp = db.get_smtp_settings()
    cfg = db.get_weekly_email(user_name) or {}
    with st.form("weekly_email_form"):
        enabled = st.checkbox("Send me the weekly report", value=bool(cfg.get('enabled')))
        recipient = st.text_input("E-mail address", value=cfg.get('recipient') or '')
        if st.form_submit_button("Save"):
            if enabled and ("@" not in recipient or "." not in recipient.split("@")[-1]):
                st.error("Enter a valid e-mail address.")
            elif enabled and not smtp:
                st.error("Set up the mail server above first.")
            else:
                db.save_weekly_email(user_name, enabled, recipient)
                st.success("Saved." if not enabled else "Saved. The next report goes out when the app is opened "
                           "after the week ends.")
    if cfg.get('last_status'):
        st.caption(f"Last attempt: {cfg['last_status']}")
    if not cfg.get('recipient') or not smtp:
        return
    monday, sunday = wr.previous_week()
    c1, c2 = st.columns(2)
    if c1.button("Send a test e-mail", key="weekly_test"):
        try:
            mailer.send_test(cfg['recipient'])
            st.success(f"Test e-mail sent to {cfg['recipient']}.")
        except mailer.MailError as e:
            st.error(str(e))
    if c2.button(f"Send last week now ({monday:%d %b} – {sunday:%d %b})", key="weekly_now",
                 help="Builds the full report and sends it immediately; it does not count as the automatic one."):
        with st.spinner("Building the report… (a week with many rides takes a minute)"):
            try:
                summary = wr.send_weekly(user_name, monday, sunday)
                st.success(f"Sent to {cfg['recipient']} ({summary['pages']} pages, {summary['rides']} rides).")
                for f in summary['failures']:
                    st.warning(f"Left out: {f}")
            except Exception as e:
                st.error(str(e))
