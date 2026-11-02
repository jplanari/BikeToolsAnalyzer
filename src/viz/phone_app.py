"""User Corner section for the phone app: pair a phone (make an access token), see the paired phones, revoke one."""
import socket
import time
from urllib.parse import quote

import streamlit as st

from src.api import runner
from src.data import db
from src.i18n import t

PORT = runner.DEFAULT_PORT
NEW_TOKEN_KEY = "phone_new_token"


def lan_address(port=PORT):
    """http://<this computer's address on the local network>:<port>; falls back to localhost. Nothing is sent: a UDP
    socket "connected" to a public address only makes the system pick the interface."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(0.2)
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    return f"http://{ip}:{port}"


def pairing_link(server, token):
    """The link the Android app opens to fill in its connection form."""
    return f"biketools://pair?server={quote(server, safe='')}&token={quote(token, safe='')}"


def web_pairing_link(server, token):
    """The same for a phone browser, served by the phone server itself (the token stays in the # part, which
    browsers never send to a server)."""
    return f"{server.rstrip('/')}/#/pair?server={quote(server, safe='')}&token={quote(token, safe='')}"


def _when(ts):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else t("never")


def render_phone_app(user_name):
    st.subheader(t("📱 Phone app"))
    st.caption(t("See your rides, training load and power profile on your phone. The phone talks to this computer: "
               "it needs the phone server running, and only reads your data."))
    st.session_state.setdefault("phone_server", lan_address())
    server = st.text_input(t("Server address"), key="phone_server", help=t("What the phone types to reach this computer. Use a VPN address "
                                                      "(for example Tailscale) to reach it away from home."))
    live = runner.running()
    if live:
        st.success(t("The phone server is running on port {port}. It stops when BikeTools closes.", port=live[1]))
        if st.button(t("Stop the phone server"), key="phone_stop"):
            runner.stop()
            st.rerun()
    else:
        st.info(t("The phone server is not running. Starting it lets devices on your network reach your rides "
                  "(Windows may ask to allow the network access)."))
        if st.button(t("Start the phone server"), key="phone_start"):
            ok, message = runner.start("0.0.0.0", PORT)
            if not ok:
                st.error(t("Could not start the server: {error}", error=message))
            else:
                st.rerun()
    with st.expander(t("Run it without the app open")):
        st.code(f"python -m src.api serve --host 0.0.0.0 --port {PORT}", language="bash")
        st.caption(t("From the BikeTools folder; stop it with Ctrl+C. Plain http is only safe on a network you "
                   "trust: away from home use a VPN such as Tailscale."))

    tokens = db.list_api_tokens(user_name)
    for tok in tokens:
        c1, c2 = st.columns([4, 1])
        c1.write(t("**{label}** · paired {created} · last used {used}", label=tok['label'] or t("Phone"),
                   created=_when(tok['created_at']), used=_when(tok['last_used_at'])))
        if c2.button(t("Revoke"), key=f"phone_revoke_{tok['id']}"):
            db.revoke_api_token(user_name, tok['id'])
            st.rerun()
    if not tokens:
        st.caption(t("No phone is paired yet."))

    label = st.text_input(t("Name for this phone"), key="phone_label", max_chars=db.tokens.MAX_LABEL)
    if st.button(t("Create access token"), key="phone_create"):
        token, _ = db.create_api_token(user_name, label)
        st.session_state[NEW_TOKEN_KEY] = token
        st.rerun()
    token = st.session_state.get(NEW_TOKEN_KEY)
    if token:
        st.success(t("Copy this now: the token is not shown again. In the app choose Connect and paste the server "
                   "address and this token, or open one of the links below on the phone."))
        st.code(token, language=None)
        st.code(pairing_link(server, token), language=None)
        st.code(web_pairing_link(server, token), language=None)
        if st.button(t("Done, hide the token"), key="phone_hide"):
            st.session_state.pop(NEW_TOKEN_KEY, None)
            st.rerun()
