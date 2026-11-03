"""Sign-in for a shared server (BIKETOOLS_SERVER_MODE=1): the screens before the app, and the rider's account box.

`gate()` is called at the top of every run. It returns the signed-in rider's name, or draws the sign-in / sign-up /
forgot-password / reset screens and stops the run. The session lives in a random token kept in a browser cookie (so a
page reload or a new tab stays signed in) and in `st.session_state`; the database only stores the token's hash.
"""
import json

import streamlit as st
import streamlit.components.v1 as components

from src import serverconf
from src.data.db import accounts
from src.i18n import t

COOKIE = "biketools_session"
TOKEN_KEY = "auth_token"
USER_KEY = "auth_user"


def _client():
    try:
        return serverconf.client_address(dict(st.context.headers))
    except Exception:
        return None


def _cookie_token():
    try:
        return st.context.cookies.get(COOKIE)
    except Exception:
        return None


def _write_cookie(value, seconds):
    """The page cannot set a cookie from Python: a zero-height component does it in the browser."""
    parts = f"{COOKIE}={value}; path=/; max-age={int(seconds)}; SameSite=Lax"
    script = ("<script>try{var s=location.protocol==='https:'||window.parent.location.protocol==='https:';"
              f"window.parent.document.cookie={json.dumps(parts)}+(s?'; Secure':'');}}catch(e){{}}</script>")
    components.html(script, height=0)


def _sign_in(token, name):
    st.session_state[TOKEN_KEY], st.session_state[USER_KEY] = token, name
    st.session_state["current_user"] = name
    _write_cookie(token, accounts.SESSION_TTL_S)


def _forget():
    for key in (TOKEN_KEY, USER_KEY, "current_user", "lang_user"):
        st.session_state.pop(key, None)
    _write_cookie("", 0)


def current_user():
    """The rider of this browser session, or None. A session ended elsewhere (sign out everywhere, a deleted
    account) stops working here on the next run."""
    token = st.session_state.get(TOKEN_KEY) or _cookie_token()
    if not token:
        return None
    name = accounts.session_user(token)
    if not name:
        if TOKEN_KEY in st.session_state:
            _forget()
        return None
    st.session_state[TOKEN_KEY], st.session_state[USER_KEY] = token, name
    st.session_state["current_user"] = name
    return name


# --- the screens before the app -----------------------------------------------------------------------------------

def _handle_links():
    """?verify=<token> and ?reset=<token> from the e-mails. Returns the reset token to ask a new password for."""
    params = st.query_params
    if "verify" in params:
        ok, msg = accounts.verify_email(params.get("verify"))
        st.query_params.clear()
        (st.success if ok else st.error)(msg)
    if "reset" in params:
        return params.get("reset")
    return None


def _reset_screen(token):
    st.subheader(t("Choose a new password"))
    with st.form("reset_form"):
        new = st.text_input(t("New password"), type="password")
        again = st.text_input(t("Repeat the new password"), type="password")
        if st.form_submit_button(t("Save password"), type="primary"):
            if new != again:
                st.error(t("The two passwords are not the same."))
            else:
                ok, msg = accounts.reset_password(token, new)
                if ok:
                    st.query_params.clear()
                    st.success(msg)
                else:
                    st.error(msg)


def _login_tab():
    with st.form("login_form"):
        email = st.text_input(t("E-mail"))
        password = st.text_input(t("Password"), type="password")
        go = st.form_submit_button(t("Sign in"), type="primary")
    if go:
        status, token, name = accounts.login(email, password, client=_client())
        if status == accounts.OK:
            _sign_in(token, name)
            st.rerun()
        elif status == accounts.UNVERIFIED:
            st.warning(t("Confirm your e-mail address first: open the link we sent you."))
            if st.button(t("Send the link again"), key="resend_verify"):
                accounts.resend_verification(email, client=_client())
                st.info(t("If that address has an account waiting for confirmation, a new link is on its way."))
        elif status == accounts.THROTTLED:
            st.error(t("Too many attempts. Wait a while and try again."))
        else:
            st.error(t("E-mail or password not right."))


def _signup_tab():
    with st.form("signup_form"):
        name = st.text_input(t("Rider name"), help=t("Shown in the app. 2 to 40 characters."))
        email = st.text_input(t("E-mail"))
        password = st.text_input(t("Password"), type="password",
                                 help=t("At least {n} characters.", n=accounts.MIN_PASSWORD))
        go = st.form_submit_button(t("Create account"), type="primary")
    if go:
        ok, msg = accounts.register(name, email, password, client=_client())
        (st.success if ok else st.error)(msg)


def _forgot_tab():
    with st.form("forgot_form"):
        email = st.text_input(t("E-mail"), key="forgot_email")
        go = st.form_submit_button(t("Send me a link"))
    if go:
        accounts.request_password_reset(email, client=_client())
        st.info(t("If that address has an account, a link to choose a new password is on its way."))


def _screens():
    st.title("🚴 BikeTools Analyzer")
    reset = _handle_links()
    if reset:
        _reset_screen(reset)
        return
    login, signup, forgot = st.tabs([t("Sign in"), t("Create account"), t("Forgot password")])
    with login:
        _login_tab()
    with signup:
        _signup_tab()
    with forgot:
        _forgot_tab()


def gate():
    """The signed-in rider's name; otherwise draws the sign-in screens and stops this run."""
    name = current_user()
    if name:
        return name
    _screens()
    st.stop()


# --- the account box in the sidebar -------------------------------------------------------------------------------

def account_box(user_name):
    """Who is signed in, and sign out / change password / delete account."""
    info = accounts.account_info(user_name) or {}
    st.sidebar.markdown(f"**{user_name}**")
    if info.get("email"):
        st.sidebar.caption(info["email"])
    token = st.session_state.get(TOKEN_KEY)
    if st.sidebar.button(t("Sign out"), key="sign_out"):
        accounts.logout(token)
        _forget()
        st.rerun()
    with st.sidebar.expander(t("Account")):
        with st.form("change_password_form"):
            current = st.text_input(t("Current password"), type="password")
            new = st.text_input(t("New password"), type="password")
            if st.form_submit_button(t("Change password")):
                ok, msg = accounts.change_password(user_name, current, new, keep_token=token)
                (st.success if ok else st.error)(msg)
        if st.button(t("Sign out everywhere"), key="sign_out_all"):
            accounts.logout_everywhere(user_name)
            _forget()
            st.rerun()
        st.markdown("---")
        st.warning(t("Deleting your account removes you with all your rides, records and plans. It cannot be undone."))
        password = st.text_input(t("Password"), type="password", key="delete_account_password")
        if st.button(t("Delete my account"), type="primary", key="delete_account"):
            ok, msg = accounts.delete_account(user_name, password)
            if ok:
                _forget()
                st.rerun()
            else:
                st.error(msg)
