"""Desktop login: a rider with a password is locked until it is typed in this browser session. Reloading the page or
opening another window asks again. The gate covers the app's screens, not the database file itself: someone with
access to the computer's files can still read them (use disk encryption for that)."""
import streamlit as st

from src.data import db
from src.i18n import t
from src.security import Throttle

UNLOCKED_KEY = "unlocked_riders"
_throttle = Throttle(limit=5, window=300)        # wrong passwords per rider name, shared by every browser session


def is_locked(user_name):
    return bool(user_name) and db.has_password(user_name) and user_name not in st.session_state.get(UNLOCKED_KEY, set())


def unlock(user_name):
    st.session_state.setdefault(UNLOCKED_KEY, set()).add(user_name)


def lock(user_name):
    st.session_state.get(UNLOCKED_KEY, set()).discard(user_name)


def try_unlock(user_name, password):
    """(ok, message) after checking a typed password; repeated failures are slowed down."""
    wait = _throttle.retry_after(user_name)
    if wait:
        return False, t("Too many wrong passwords. Try again in {s} seconds.", s=wait)
    if db.verify_user_password(user_name, password):
        _throttle.reset(user_name)
        unlock(user_name)
        return True, ""
    _throttle.fail(user_name)
    return False, t("Wrong password.")


def render_unlock(user_name):
    """The lock screen in the main area; the caller stops the page after it."""
    st.header(t("🔒 {name} is protected", name=user_name))
    with st.form("unlock_form"):
        password = st.text_input(t("Password"), type="password", key="unlock_password")
        if st.form_submit_button(t("Unlock")):
            ok, message = try_unlock(user_name, password)
            if ok:
                st.rerun()
            else:
                st.error(message)


def render_password_settings(user_name):
    """User Corner: set, change or remove the rider's password."""
    st.subheader(t("🔒 Password"))
    protected = db.has_password(user_name)
    st.caption(t("A password keeps this rider's data closed until it is typed in, here and on the phone app. It does "
               "not encrypt the files on this computer.") if protected else
               t("Optional. With a password this rider's data stays closed until it is typed in, here and on the "
                 "phone app. It does not encrypt the files on this computer."))
    with st.form(f"password_form_{user_name}", clear_on_submit=True):
        current = st.text_input(t("Current password"), type="password") if protected else None
        new = st.text_input(t("New password"), type="password")
        again = st.text_input(t("Repeat the new password"), type="password")
        c1, c2 = st.columns(2)
        save = c1.form_submit_button(t("Change password") if protected else t("Set password"))
        remove = c2.form_submit_button(t("Remove password")) if protected else False
    if save:
        if new != again:
            st.error(t("The two passwords differ."))
        else:
            ok, message = db.set_user_password(user_name, new, current)
            (st.success if ok else st.error)(message)
    elif remove:
        ok, message = db.clear_user_password(user_name, current)
        (st.success if ok else st.error)(message)
        if ok:
            st.rerun()
