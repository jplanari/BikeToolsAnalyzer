"""Setting up the Strava / Wahoo / Polar API apps inside the app, instead of editing .streamlit/secrets.toml."""
import streamlit as st

from src.data import credentials as cr
from src.data import db


def _secrets_section(provider):
    try:
        sec = st.secrets[provider]
        return {k: sec[k] for k in sec}
    except Exception:                 # no secrets file, or no such section
        return None


def get_credentials(provider):
    """What the app uses for this provider (see src.data.credentials.resolve), or None when not set up."""
    return cr.resolve(provider, db.get_app_credentials(provider), _secrets_section(provider))


def credentials_form(provider, where="sidebar"):
    """Client ID / secret / redirect form for one provider; saves to the database and reruns."""
    info = cr.PROVIDERS[provider]
    current = get_credentials(provider) or {}
    box = st.sidebar if where == "sidebar" else st
    with box.form(f"cred_form_{provider}_{where}"):
        st.markdown(f"1. Open [{info['portal'].split('//')[1]}]({info['portal']}). {info['steps']}")
        st.markdown("2. Paste what it gives you:")
        cid = st.text_input("Client ID", value=current.get("client_id", ""), key=f"cred_{provider}_id_{where}")
        secret = st.text_input("Client secret", value=current.get("client_secret", ""), type="password",
                               key=f"cred_{provider}_secret_{where}")
        redirect = st.text_input("Redirect address", value=current.get("redirect_uri", info["redirect"]),
                                 key=f"cred_{provider}_redirect_{where}",
                                 help="Must be exactly the address registered for the app.")
        tz = ""
        if provider == "wahoo":
            tz = st.text_input("Time zone (optional)", value=current.get("timezone", ""),
                               key=f"cred_{provider}_tz_{where}", placeholder="Europe/Madrid",
                               help="Used to schedule planned sessions on your head unit. Empty: this computer's.")
        if st.form_submit_button("Save", type="primary"):
            if not cid.strip() or not secret.strip():
                st.error("Enter both the client ID and the client secret.")
            else:
                db.save_app_credentials(provider, cid, secret, redirect.strip(), tz.strip())
                st.rerun()


def setup_box(provider):
    """Shown in the sidebar when a provider has no credentials yet."""
    label = cr.PROVIDERS[provider]["label"]
    st.sidebar.warning(f"{label} isn't set up yet. Register an app with {label} and enter its credentials below "
                       "(one time).")
    with st.sidebar.expander(f"Set up {label}", expanded=True):
        credentials_form(provider, where="sidebar")


def render_connections():
    """User Corner: where each provider's credentials come from, and a way to change or remove them."""
    st.subheader("🔌 Connections")
    st.caption("The API app each service needs (a client ID and secret you get when you register one). They are "
               "saved on this computer, in the app's database.")
    for provider, info in cr.PROVIDERS.items():
        current = get_credentials(provider)
        saved = db.get_app_credentials(provider)
        status = ("not set up" if not current else
                  "saved in this app" if current["source"] == "app" else "read from `.streamlit/secrets.toml`")
        with st.expander(f"{info['label']}: {status}"):
            if current and current["source"] == "secrets":
                st.caption("Saving here overrides the secrets file.")
            if current:
                st.caption(f"Client ID: `{current['client_id']}`")
                st.warning("Changing the client ID disconnects riders already connected through the old app: "
                           "they would press Connect again.")
            credentials_form(provider, where="corner")
            if saved and st.button(f"Remove saved {info['label']} credentials", key=f"cred_remove_{provider}"):
                db.delete_app_credentials(provider)
                st.rerun()
