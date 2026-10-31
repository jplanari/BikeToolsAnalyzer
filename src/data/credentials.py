"""
App credentials for the providers (Strava, Wahoo, Polar): the client ID and secret of the API app the rider
registered. They can be entered in the app (saved in the local database) or, as before, written to
.streamlit/secrets.toml; what was entered in the app wins. No Streamlit.
"""

PROVIDERS = {
    "strava": {
        "label": "Strava",
        "portal": "https://www.strava.com/settings/api",
        "redirect": "http://localhost:8501/",
        "steps": "Create an API application and set its **Authorization Callback Domain** to `localhost`.",
    },
    "wahoo": {
        "label": "Wahoo",
        "portal": "https://developers.wahooligan.com",
        "redirect": "https://jplanari.github.io/BikeToolsAnalyzer/oauth/",
        "steps": "Register an app and add the redirect address below as its callback URL. Wahoo only accepts an "
                 "https address, so this is the forwarding page from the README (publish `/docs` with GitHub Pages, "
                 "and change the user and repository name if you forked the project).",
    },
    "polar": {
        "label": "Polar",
        "portal": "https://admin.polaraccesslink.com",
        "redirect": "http://localhost:8501/",
        "steps": "Register a client and add the redirect address below to it; it must be **identical** to the one "
                 "Polar has. Leave it empty to use the default redirect URL registered for the client.",
    },
}


def resolve(provider, db_row, secrets_section):
    """
    {'client_id', 'client_secret', 'redirect_uri', 'timezone', 'source'} for the provider, or None when neither
    the database row (entered in the app) nor the secrets.toml section has both a client ID and a secret.
    """
    default_redirect = PROVIDERS[provider]["redirect"]
    for source, sec in (("app", db_row), ("secrets", secrets_section)):
        if not sec:
            continue
        try:
            client_id, client_secret = str(sec["client_id"]).strip(), str(sec["client_secret"]).strip()
        except (KeyError, TypeError):
            continue
        if not client_id or not client_secret:
            continue
        redirect = sec.get("redirect_uri") if hasattr(sec, "get") else None
        if redirect is None:
            redirect = default_redirect
        return {"client_id": client_id, "client_secret": client_secret, "redirect_uri": str(redirect).strip(),
                "timezone": str((sec.get("timezone") if hasattr(sec, "get") else "") or ""), "source": source}
    return None
