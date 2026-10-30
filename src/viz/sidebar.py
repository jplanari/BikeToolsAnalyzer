"""Sidebar: navigation, user profile, data import and the Strava / Wahoo connections."""
import streamlit as st
import pandas as pd
import io
from src.data.db import (save_user,
                         get_all_users,
                         delete_user,
                         get_strava_auth,
                         save_strava_auth,
                         delete_strava_auth,
                         get_wahoo_auth,
                         save_wahoo_auth,
                         delete_wahoo_auth,
                         get_polar_auth,
                         delete_polar_auth,
                         create_oauth_state,
                         consume_oauth_state,
                         data_stamp)
from src.viz import theme
from src.data.upload import expand_upload, UPLOAD_TYPES
from src.data.upload_import import import_activities
from src.data.thresholds import backfill_hr20, refresh_auto_thresholds
from src.data.cached import get_user_data
from src.data import strava as sv
from src.data import wahoo as wh
from src.data import wahoo_import as wi
from src.data import polar as pl
from src.data import polar_import as pi
from src.data.strava_import import (list_recent_rides,
                                    list_rides_between,
                                    fetch_ride_file,
                                    import_history,
                                    sync_new_rides)


def _follow_the_data(user_name):
    """FTP / LTHR left empty are re-estimated from the rider's rides (old rides get their HR peak once)."""
    user = get_user_data(user_name)
    if not user or not (user.get('ftp_auto') or user.get('lthr_auto')):
        return
    stamp_key = f"thresholds_stamp_{user_name}"
    if st.session_state.get(stamp_key) == data_stamp():          # nothing was saved since the last estimate
        return
    if user.get('lthr_auto') and not st.session_state.get(f"hr20_done_{user_name}"):
        with st.spinner("Estimating LTHR from your rides…"):
            if backfill_hr20(user_name) == 0:
                st.session_state[f"hr20_done_{user_name}"] = True     # nothing left to read: stop looking
    refresh_auto_thresholds(user_name)
    st.session_state[stamp_key] = data_stamp()


def render_sidebar():
    """Renders the sidebar and returns all user configuration settings."""
    
    user_ftp = 200
    user_weight = 70.0
    user_lthr = 170
    user_height = 175

    pages = ["Analyze Upload", "Training", "User Corner"]
    app_mode = st.radio("Go to", pages, horizontal=True, key="nav_top", label_visibility="collapsed")

    st.sidebar.markdown("---")
    st.sidebar.header("👤 User Profile")

    # User Management
    tab_select, tab_create = st.sidebar.tabs(["Select User", "Create/Edit"])
    
    selected_user_name = None
    
    with tab_select:
        users = get_all_users()
        if users:
            selected_user_name = st.selectbox("Current User", users, key="current_user")
            if selected_user_name:
                _follow_the_data(selected_user_name)
                u_data = get_user_data(selected_user_name)
                if u_data:
                    user_ftp = u_data['ftp']
                    user_lthr = u_data['lthr']
                    user_weight = u_data['weight']
                    user_height = u_data['height']


            with st.popover("Delete user"):
                st.warning(f"This removes **{selected_user_name}** with all their rides, records and plans. "
                           "It cannot be undone.")
                if st.button(f"Yes, delete {selected_user_name}", type="primary", key="confirm_delete_user"):
                    delete_user(selected_user_name)
                    st.rerun()
        else:
            st.info("No users found. Create one!")

    with tab_create:
        with st.form("user_form"):
            new_name = st.text_input("Name")
            new_ftp = st.number_input("FTP (Watts) · optional", value=None, min_value=1, step=5,
                                      placeholder="estimated from your rides", help=theme.FTP_HELP)
            new_lthr = st.number_input("LTHR (bpm) · optional", value=None, min_value=1, step=1,
                                       placeholder="estimated from your rides", help=theme.LTHR_HELP)
            new_weight = st.number_input("Weight (kg)", value=70.0)
            new_height = st.number_input("Height (cm)", value=175)
            submitted = st.form_submit_button("Save User")
            if submitted and new_name:
                success, msg = save_user(new_name, int(new_ftp) if new_ftp else None,
                                          int(new_lthr) if new_lthr else None, new_weight, new_height)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    st.sidebar.markdown("---")
    st.sidebar.header("📂 Data Import")
    
    # 1. Initialize current_file to None by default
    current_file = None
    
    import_method = st.sidebar.radio("Source", ["Upload Ride", "Strava", "Wahoo", "Polar"])

    if import_method == "Upload Ride":
        uploaded = st.sidebar.file_uploader(
            "Upload rides", type=UPLOAD_TYPES, accept_multiple_files=True,
            help="FIT, GPX or TCX files from Garmin, Hammerhead, Magene, Wahoo, Zwift and others, or a zip / gz of "
                 "them (Garmin's bulk export works). One file opens for analysis; several are imported to your history.")
        current_file = _render_upload(uploaded, selected_user_name, user_ftp, user_lthr)

    elif import_method == "Strava":
        current_file = _render_strava_sidebar(selected_user_name, user_ftp, user_lthr)

    elif import_method == "Wahoo":
        current_file = _render_wahoo_sidebar(selected_user_name, user_ftp, user_lthr)

    elif import_method == "Polar":
        current_file = _render_polar_sidebar(selected_user_name, user_ftp, user_lthr)

    # Every section of the ride analysis is always shown; only the profile values are passed on
    settings = {
        'ftp': user_ftp,
        'lthr': user_lthr,
        'weight': user_weight,
        'height': user_height
    }

    st.sidebar.markdown("---")
    theme.appearance_controls()

    # RETURN 'current_file' (which works for both logic paths)
    return app_mode, selected_user_name, current_file, settings


def _strava_config():
    """App credentials from .streamlit/secrets.toml ([strava] section), or None."""
    try:
        sec = st.secrets["strava"]
        return sv.StravaConfig(
            client_id=str(sec["client_id"]),
            client_secret=str(sec["client_secret"]),
            redirect_uri=str(sec.get("redirect_uri", "http://localhost:8501/")),
        )
    except Exception:
        return None


def _strava_usage():
    if 'strava_usage' not in st.session_state:
        st.session_state['strava_usage'] = sv.RateUsage()
    return st.session_state['strava_usage']


def handle_strava_callback():
    """
    Completes the OAuth round trip when Strava redirects back to the app
    (?code=...&scope=...&state=...). Call before rendering the sidebar.
    """
    params = st.query_params
    if "state" not in params or not ("code" in params or "error" in params):
        return

    state, code = params.get("state"), params.get("code")
    error, scope = params.get("error"), params.get("scope")
    st.query_params.clear()  # the code is single-use; never process it twice

    if state.startswith(WAHOO_STATE_PREFIX):
        return _finish_wahoo_callback(state[len(WAHOO_STATE_PREFIX):], code, error)
    if state.startswith(POLAR_STATE_PREFIX):
        return _finish_polar_callback(state[len(POLAR_STATE_PREFIX):], code, error)

    user_name = consume_oauth_state(state)
    if not user_name:
        st.error("Strava connection failed: the request expired or did not come from this app. Please try again.")
        return
    st.session_state["current_user"] = user_name

    if error:
        st.warning("Strava access was not granted.")
        return
    if not sv.granted_scope_ok(scope):
        st.warning("Strava was connected without permission to read activities. "
                   "Please reconnect and keep the activity access box ticked.")
        return

    cfg = _strava_config()
    if cfg is None:
        st.error("Strava is not configured (see README).")
        return
    try:
        auth = sv.exchange_code(cfg, code)
    except sv.StravaError as e:
        st.error(f"Strava connection failed: {e}")
        return
    save_strava_auth(user_name, auth)
    st.toast(f"✅ Strava connected for {user_name}")


WAHOO_STATE_PREFIX = "wahoo."      # tells the shared redirect which provider is calling back


def _wahoo_config():
    """App credentials from .streamlit/secrets.toml ([wahoo] section), or None."""
    try:
        sec = st.secrets["wahoo"]
        return wh.WahooConfig(
            client_id=str(sec["client_id"]),
            client_secret=str(sec["client_secret"]),
            redirect_uri=str(sec.get("redirect_uri", "http://localhost:8501/")),
            timezone=str(sec.get("timezone", "")),
        )
    except Exception:
        return None


def _finish_wahoo_callback(state, code, error):
    user_name = consume_oauth_state(state)
    if not user_name:
        st.error("Wahoo connection failed: the request expired or did not come from this app. Please try again.")
        return
    st.session_state["current_user"] = user_name
    if error or not code:
        st.warning("Wahoo access was not granted.")
        return
    cfg = _wahoo_config()
    if cfg is None:
        st.error("Wahoo is not configured (see README).")
        return
    try:
        auth = wh.exchange_code(cfg, code)
    except wh.WahooError as e:
        st.error(f"Wahoo connection failed: {e}")
        return
    save_wahoo_auth(user_name, auth)
    st.toast(f"✅ Wahoo connected for {user_name}")


POLAR_STATE_PREFIX = "polar."


def _polar_config():
    """App credentials from .streamlit/secrets.toml ([polar] section), or None."""
    try:
        sec = st.secrets["polar"]
        return pl.PolarConfig(client_id=str(sec["client_id"]), client_secret=str(sec["client_secret"]),
                              redirect_uri=str(sec.get("redirect_uri", "http://localhost:8501/")))
    except Exception:
        return None


def _finish_polar_callback(state, code, error):
    user_name = consume_oauth_state(state)
    if not user_name:
        st.error("Polar connection failed: the request expired or did not come from this app. Please try again.")
        return
    st.session_state["current_user"] = user_name
    if error or not code:
        st.warning("Polar access was not granted.")
        return
    cfg = _polar_config()
    if cfg is None:
        st.error("Polar is not configured (see README).")
        return
    try:
        pi.connect(cfg, user_name, code)
    except pl.PolarError as e:
        st.error(f"Polar connection failed: {e}")
        return
    st.toast(f"✅ Polar connected for {user_name}")


def _polar_label(e):
    when = (e.get("start_time") or "")[:10]
    km = float(e.get("distance") or 0) / 1000
    return f"{when} · {(e.get('detailed_sport_info') or e.get('sport') or 'Ride').replace('_', ' ').title()} · {km:.0f} km · {pl.duration_minutes(e)} min"


def _render_polar_sidebar(user_name, user_ftp, user_lthr=None):
    """Returns a file-like ride (the FIT file Polar Flow holds) to analyse, or None."""
    cfg = _polar_config()
    if cfg is None:
        st.sidebar.warning("Polar isn't set up yet. Register a client at admin.polaraccesslink.com and add its "
                           "`client_id` and `client_secret` to `.streamlit/secrets.toml` (see README).")
        return None
    if not user_name:
        st.sidebar.info("Select or create a user profile first.")
        return None

    auth = get_polar_auth(user_name)
    if not auth:
        key = f"polar_oauth_state_{user_name}"
        if key not in st.session_state:
            st.session_state[key] = POLAR_STATE_PREFIX + create_oauth_state(user_name)
        st.sidebar.link_button("🔗 Connect Polar", pl.authorize_url(cfg, st.session_state[key]))
        st.sidebar.caption("Polar opens in a new tab; you'll continue there after approving.")
        return None

    st.sidebar.caption("Connected to **Polar Flow**")
    st.sidebar.caption(f"Polar only shares the last {pl.WINDOW_DAYS} days, and only rides recorded after you "
                       "connected: sync at least once a month.")
    try:
        cache_key = f"polar_recent_{user_name}"
        if cache_key not in st.session_state or st.sidebar.button("↻ Refresh list", key="polar_refresh"):
            with st.spinner("Loading your Polar rides..."):
                st.session_state[cache_key] = pi.list_recent_rides(user_name)
        rides = st.session_state[cache_key]

        if rides:
            choice = st.sidebar.selectbox("Recent rides", rides, format_func=_polar_label)
            if st.sidebar.button("Analyze ride", key="polar_analyze"):
                with st.spinner("Downloading ride from Polar..."):
                    name, payload = pi.fetch_ride_file(user_name, choice)
                file_obj = io.BytesIO(payload)
                file_obj.name = name
                st.session_state[f'polar_file_{user_name}'] = file_obj
        else:
            st.sidebar.info("No cycling rides found on Polar in the last 30 days.")

        if st.sidebar.button("⟳ Sync new rides", key="polar_sync",
                             help="Imports the rides Polar lists that are not in your history yet."):
            bar = st.sidebar.progress(0.0, text="Checking Polar...")

            def _sync_progress(done, total, name):
                bar.progress(done / total if total else 1.0, text=f"{done}/{total} {name}"[:60])

            res = pi.sync_new_rides(user_name, user_ftp, progress=_sync_progress, lthr=user_lthr)
            bar.empty()
            if res.imported == 0 and not res.failed and not res.stopped_reason:
                st.sidebar.success("Up to date.")
            else:
                _show_import_result(res)

    except pl.PolarAuthError as e:
        st.sidebar.error(f"{e}")
    except pl.PolarError as e:
        st.sidebar.error(f"Polar: {e}")

    if st.sidebar.button("Disconnect Polar", key="polar_disconnect"):
        pl.deregister_user(auth["access_token"], auth.get("polar_user_id"))
        delete_polar_auth(user_name)
        for k in [f"polar_recent_{user_name}", f"polar_file_{user_name}"]:
            st.session_state.pop(k, None)
        st.rerun()

    return st.session_state.get(f'polar_file_{user_name}')


def _wahoo_label(w):
    when = (w.get("starts") or "")[:10]
    return f"{when} · {w.get('name') or 'Ride'} · {w.get('minutes') or 0} min"


def _activity_label(a):
    when = (a.get("start_date_local") or a.get("start_date") or "")[:10]
    km = (a.get("distance") or 0) / 1000
    return f"{when} · {a.get('name', 'Ride')} · {km:.0f} km"


def _search_by_date(provider, user_name, search):
    """
    'Find a ride by date' box under the recent-rides list: any ride in a date range, not only the latest ones.
    search(start_date, end_date) -> rides, newest first. Returns the found rides (possibly empty) while a search is
    active, None otherwise; the result is kept in the session so the list survives reruns.
    """
    key = f"{provider}_search_{user_name}"
    with st.sidebar.expander("🔎 Find a ride by date"):
        today = pd.Timestamp.today().date()
        start = st.date_input("From", today - pd.Timedelta(days=90), max_value=today, key=f"{key}_from")
        end = st.date_input("To", today, max_value=today, key=f"{key}_to")
        c1, c2 = st.columns(2)
        if c1.button("Search", key=f"{key}_go"):
            if start > end:
                st.warning("'From' is after 'To'.")
            else:
                with st.spinner("Searching..."):
                    st.session_state[key] = {"rides": search(start, end), "range": (start, end)}
        if c2.button("Clear", key=f"{key}_clear"):
            st.session_state.pop(key, None)
    found = st.session_state.get(key)
    if found is None:
        return None
    a, b = found["range"]
    st.sidebar.caption(f"Showing {len(found['rides'])} ride(s) from {a:%d %b %Y} to {b:%d %b %Y}.")
    return found["rides"]


def _render_upload(uploaded, user_name, ftp, lthr):
    """One ride: returned for analysis. Several (or a zip of them): an import button that saves them all."""
    if not uploaded:
        return None
    activities, notes = [], []
    for f in uploaded:
        found, n = expand_upload(f.name, f.getvalue())
        activities.extend(found)
        notes.extend(n)
    for n in notes[:5]:
        st.sidebar.warning(n)
    if len(notes) > 5:
        st.sidebar.caption(f"…and {len(notes) - 5} more files skipped.")
    if not activities:
        return None
    if len(activities) == 1:
        file_obj = io.BytesIO(activities[0].data)
        file_obj.name = activities[0].name
        return file_obj

    if not user_name:
        st.sidebar.info("Select or create a user profile first.")
        return None
    signature = tuple((f.name, f.size) for f in uploaded)
    st.sidebar.info(f"{len(activities)} rides found.")
    if st.sidebar.button(f"Import {len(activities)} rides to history", key="bulk_import", type="primary"):
        bar = st.sidebar.progress(0.0, text="Importing…")
        res = import_activities(user_name, activities, ftp, lthr,
                                progress=lambda done, total, name: bar.progress(done / total if total else 1.0,
                                                                                text=f"{done}/{total} {name}"))
        bar.empty()
        st.session_state["upload_import_result"] = (signature, res)
    done = st.session_state.get("upload_import_result")
    if done and done[0] == signature:
        _show_import_result(done[1])
    return None


def _show_import_result(res):
    """Summary of an ImportResult in the sidebar."""
    st.sidebar.success(f"Imported {res.imported} of {res.found} rides.")
    skipped = []
    if res.already_imported: skipped.append(f"{res.already_imported} already imported")
    if res.duplicates_of_uploads: skipped.append(f"{res.duplicates_of_uploads} matched uploaded files")
    if res.no_data: skipped.append(f"{res.no_data} without recorded data or not cycling")
    if res.not_completed: skipped.append(f"{res.not_completed} planned but not ridden")
    if skipped:
        st.sidebar.caption("Skipped: " + ", ".join(skipped) + ".")
    if res.failed:
        st.sidebar.warning(f"{len(res.failed)} failed:\n\n" + "\n".join(f"- {f}" for f in res.failed[:10]))
    if res.stopped_reason:
        st.sidebar.info(f"{res.stopped_reason} ({res.remaining} left)")


def _render_strava_sidebar(user_name, user_ftp, user_lthr=None):
    """Returns a file-like ride to analyse, or None."""
    cfg = _strava_config()
    if cfg is None:
        st.sidebar.warning("Strava isn't set up yet. Add your API app's `client_id` and "
                           "`client_secret` to `.streamlit/secrets.toml` (see README).")
        return None
    if not user_name:
        st.sidebar.info("Select or create a user profile first.")
        return None

    auth = get_strava_auth(user_name)
    if not auth:
        key = f"oauth_state_{user_name}"
        if key not in st.session_state:
            st.session_state[key] = create_oauth_state(user_name)
        st.sidebar.link_button("🔗 Connect Strava", sv.authorize_url(cfg, st.session_state[key]))
        st.sidebar.caption("Strava opens in a new tab; you'll continue there after approving.")
        return None

    who = auth.get("athlete_name") or "your account"
    st.sidebar.caption(f"Connected to Strava as **{who}**")
    usage = _strava_usage()
    current_file = None

    try:
        # Recent rides (cached per session to spare the rate limit)
        cache_key = f"strava_recent_{user_name}"
        if cache_key not in st.session_state or st.sidebar.button("↻ Refresh list"):
            with st.spinner("Loading your Strava rides..."):
                st.session_state[cache_key] = list_recent_rides(cfg, user_name, usage=usage)
        recent = st.session_state[cache_key]
        found = _search_by_date("strava", user_name,
                                lambda a, b: list_rides_between(cfg, user_name, a, b, usage=usage))
        rides = recent if found is None else found

        if rides:
            choice = st.sidebar.selectbox("Recent rides" if found is None else "Rides found", rides,
                                          format_func=_activity_label)
            if st.sidebar.button("Analyze ride"):
                with st.spinner("Downloading ride from Strava..."):
                    name, payload = fetch_ride_file(cfg, user_name, choice, usage=usage)
                file_obj = io.BytesIO(payload)
                file_obj.name = name
                st.session_state[f'strava_file_{user_name}'] = file_obj
        else:
            st.sidebar.info("No recent rides found on Strava." if found is None else "No rides in that period.")

        if st.sidebar.button("⟳ Sync new rides", help="Imports rides added to Strava since the last one "
                             "you imported (a few days of overlap catch late uploads)."):
            bar = st.sidebar.progress(0.0, text="Checking Strava...")

            def _sync_progress(done, total, name):
                bar.progress(done / total if total else 1.0, text=f"{done}/{total} {name}"[:60])

            res, since_dt = sync_new_rides(cfg, user_name, user_ftp, progress=_sync_progress, usage=usage,
                                           lthr=user_lthr)
            bar.empty()
            if res.imported == 0 and not res.failed and not res.stopped_reason:
                st.sidebar.success(f"Up to date (checked since {since_dt:%d %b %Y}).")
            else:
                _show_import_result(res)

        with st.sidebar.expander("📥 Import ride history"):
            st.caption("Imports every ride since the chosen date into your history "
                       "(PMC, power records). Rides already saved are skipped.")
            since = st.date_input("Since", value=pd.Timestamp.today().date() - pd.Timedelta(days=90))
            if st.button("Import history"):
                bar = st.progress(0.0, text="Listing activities...")

                def _progress(done, total, name):
                    frac = done / total if total else 1.0
                    bar.progress(frac, text=f"{done}/{total} {name}"[:60])

                res = import_history(cfg, user_name, since, user_ftp, progress=_progress, usage=usage,
                                     lthr=user_lthr)
                bar.empty()
                _show_import_result(res)

    except sv.StravaAuthError as e:
        st.sidebar.error(f"{e}")
    except sv.StravaError as e:
        st.sidebar.error(f"Strava: {e}")

    if st.sidebar.button("Disconnect Strava"):
        sv.revoke(cfg, auth["refresh_token"])
        delete_strava_auth(user_name)
        for k in [f"strava_recent_{user_name}", f"strava_file_{user_name}"]:
            st.session_state.pop(k, None)
        st.rerun()

    # Keyed per profile so a ride is never auto-saved to the wrong rider
    current_file = st.session_state.get(f'strava_file_{user_name}', current_file)
    return current_file


def _render_wahoo_sidebar(user_name, user_ftp, user_lthr=None):
    """Returns a file-like ride (the original FIT file) to analyse, or None."""
    cfg = _wahoo_config()
    if cfg is None:
        st.sidebar.warning("Wahoo isn't set up yet. Register an app at developers.wahooligan.com and add its "
                           "`client_id` and `client_secret` to `.streamlit/secrets.toml` (see README).")
        return None
    if not user_name:
        st.sidebar.info("Select or create a user profile first.")
        return None

    auth = get_wahoo_auth(user_name)
    if not auth:
        key = f"wahoo_oauth_state_{user_name}"
        if key not in st.session_state:
            st.session_state[key] = WAHOO_STATE_PREFIX + create_oauth_state(user_name)
        st.sidebar.link_button("🔗 Connect Wahoo", wh.authorize_url(cfg, st.session_state[key]))
        st.sidebar.caption("Wahoo opens in a new tab; you'll continue there after approving.")
        return None

    st.sidebar.caption(f"Connected to Wahoo as **{auth.get('wahoo_name') or 'your account'}**")
    usage = st.session_state.setdefault('wahoo_usage', wh.RateUsage())

    try:
        cache_key = f"wahoo_recent_{user_name}"
        if cache_key not in st.session_state or st.sidebar.button("↻ Refresh list", key="wahoo_refresh"):
            with st.spinner("Loading your Wahoo rides..."):
                st.session_state[cache_key] = wi.list_recent_rides(cfg, user_name, usage=usage)
        recent = st.session_state[cache_key]
        found = _search_by_date("wahoo", user_name,
                                lambda a, b: wi.list_rides_between(cfg, user_name, a, b, usage=usage))
        rides = recent if found is None else found

        if rides:
            choice = st.sidebar.selectbox("Recent rides" if found is None else "Rides found", rides,
                                          format_func=_wahoo_label)
            if st.sidebar.button("Analyze ride", key="wahoo_analyze"):
                with st.spinner("Downloading ride from Wahoo..."):
                    name, payload = wi.fetch_ride_file(cfg, user_name, choice, usage=usage)
                file_obj = io.BytesIO(payload)
                file_obj.name = name
                st.session_state[f'wahoo_file_{user_name}'] = file_obj
        else:
            st.sidebar.info("No recent cycling workouts found on Wahoo." if found is None
                            else "No rides in that period.")

        if st.sidebar.button("⟳ Sync new rides", key="wahoo_sync",
                             help="Imports workouts added to Wahoo since the last one you imported."):
            bar = st.sidebar.progress(0.0, text="Checking Wahoo...")

            def _sync_progress(done, total, name):
                bar.progress(done / total if total else 1.0, text=f"{done}/{total} {name}"[:60])

            res, since_dt = wi.sync_new_rides(cfg, user_name, user_ftp, progress=_sync_progress, usage=usage,
                                              lthr=user_lthr)
            bar.empty()
            if res.imported == 0 and not res.failed and not res.stopped_reason:
                st.sidebar.success(f"Up to date (checked since {since_dt:%d %b %Y}).")
            else:
                _show_import_result(res)

        with st.sidebar.expander("📥 Import ride history"):
            st.caption("Imports every cycling workout since the chosen date into your history. Rides already "
                       "saved, or starting within 2 minutes of an uploaded or Strava ride, are skipped.")
            since = st.date_input("Since", value=pd.Timestamp.today().date() - pd.Timedelta(days=90),
                                  key="wahoo_since")
            if st.button("Import history", key="wahoo_history"):
                bar = st.progress(0.0, text="Listing workouts...")

                def _progress(done, total, name):
                    bar.progress(done / total if total else 1.0, text=f"{done}/{total} {name}"[:60])

                res = wi.import_history(cfg, user_name, since, user_ftp, progress=_progress, usage=usage,
                                        lthr=user_lthr)
                bar.empty()
                _show_import_result(res)

    except wh.WahooAuthError as e:
        st.sidebar.error(f"{e}")
    except wh.WahooError as e:
        st.sidebar.error(f"Wahoo: {e}")

    if st.sidebar.button("Disconnect Wahoo", key="wahoo_disconnect"):
        try:
            wh.revoke(wi.get_valid_auth(cfg, user_name)["access_token"])
        except wh.WahooError:
            pass
        delete_wahoo_auth(user_name)
        for k in [f"wahoo_recent_{user_name}", f"wahoo_file_{user_name}"]:
            st.session_state.pop(k, None)
        st.rerun()

    return st.session_state.get(f'wahoo_file_{user_name}')
