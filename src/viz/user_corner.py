"""User Corner page: profile summary and ride-history maintenance. The ride history table is render_history."""
import streamlit as st
import pandas as pd
from src.data.history import recompute_history
from src.data.db import set_user_thresholds, DEFAULT_FTP, DEFAULT_LTHR
from src.viz import theme
from src.data.cached import get_user_data, get_user_rides, count_rides_needing_update
from src.viz.formatting import format_duration


def render_history(user_name):
    """Renders the history table purely for viewing stats."""
    st.header("📜 Ride History")
    
    if not user_name:
        st.warning("Please select a user in the sidebar to view their history.")
        return

    history = get_user_rides(user_name)
    if not history:
        st.info("No rides saved yet.")
        return

    hist_df = pd.DataFrame(history)
    st.info("Here is a log of your past rides.")

    cols = ['date_time', 'filename', 'distance_km', 'elevation_m', 'moving_s', 'norm_power',
            'tss', 'tss_source', 'vi', 'ef', 'decoupling_pct']
    view = hist_df[[c for c in cols if c in hist_df.columns]].copy()
    if 'moving_s' in view.columns:
        view['moving_s'] = view['moving_s'].map(lambda x: format_duration(x) if pd.notna(x) else "")
    st.dataframe(
        view.rename(columns={'date_time': 'Date', 'filename': 'File', 'distance_km': 'km',
                             'elevation_m': 'Elev (m)', 'moving_s': 'Moving', 'norm_power': 'NP',
                             'tss': 'TSS', 'tss_source': 'TSS from', 'vi': 'VI', 'ef': 'EF',
                             'decoupling_pct': 'Decoupling %'}),
        hide_index=True,
        column_config={"km": st.column_config.NumberColumn(format="%.1f"),
                       "VI": st.column_config.NumberColumn(format="%.2f"),
                       "EF": st.column_config.NumberColumn(format="%.2f"),
                       "Decoupling %": st.column_config.NumberColumn(format="%.1f")},
    )


def _render_history_update(user_name, user_data):
    """Offers to rebuild outdated rides from their stored files."""
    n = count_rides_needing_update(user_name)
    if n == 0:
        return
    with st.expander(f"🛠️ Update ride history ({n} rides need updating)"):
        st.caption("These rides were saved by an older version of the app, so they are missing "
                   "newer metrics (moving time, VI, EF, decoupling, zone times, the full power "
                   "curve) or their date. This re-reads the stored files and recomputes them. "
                   "Each ride keeps the FTP it was originally scored with; rides without power "
                   "get an HR-based TSS using your LTHR.")
        if st.button("Update ride history"):
            bar = st.progress(0.0, text="Recomputing rides...")
            updated, failed = recompute_history(
                user_name, user_data['ftp'], user_data['lthr'],
                progress=lambda done, total: bar.progress(done / total if total else 1.0,
                                                          text=f"{done}/{total} rides"))
            bar.empty()
            if failed:
                st.warning(f"Updated {updated} rides; {len(failed)} could not be read:\n\n"
                           + "\n".join(f"- {f}" for f in failed[:10]))
            else:
                st.toast(f"✅ Updated {updated} rides.")
                st.rerun()


def _by_hand(new, old, auto):
    """The value to store as a manual choice; an untouched estimated value keeps following the data."""
    return int(new) if int(new) != int(old) or not auto else None


def _threshold_controls(user_name, user):
    """Says where FTP / LTHR come from and lets the rider set them by hand or hand them back to the data."""
    notes = [f"{label} is estimated from your rides (95 % of the best 20 minutes; {default} until a ride has one)"
             for label, default, auto in (("FTP", f"{DEFAULT_FTP} W", user.get('ftp_auto')),
                                          ("LTHR", f"{DEFAULT_LTHR} bpm", user.get('lthr_auto'))) if auto]
    if notes:
        st.caption("ℹ️ " + "; ".join(notes) + ".")
    with st.popover("Set FTP / LTHR"):
        st.markdown(theme.FTP_HELP)
        st.markdown(theme.LTHR_HELP)
        with st.form(f"thresholds_{user_name}"):
            ftp = st.number_input("FTP (Watts)", value=int(user['ftp']), min_value=1, step=5)
            lthr = st.number_input("LTHR (bpm)", value=int(user['lthr']), min_value=1, step=1)
            if st.form_submit_button("Save by hand"):
                set_user_thresholds(user_name, ftp=_by_hand(ftp, user['ftp'], user.get('ftp_auto')),
                                    lthr=_by_hand(lthr, user['lthr'], user.get('lthr_auto')))
                st.rerun()
        a1, a2 = st.columns(2)
        if a1.button("Estimate FTP from rides", key=f"auto_ftp_{user_name}", disabled=bool(user.get('ftp_auto'))):
            set_user_thresholds(user_name, auto_ftp=True)
            st.rerun()
        if a2.button("Estimate LTHR from rides", key=f"auto_lthr_{user_name}", disabled=bool(user.get('lthr_auto'))):
            set_user_thresholds(user_name, auto_lthr=True)
            st.rerun()


def render_user_corner(user_name):
    st.header("👤 User Corner")

    if not user_name:
        st.warning("Please select a user in the sidebar to view profile details.")
        return
        
    user_data = get_user_data(user_name)
    if not user_data:
        st.error("User data not found.")
        return

    # 1. Profile Summary
    c1, c2, c3 = st.columns(3)
    c1.metric("FTP", f"{user_data['ftp']} W")
    c2.metric("LTHR", f"{user_data['lthr']} bpm")
    c3.metric("Weight", f"{user_data['weight']} kg")
    _threshold_controls(user_name, user_data)
    
    st.markdown("---")

    _render_history_update(user_name, user_data)

    st.info("Charts and analytics are on the **Training** page, including your critical-power model and the "
            "buttons that save it to your profile.")
