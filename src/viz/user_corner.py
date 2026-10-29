"""User Corner page: profile summary and ride-history maintenance. The ride history table is render_history."""
import streamlit as st
import pandas as pd
from src.data.history import recompute_history
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
    
    st.markdown("---")

    _render_history_update(user_name, user_data)

    st.info("Charts and analytics are on the **Training** page, including your critical-power model and the "
            "buttons that save it to your profile.")
