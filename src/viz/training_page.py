"""Training page: load, power profile, durability and intensity distribution."""
import json

import pandas as pd
import streamlit as st

from src.data.db import (replace_planned_sessions, add_structured_session as db_add_session,
                         delete_planned_session, update_user_ftp)
from src.data.cached import (get_user_data, get_user_tss_history, get_ride_stats, get_power_records,
                             get_kj_records, get_zone_rows, get_planned_sessions,
                             count_rides_needing_update, compute_pmc, cp_history)
from src.analysis import training_load as tl
from src.analysis import power_profile as pp
from src.analysis import durability as du
from src.analysis import intensity as it
from src.analysis import workout as wo
from src.analysis.power import estimate_critical_power
from src.viz import training_plots as tp
from src.viz.sections import section_picker, remembered
from src.analysis.labels import (PERIODS, period_start, dur_label, ramp_status, acwr_status, form_status,
                                 monotony_status, tid_label)

def _theme():
    try:
        return st.context.theme.type or 'light'
    except Exception:
        return 'light'


def _chart(fig):
    st.plotly_chart(fig, use_container_width=True)


_period_start, _dur_label = period_start, dur_label


def _metric(col, label, value, status, help_text):
    col.metric(label, value, delta=status, delta_color="off", help=help_text)


# --- Page --------------------------------------------------------------------

def render_training_page(user_name):
    st.header("🏋️ Training")
    if not user_name:
        st.warning("Please select a user in the sidebar.")
        return
    user = get_user_data(user_name)
    if not user:
        st.error("User data not found.")
        return

    n_old = count_rides_needing_update(user_name)
    if n_old:
        st.warning(f"{n_old} rides were saved by an older version and are missing data used here. "
                   "Run **Update ride history** in the User Corner to include them.")

    today = pd.Timestamp.today().normalize()
    mode = _theme()
    _threshold_hint(user_name, user, today)
    _report_expander(user_name, today)
    sections = ["📈 Load", "⚡ Power profile", "🔋 Durability", "🎯 Intensity"]
    section = section_picker(sections, key="training_section")
    if section == sections[0]:
        _render_load(user_name, user['ftp'], today, mode)
    elif section == sections[1]:
        _render_power(user_name, user, today, mode)
    elif section == sections[2]:
        _render_durability(user_name, today, mode)
    else:
        _render_intensity(user_name, today, mode)


def _report_expander(user_name, today):
    """Builds the PDF only on click (matplotlib is slow-ish) and keeps it for the download button."""
    with st.expander("📄 PDF report"):
        label = st.selectbox("Period", list(PERIODS), index=1, key="report_period")
        key = (user_name, label, today)
        if st.button("Create report", key="report_make"):
            from src.data.report_data import collect_report_data
            from src.viz.report_pdf import build_report_pdf
            with st.spinner("Building the report…"):
                pdf = build_report_pdf(collect_report_data(user_name, PERIODS[label], today))
            st.session_state["report_pdf"] = (key, pdf)
        ready = st.session_state.get("report_pdf")
        if ready and ready[0] == key:
            st.download_button("⬇️ Download PDF", ready[1], mime="application/pdf",
                               file_name=f"training-report-{user_name}-{today:%Y-%m-%d}.pdf", key="report_dl")
        elif ready:
            st.caption("Period or user changed: create the report again.")


def _threshold_hint(user_name, user, today):
    """Suggests a new FTP when recent riding supports it; nothing changes until you accept."""
    s = pp.suggest_threshold(get_power_records(user_name), user['ftp'], today)
    if not s:
        return
    dismissed = f"ftp_hint_dismissed_{user_name}_{s['ftp']}"
    if st.session_state.get(dismissed):
        return
    when = f"{s['best_20min_date']:%d %b}"
    cp = f" Critical power from the same efforts: {s['cp']:.0f} W." if s['cp'] else ""
    if s['direction'] == 'up':
        st.success(f"📈 **Your FTP may be higher.** Your best 20 minutes in the last {s['window_days']} days "
                   f"was **{s['best_20min_w']:.0f} W** ({when}), which suggests an FTP of about "
                   f"**{s['ftp']} W** (95 %), {s['change_w']:+d} W vs your current {user['ftp']} W.{cp}")
        b1, b2, _ = st.columns([1, 1, 4])
        if b1.button(f"Set FTP to {s['ftp']} W", key=f"accept_ftp_{user_name}",
                     help="Changes your profile FTP for new rides, zones and planning. Rides already saved "
                          "keep the FTP they were scored with."):
            update_user_ftp(user_name, s['ftp'])
            st.rerun()
        if b2.button("Not now", key=f"dismiss_ftp_{user_name}"):
            st.session_state[dismissed] = True
            st.rerun()
    else:
        st.info(f"ℹ️ Your best 20 minutes in the last {s['window_days']} days was {s['best_20min_w']:.0f} W "
                f"({when}), about {s['ftp']} W as an FTP, below your current {user['ftp']} W. That only means "
                f"no recent effort backs it up. If you haven't done a maximal 20-minute test lately, "
                f"your FTP may well be unchanged.{cp}")


# --- Load ----------------------------------------------------------------------

def _plan_editor(user_name, ftp, today):
    """
    Planned sessions: quick TSS entries, sessions built from intervals, and the builder.
    Returns the plan as [{'date', 'tss'}, ...] (quick edits included before they are saved).
    """
    quick = get_planned_sessions(user_name, after=today, structured=False)[['date', 'tss', 'note']]
    if quick.empty:
        quick = pd.DataFrame({'date': pd.Series(dtype='datetime64[ns]'),
                              'tss': pd.Series(dtype=float), 'note': pd.Series(dtype=str)})
    built = get_planned_sessions(user_name, after=today, structured=True)
    with st.expander(f"🗓️ Planned sessions ({len(quick) + len(built)})", expanded=False):
        st.caption("Plan the coming days and the chart forecasts fitness, fatigue and form from them. "
                   "Rest days need no entry. Build a session from intervals, or just enter its TSS.")
        _session_builder(user_name, ftp, today)
        _built_sessions(user_name, built)

        st.markdown("**Quick entry: just the TSS**")
        edited = st.data_editor(
            quick, num_rows="dynamic", hide_index=True, use_container_width=True, key=f"plan_{user_name}",
            column_config={
                'date': st.column_config.DateColumn("Date", min_value=(today + pd.Timedelta(days=1)).date(),
                                                    format="ddd DD MMM YYYY", required=True),
                'tss': st.column_config.NumberColumn("TSS", min_value=0, max_value=500, step=5, required=True),
                'note': st.column_config.TextColumn("Note (optional)"),
            })
        st.caption("Edits preview immediately; **Save plan** keeps them.")
        if st.button("Save plan", key=f"save_plan_{user_name}"):
            n = replace_planned_sessions(user_name, edited, today)
            st.toast(f"✅ Saved {n} planned sessions")

    plan = [] if edited is None or edited.empty else edited[['date', 'tss']].to_dict('records')
    plan += built[['date', 'tss']].to_dict('records')
    return plan


def _session_builder(user_name, ftp, today):
    st.markdown("**Build a session from intervals**")
    c1, c2 = st.columns([1, 2])
    day = c1.date_input("Date", value=(today + pd.Timedelta(days=1)).date(),
                        min_value=(today + pd.Timedelta(days=1)).date(), key=f"build_date_{user_name}")
    title = c2.text_input("Name (optional)", key=f"build_name_{user_name}",
                          placeholder="e.g. Threshold 4×8")
    st.caption("Each row is a block: it repeats *Reps* times, *Work* min at *Work W*, with *Rest* min at "
               "*Rest W* between the repeats. A warm-up is one rep with no rest.")
    blocks = st.data_editor(
        pd.DataFrame(wo.default_blocks(ftp), columns=wo.BLOCK_COLUMNS), num_rows="dynamic",
        hide_index=True, use_container_width=True, key=f"build_blocks_{user_name}",
        column_config={
            'name': st.column_config.TextColumn("Block"),
            'reps': st.column_config.NumberColumn("Reps", min_value=1, max_value=50, step=1),
            'work_min': st.column_config.NumberColumn("Work (min)", min_value=0, max_value=600, step=0.5),
            'work_w': st.column_config.NumberColumn("Work (W)", min_value=0, max_value=2000, step=5),
            'rest_min': st.column_config.NumberColumn("Rest (min)", min_value=0, max_value=600, step=0.5),
            'rest_w': st.column_config.NumberColumn("Rest (W)", min_value=0, max_value=2000, step=5),
        })
    s = wo.session_stats(blocks, ftp)
    m = st.columns(5)
    m[0].metric("Duration", f"{s['duration_s'] // 3600}:{s['duration_s'] % 3600 // 60:02d} h")
    m[1].metric("Average", f"{s['avg_power']:.0f} W")
    m[2].metric("Normalized", f"{s['np']:.0f} W")
    m[3].metric("IF", f"{s['if']:.2f}")
    m[4].metric("TSS", f"{s['tss']:.0f}")
    if not ftp:
        st.warning("Set an FTP in your profile to score the session.")
    if st.button("Add to plan", key=f"add_session_{user_name}"):
        clean = wo.clean_blocks(blocks)
        if not clean:
            st.warning("Add at least one block with some work time.")
        else:
            db_add_session(user_name, day, clean, s['tss'], title.strip() or wo.describe(clean))
            st.rerun()


def _built_sessions(user_name, built):
    if built.empty:
        return
    st.markdown("**Built sessions**")
    for _, r in built.iterrows():
        c = st.columns([2, 6, 1, 1])
        c[0].write(f"{r['date']:%a %d %b}")
        try:
            detail = wo.describe(json.loads(r['blocks']))
        except (TypeError, ValueError):
            detail = ""
        c[1].write(f"**{r['note']}**  \n{detail}" if r['note'] and r['note'] != detail else detail)
        c[2].write(f"{r['tss']:.0f} TSS")
        if c[3].button("✖", key=f"del_plan_{int(r['id'])}", help="Remove this session"):
            delete_planned_session(user_name, int(r['id']))
            st.rerun()


def _render_load(user_name, ftp, today, mode):
    history = get_user_tss_history(user_name)
    planned = _plan_editor(user_name, ftp, today)
    pmc = compute_pmc(history, today=today, planned=planned)
    if pmc.empty:
        st.info("No rides with TSS yet. Upload a ride or import your Strava history.")
        return

    now = pmc.loc[:today].iloc[-1]
    ramp = tl.ramp_rate(pmc)
    ratio = tl.acwr(pmc)
    mono = tl.monotony_strain(pmc)
    ramp_now = ramp.loc[:today].iloc[-1]
    acwr_now = ratio.loc[:today].iloc[-1]
    mono_now = mono['monotony'].loc[:today].iloc[-1]

    c = st.columns(6)
    _metric(c[0], "Fitness (CTL)", f"{now['CTL']:.0f}", None, "Chronic training load: 42-day weighted average TSS.")
    _metric(c[1], "Fatigue (ATL)", f"{now['ATL']:.0f}", None, "Acute training load: 7-day weighted average TSS.")
    _metric(c[2], "Form (TSB)", f"{now['TSB']:+.0f}", form_status(now['TSB']),
            "Yesterday's fitness minus fatigue. Rough guide: −10 to −30 productive training, "
            "+5 to +25 fresh / race-ready, below −30 high fatigue.")
    _metric(c[3], "Ramp rate", f"{ramp_now:+.1f}/wk" if pd.notna(ramp_now) else "N/A", ramp_status(ramp_now),
            "CTL change over the last 7 days. Rule of thumb: more than ~8 points/week is aggressive "
            "and hard to sustain.")
    _metric(c[4], "Acute:chronic", f"{acwr_now:.2f}" if pd.notna(acwr_now) else "N/A", acwr_status(acwr_now),
            "ATL / CTL. ~0.8–1.3 is the commonly cited range; above ~1.5 is a sharp load spike. "
            "Needs 4 weeks of history.")
    _metric(c[5], "Monotony (7 d)", f"{mono_now:.2f}" if pd.notna(mono_now) else "N/A", monotony_status(mono_now),
            "Foster's monotony: average daily load ÷ its day-to-day variation over the last week. "
            "Above ~2 = little variation between hard and easy days.")

    if pmc['planned'].any():
        last = pmc.index[-1]
        st.info(f"With the planned sessions, your form on **{last:%a %d %b}** would be "
                f"**{pmc['TSB'].iloc[-1]:+.0f}** ({form_status(pmc['TSB'].iloc[-1])}), "
                f"fitness {pmc['CTL'].iloc[-1]:.0f}.")

    label = remembered(st.radio, "load_period", "3 months", "Show", list(PERIODS), horizontal=True)
    start = _period_start(label, today)
    view = pmc if start is None else pmc.loc[start:]
    _chart(tp.plot_training_load(view, today, mode))

    c1, c2 = st.columns(2)
    with c1:
        _chart(tp.plot_ramp_rate(ramp.loc[view.index[0]:today], mode))
    with c2:
        _chart(tp.plot_acwr(ratio.loc[view.index[0]:today], mode))

    st.subheader("Volume")
    v1, v2 = st.columns([1, 3])
    freq = remembered(v1.radio, "vol_freq", "Week", "Group by", ["Week", "Month"], horizontal=True)
    metrics = {'Hours': 'hours', 'Distance (km)': 'km', 'TSS': 'tss', 'Work (kJ)': 'kj',
               'Elevation (m)': 'elevation_m', 'Rides': 'rides'}
    m_label = remembered(v2.radio, "vol_metric", "Hours", "Measure", list(metrics), horizontal=True)
    stats = get_ride_stats(user_name, start, today)
    vol = tl.volume(stats, 'W' if freq == "Week" else 'M')
    if vol.empty:
        st.info("No rides in this period.")
    else:
        _chart(tp.plot_volume(vol, metrics[m_label], m_label, mode))
        with st.expander("Table"):
            st.dataframe(vol.rename_axis(freq).round(1), use_container_width=True)


# --- Power profile -------------------------------------------------------------

def _render_power(user_name, user, today, mode):
    recs = get_power_records(user_name)
    if recs.empty:
        st.info("No power data yet.")
        return
    weight, ftp = user['weight'], user['ftp']
    curves = pp.window_curves(recs, today)
    _chart(tp.plot_power_windows(curves, mode, weight))

    recent = pp.as_dict(curves['Last 90 days'])
    two = estimate_critical_power(recent) if recent else None
    three = pp.fit_cp3(recent)
    st.markdown("##### Critical power (from your last 90 days)")
    c = st.columns(4)
    c[0].metric("CP (2-parameter)", f"{two['cp']:.0f} W" if two else "N/A",
                help="Fitted to your best 3–20 min efforts. The power you can sustain for a long time "
                     "(close to FTP).")
    c[1].metric("W′", f"{two['w_prime'] / 1000:.1f} kJ" if two else "N/A",
                help="Work capacity above CP: how much you can spend on surges before you must ease off.")
    c[2].metric("CP (3-parameter)", f"{three['cp']:.0f} W" if three else "N/A",
                help="Morton's model, fitted to 1–30 min efforts. It also models the short end of the curve, "
                     "so it usually matches real efforts better.")
    c[3].metric("Pmax (model)", f"{three['pmax']:.0f} W" if three else "N/A",
                help="The 3-parameter model's estimate of your maximal instantaneous power. "
                     "Reliable only if you have recent all-out sprints.")
    if not two and not three:
        st.caption("Needs maximal efforts between 3 and 20 minutes in the last 90 days.")

    hist = cp_history(recs, today=today)
    _chart(tp.plot_cp_history(hist, mode))

    st.markdown("##### Power profile")
    allt = pp.as_dict(curves['All time'])
    prof = pp.power_profile(recent, weight, ftp).merge(
        pp.power_profile(allt, weight, ftp)[['duration', 'w_per_kg']].rename(columns={'w_per_kg': 'all_time'}),
        on='duration')
    st.dataframe(prof.rename(columns={'duration': 'Duration', 'watts': 'Watts (90 d)',
                                      'w_per_kg': 'W/kg (90 d)', 'x_ftp': '× FTP (90 d)',
                                      'all_time': 'W/kg (all time)'}),
                 hide_index=True, use_container_width=True,
                 column_config={'Watts (90 d)': st.column_config.NumberColumn(format="%.0f"),
                                'W/kg (90 d)': st.column_config.NumberColumn(format="%.2f"),
                                '× FTP (90 d)': st.column_config.NumberColumn(format="%.2f"),
                                'W/kg (all time)': st.column_config.NumberColumn(format="%.2f")})
    st.caption("“× FTP” shows the shape of your curve: e.g. strong short efforts relative to FTP point to "
               "punch/sprint, a flat curve to a diesel/time-trial profile.")

    with st.expander("🏆 Records with dates"):
        best = curves['All time'].copy()
        r90 = curves['Last 90 days'].set_index('duration')['watts']
        best['Duration'] = best['duration'].map(_dur_label)
        best['W/kg'] = (best['watts'] / weight).round(2)
        best['Set on'] = best['date'].dt.strftime('%d %b %Y')
        best['Last 90 d'] = best['duration'].map(r90)
        st.dataframe(best[['Duration', 'watts', 'W/kg', 'Set on', 'Last 90 d']].rename(columns={'watts': 'Watts'}),
                     hide_index=True, use_container_width=True,
                     column_config={'Watts': st.column_config.NumberColumn(format="%.0f"),
                                    'Last 90 d': st.column_config.NumberColumn(format="%.0f")})


# --- Durability ------------------------------------------------------------------

def _render_durability(user_name, today, mode):
    st.caption("How much of your fresh best power you can still produce after 1, 2 and 3 MJ of work "
               "(about 1, 2 and 3 hours at ~280 W). High retention = good durability, which decides "
               "the end of long races and gran fondos.")
    label = remembered(st.radio, "dur_period", "6 months", "Period", list(PERIODS), horizontal=True)
    start = _period_start(label, today)
    table = du.durability_table(get_power_records(user_name, start, today),
                                get_kj_records(user_name, start, today), start, today)
    if table.empty:
        st.info("No rides in this period went beyond 1000 kJ (or they haven't been updated yet).")
        return
    _chart(tp.plot_durability(table, mode))
    show = table.assign(**{'After': table['kj'].map(lambda k: f"{k / 1000:g} MJ"),
                           'Duration': table['duration'].map(_dur_label)})
    with st.expander("Table"):
        st.dataframe(show[['After', 'Duration', 'fresh', 'fatigued', 'retention_pct']].rename(
            columns={'fresh': 'Fresh best (W)', 'fatigued': 'Best after (W)', 'retention_pct': 'Kept (%)'}),
            hide_index=True, use_container_width=True,
            column_config={'Fresh best (W)': st.column_config.NumberColumn(format="%.0f"),
                           'Best after (W)': st.column_config.NumberColumn(format="%.0f"),
                           'Kept (%)': st.column_config.NumberColumn(format="%.0f")})
    st.caption("A low value can also just mean you never went hard late in a long ride in this period.")


# --- Intensity ---------------------------------------------------------------------

def _render_intensity(user_name, today, mode):
    c1, c2, c3 = st.columns(3)
    kind = remembered(c1.radio, "tid_kind", "Power", "Based on", ["Power", "Heart rate"], horizontal=True)
    freq = remembered(c2.radio, "tid_freq", "Week", "Group by", ["Week", "Month"], horizontal=True)
    label = remembered(c3.selectbox, "tid_period", "3 months", "Period", list(PERIODS))
    start = _period_start(label, today)
    rows = get_zone_rows(user_name, 'power' if kind == "Power" else 'hr', start, today)
    if rows.empty:
        st.info("No zone data in this period.")
        return

    tot = it.three_zone(rows)
    pi = it.polarization_index(tot['low'], tot['mid'], tot['high'])
    total = sum(tot.values())
    m = st.columns(4)
    m[0].metric("Easy (Z1–2)", f"{tot['low'] / total:.0%}")
    m[1].metric("Moderate (Z3–4)", f"{tot['mid'] / total:.0%}")
    m[2].metric("Hard (Z5+)", f"{tot['high'] / total:.0%}")
    m[3].metric("Polarization index", f"{pi:.2f}" if pi is not None else "N/A",
                delta=tid_label(tot['low'], tot['mid'], tot['high'], pi), delta_color="off",
                help="log10(easy / moderate × hard × 100) on time shares (Treff et al. 2019). "
                     "Above 2.0 = polarized (mostly easy, some hard, little in between). "
                     "Not defined when there is more hard than easy time.")

    dist = it.distribution(rows, 'W' if freq == "Week" else 'M')
    _chart(tp.plot_intensity_distribution(dist, mode))
    _chart(tp.plot_polarization(dist, mode))
    with st.expander("Table"):
        st.dataframe((dist.assign(low=dist['low'] * 100, mid=dist['mid'] * 100, high=dist['high'] * 100)
                      .rename(columns={'hours': 'Hours', 'low': 'Easy %', 'mid': 'Moderate %',
                                       'high': 'Hard %', 'pi': 'PI'}).rename_axis(freq).round(2)),
                     use_container_width=True)
    st.caption(("Power zones (Coggan, from your FTP)" if kind == "Power" else "Heart-rate zones (Friel, from your LTHR)")
               + " grouped into three: Z1–2 easy, Z3–4 moderate, Z5+ hard. Moving time only.")
