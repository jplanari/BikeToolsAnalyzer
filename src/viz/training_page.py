"""Training page: load, power profile, durability, intensity distribution and deeper analytics."""
import json

import pandas as pd
import streamlit as st

from src.data.db import (replace_planned_sessions, add_structured_session as db_add_session,
                         delete_planned_session, update_user_ftp, update_user_physics)
from src.data.cached import (get_user_data, get_user_tss_history, get_ride_stats, get_power_records,
                             get_kj_records, get_zone_rows, get_planned_sessions,
                             count_rides_needing_update, compute_pmc, cp_history, get_user_best_power)
from src.analysis import training_load as tl
from src.analysis import power_profile as pp
from src.analysis import durability as du
from src.analysis import intensity as it
from src.analysis import nutrition as nut
from src.analysis import workout as wo
from src.analysis.power import estimate_critical_power
from src.viz import training_plots as tp
from src.viz.sections import section_picker, remembered
from src.viz.deeper_analytics import render_deeper_analytics
from src.analysis.labels import (PERIODS, period_start, dur_label, ramp_status, acwr_status, form_status,
                                 monotony_status, tid_label)
from src.viz.save_file import file_button
from src.i18n import t, fdate, mark, tr

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
    st.header(t("🏋️ Training"))
    if not user_name:
        st.warning(t("Please select a user in the sidebar."))
        return
    user = get_user_data(user_name)
    if not user:
        st.error(t("User data not found."))
        return

    n_old = count_rides_needing_update(user_name)
    if n_old:
        st.warning(t('{n_old} rides were saved by an older version and are missing data used here. Run **Update '
                        'ride history** in the User Corner to include them.', n_old=n_old))

    today = pd.Timestamp.today().normalize()
    mode = _theme()
    _threshold_hint(user_name, user, today)
    _report_expander(user_name, today)
    sections = [mark("📈 Load"), mark("⚡ Power profile"), mark("🔋 Durability"), mark("🎯 Intensity"), mark("🍌 Fuel"), mark("🔬 Deeper analytics")]
    section = section_picker(sections, key="training_section")
    if section == sections[0]:
        _render_load(user_name, user['ftp'], today, mode)
    elif section == sections[1]:
        _render_power(user_name, user, today, mode)
    elif section == sections[2]:
        _render_durability(user_name, today, mode)
    elif section == sections[3]:
        _render_intensity(user_name, today, mode)
    elif section == sections[4]:
        _render_fuel(user_name, user, today, mode)
    else:
        render_deeper_analytics(user_name, user)


def _report_expander(user_name, today):
    """Builds the PDF only on click (matplotlib is slow-ish) and keeps it for the download button."""
    with st.expander(t("📄 PDF report")):
        label = st.selectbox(t("Period"), list(PERIODS), index=1, key="report_period", format_func=tr)
        key = (user_name, label, today)
        if st.button(t("Create report"), key="report_make"):
            from src.data.reports.report_data import collect_report_data
            from src.viz.report_pdf import build_report_pdf
            with st.spinner(t("Building the report…")):
                pdf = build_report_pdf(collect_report_data(user_name, PERIODS[label], today))
            st.session_state["report_pdf"] = (key, pdf)
        ready = st.session_state.get("report_pdf")
        if ready and ready[0] == key:
            file_button(ready[1], f"training-report-{user_name}-{today:%Y-%m-%d}.pdf", key="report_dl")
        elif ready:
            st.caption(t("Period or user changed: create the report again."))


def _threshold_hint(user_name, user, today):
    """Suggests a new FTP when recent riding supports it; nothing changes until you accept."""
    s = pp.suggest_threshold(get_power_records(user_name), user['ftp'], today)
    if not s:
        return
    dismissed = f"ftp_hint_dismissed_{user_name}_{s['ftp']}"
    if st.session_state.get(dismissed):
        return
    when = fdate(s['best_20min_date'], "%d %b")
    cp = t(" Critical power from the same efforts: {cp:.0f} W.", cp=s['cp']) if s['cp'] else ""
    if s['direction'] == 'up':
        st.success(t('📈 **Your FTP may be higher.** Your best 20 minutes in the last {window_days} days was '
                        '**{best_20min_w:.0f} W** ({when}), which suggests an FTP of about **{ftp} W** (95 %), '
                        '{change_w:+d} W vs your current {ftp_2} W.{cp}', window_days=s['window_days'], best_20min_w=s['best_20min_w'], when=when, ftp=s['ftp'], change_w=s['change_w'], ftp_2=user['ftp'], cp=cp))
        b1, b2, _ = st.columns([1, 1, 4])
        if b1.button(t('Set FTP to {ftp} W', ftp=s['ftp']), key=f"accept_ftp_{user_name}",
                     help=t("Changes your profile FTP for new rides, zones and planning. Rides already saved "
                          "keep the FTP they were scored with.")):
            update_user_ftp(user_name, s['ftp'])
            st.rerun()
        if b2.button(t("Not now"), key=f"dismiss_ftp_{user_name}"):
            st.session_state[dismissed] = True
            st.rerun()
    else:
        st.info(t('ℹ️ Your best 20 minutes in the last {window_days} days was {best_20min_w:.0f} W ({when}), '
                     'about {ftp} W as an FTP, below your current {ftp_2} W. That only means no recent effort backs '
                     "it up. If you haven't done a maximal 20-minute test lately, your FTP may well be "
                     'unchanged.{cp}', window_days=s['window_days'], best_20min_w=s['best_20min_w'], when=when, ftp=s['ftp'], ftp_2=user['ftp'], cp=cp))


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
    with st.expander(t('🗓️ Planned sessions ({n})', n=len(quick) + len(built)), expanded=False):
        st.caption(t("Plan the coming days and the chart forecasts fitness, fatigue and form from them. "
                   "Rest days need no entry. Build a session from intervals, or just enter its TSS."))
        _session_builder(user_name, ftp, today)
        _built_sessions(user_name, built, ftp, today)

        st.markdown(t("**Quick entry: just the TSS**"))
        edited = st.data_editor(
            quick, num_rows="dynamic", hide_index=True, use_container_width=True, key=f"plan_{user_name}",
            column_config={
                'date': st.column_config.DateColumn(t("Date"), min_value=(today + pd.Timedelta(days=1)).date(),
                                                    format="ddd DD MMM YYYY", required=True),
                'tss': st.column_config.NumberColumn("TSS", min_value=0, max_value=500, step=5, required=True),
                'note': st.column_config.TextColumn(t("Note (optional)")),
            })
        st.caption(t("Edits preview immediately; **Save plan** keeps them."))
        if st.button(t("Save plan"), key=f"save_plan_{user_name}"):
            n = replace_planned_sessions(user_name, edited, today)
            st.toast(t('✅ Saved {n} planned sessions', n=n))

    plan = [] if edited is None or edited.empty else edited[['date', 'tss']].to_dict('records')
    plan += built[['date', 'tss']].to_dict('records')
    return plan


def _session_builder(user_name, ftp, today):
    st.markdown(t("**Build a session from intervals**"))
    c1, c2 = st.columns([1, 2])
    day = c1.date_input(t("Date"), value=(today + pd.Timedelta(days=1)).date(),
                        min_value=(today + pd.Timedelta(days=1)).date(), key=f"build_date_{user_name}")
    title = c2.text_input(t("Name (optional)"), key=f"build_name_{user_name}",
                          placeholder=t("e.g. Threshold 4×8"))
    st.caption(t("Each row is a block: it repeats *Reps* times, *Work* min at *Work W*, with *Rest* min at "
               "*Rest W* between the repeats. A warm-up is one rep with no rest."))
    blocks = st.data_editor(
        pd.DataFrame(wo.default_blocks(ftp), columns=wo.BLOCK_COLUMNS), num_rows="dynamic",
        hide_index=True, use_container_width=True, key=f"build_blocks_{user_name}",
        column_config={
            'name': st.column_config.TextColumn(t("Block")),
            'reps': st.column_config.NumberColumn(t("Reps"), min_value=1, max_value=50, step=1),
            'work_min': st.column_config.NumberColumn(t("Work (min)"), min_value=0, max_value=600, step=0.5),
            'work_w': st.column_config.NumberColumn(t("Work (W)"), min_value=0, max_value=2000, step=5),
            'rest_min': st.column_config.NumberColumn(t("Rest (min)"), min_value=0, max_value=600, step=0.5),
            'rest_w': st.column_config.NumberColumn(t("Rest (W)"), min_value=0, max_value=2000, step=5),
        })
    s = wo.session_stats(blocks, ftp)
    m = st.columns(5)
    m[0].metric(t("Duration"), f"{s['duration_s'] // 3600}:{s['duration_s'] % 3600 // 60:02d} h")
    m[1].metric(t("Average"), f"{s['avg_power']:.0f} W")
    m[2].metric(t("Normalized"), f"{s['np']:.0f} W")
    m[3].metric("IF", f"{s['if']:.2f}")
    m[4].metric("TSS", f"{s['tss']:.0f}")
    if not ftp:
        st.warning(t("Set an FTP in your profile to score the session."))
    if st.button(t("Add to plan"), key=f"add_session_{user_name}"):
        clean = wo.clean_blocks(blocks)
        if not clean:
            st.warning(t("Add at least one block with some work time."))
        else:
            db_add_session(user_name, day, clean, s['tss'], title.strip() or wo.describe(clean))
            st.rerun()


def _wahoo_ready(user_name):
    """(WahooConfig, True) when Wahoo is configured and this profile is connected, else (None/cfg, False)."""
    from src.viz.sidebar import _wahoo_config
    from src.data.db import get_wahoo_auth
    cfg = _wahoo_config()
    return cfg, bool(cfg and get_wahoo_auth(user_name))


def _send_controls(user_name, ftp, cfg, row, sent, today):
    """Send / resend button for one built session; shows what happened."""
    from src.data.providers import wahoo as wh
    from src.data.providers import wahoo_send as ws
    sid = int(row['id'])
    label = t("↻ Update on Wahoo") if sent else t("📲 Send to Wahoo")
    if st.button(label, key=f"wahoo_send_{sid}", help=t("Creates (or updates) this workout in your Wahoo account "
                 "so it shows up on your ELEMNT.")):
        try:
            res = ws.send_session(cfg, user_name, row.to_dict(), ftp, st.session_state.get("wahoo_send_time"),
                                  outdoor=st.session_state.get("wahoo_send_outdoor", True))
            st.toast(t("✅ Updated on Wahoo") if res['updated'] else t("✅ Sent to Wahoo"))
            st.rerun()
        except (ws.SendError, wh.WahooError) as e:
            st.error(t('Wahoo: {e}', e=e))
    if sent:
        if st.button(t("Remove from Wahoo"), key=f"wahoo_remove_{sid}"):
            try:
                ws.remove_session(cfg, user_name, sid)
                st.rerun()
            except wh.WahooError as e:
                st.error(t('Wahoo: {e}', e=e))


def _built_sessions(user_name, built, ftp=None, today=None):
    if built.empty:
        return
    st.markdown(t("**Built sessions**"))
    cfg, wahoo_ok = _wahoo_ready(user_name)
    sent = {}
    if wahoo_ok:
        import datetime as _dt
        from src.data.db import get_wahoo_sent
        from src.data.providers import wahoo_send as ws
        sent = get_wahoo_sent(user_name)
        c1, c2 = st.columns([1, 1])
        c1.time_input(t("Start time on Wahoo"), value=_dt.time(8, 0), key="wahoo_send_time",
                      help=t("Local time shown on the workout. Wahoo only delivers sessions to the ELEMNT from "
                             "today up to 6 days ahead."))
        c2.checkbox(t("Outdoor ride"), value=True, key="wahoo_send_outdoor")
    for _, r in built.iterrows():
        c = st.columns([2, 6, 1, 1])
        c[0].write(fdate(r['date'], "%a %d %b"))
        try:
            detail = wo.describe(json.loads(r['blocks']))
        except (TypeError, ValueError):
            detail = ""
        c[1].write(f"**{r['note']}**  \n{detail}" if r['note'] and r['note'] != detail else detail)
        c[2].write(f"{r['tss']:.0f} TSS")
        if c[3].button("✖", key=f"del_plan_{int(r['id'])}", help=t("Remove this session")):
            if int(r['id']) in sent:           # also take it off the head unit
                try:
                    ws.remove_session(cfg, user_name, int(r['id']))
                except Exception as e:
                    st.warning(t('Could not remove it from Wahoo ({e}); delete it in the Wahoo app.', e=e))
            delete_planned_session(user_name, int(r['id']))
            st.rerun()
        if wahoo_ok:
            with c[1]:
                _send_controls(user_name, ftp, cfg, r, int(r['id']) in sent, today)
                if int(r['id']) in sent and not ws.in_sync_window(r['date'], today):
                    st.caption(t("Sent. It reaches your ELEMNT within 6 days of the ride; use ↻ Update then if it "
                               "doesn't show up."))
                elif int(r['id']) in sent:
                    st.caption(t("Sent to Wahoo."))


def _render_load(user_name, ftp, today, mode):
    history = get_user_tss_history(user_name)
    planned = _plan_editor(user_name, ftp, today)
    pmc = compute_pmc(history, today=today, planned=planned)
    if pmc.empty:
        st.info(t("No rides with TSS yet. Upload a ride or import your Strava history."))
        return

    now = pmc.loc[:today].iloc[-1]
    ramp = tl.ramp_rate(pmc)
    ratio = tl.acwr(pmc)
    mono = tl.monotony_strain(pmc)
    ramp_now = ramp.loc[:today].iloc[-1]
    acwr_now = ratio.loc[:today].iloc[-1]
    mono_now = mono['monotony'].loc[:today].iloc[-1]

    c = st.columns(6)
    _metric(c[0], t("Fitness (CTL)"), f"{now['CTL']:.0f}", None, t("Chronic training load: 42-day weighted average TSS."))
    _metric(c[1], t("Fatigue (ATL)"), f"{now['ATL']:.0f}", None, t("Acute training load: 7-day weighted average TSS."))
    _metric(c[2], t("Form (TSB)"), f"{now['TSB']:+.0f}", form_status(now['TSB']),
            t("Yesterday's fitness minus fatigue. Rough guide: −10 to −30 productive training, "
              "+5 to +25 fresh / race-ready, below −30 high fatigue."))
    _metric(c[3], t("Ramp rate"), t("{ramp:+.1f}/wk", ramp=ramp_now) if pd.notna(ramp_now) else t("N/A"), ramp_status(ramp_now),
            t("CTL change over the last 7 days. Rule of thumb: more than ~8 points/week is aggressive "
              "and hard to sustain."))
    _metric(c[4], t("Acute:chronic"), f"{acwr_now:.2f}" if pd.notna(acwr_now) else t("N/A"), acwr_status(acwr_now),
            t("ATL / CTL. ~0.8–1.3 is the commonly cited range; above ~1.5 is a sharp load spike. "
              "Needs 4 weeks of history."))
    _metric(c[5], t("Monotony (7 d)"), f"{mono_now:.2f}" if pd.notna(mono_now) else t("N/A"), monotony_status(mono_now),
            t("Foster's monotony: average daily load ÷ its day-to-day variation over the last week. "
              "Above ~2 = little variation between hard and easy days."))

    if pmc['planned'].any():
        last = pmc.index[-1]
        st.info(t('With the planned sessions, your form on **{last:%a %d %b}** would be **{form:+.0f}** '
                     '({form_status}), fitness {value4:.0f}.', last=last, form=pmc['TSB'].iloc[-1], form_status=form_status(pmc['TSB'].iloc[-1]), value4=pmc['CTL'].iloc[-1]))

    label = remembered(st.radio, "load_period", "3 months", t("Show"), list(PERIODS), horizontal=True, format_func=tr)
    start = _period_start(label, today)
    view = pmc if start is None else pmc.loc[start:]
    _chart(tp.plot_training_load(view, today, mode))
    with st.expander(t("Raw load data")):
        st.dataframe(view, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        _chart(tp.plot_ramp_rate(ramp.loc[view.index[0]:today], mode))
    with c2:
        _chart(tp.plot_acwr(ratio.loc[view.index[0]:today], mode))

    st.subheader(t("Volume"))
    v1, v2 = st.columns([1, 3])
    freq = remembered(v1.radio, "vol_freq", "Week", t("Group by"), [mark("Week"), mark("Month")], horizontal=True, format_func=tr)
    metrics = {mark('Hours'): 'hours', mark('Distance (km)'): 'km', 'TSS': 'tss', mark('Work (kJ)'): 'kj',
               mark('Elevation (m)'): 'elevation_m', mark('Rides'): 'rides'}
    m_label = remembered(v2.radio, "vol_metric", "Hours", t("Measure"), list(metrics), horizontal=True, format_func=tr)
    stats = get_ride_stats(user_name, start, today)
    vol = tl.volume(stats, 'W' if freq == "Week" else 'M')
    if vol.empty:
        st.info(t("No rides in this period."))
    else:
        _chart(tp.plot_volume(vol, metrics[m_label], m_label, mode))
        with st.expander(t("Table")):
            st.dataframe(vol.rename_axis(freq).round(1), use_container_width=True)


# --- Power profile -------------------------------------------------------------

def _render_power(user_name, user, today, mode):
    recs = get_power_records(user_name)
    if recs.empty:
        st.info(t("No power data yet."))
        return
    weight, ftp = user['weight'], user['ftp']
    curves = pp.window_curves(recs, today)
    _chart(tp.plot_power_windows(curves, mode, weight))

    recent = pp.as_dict(curves['Last 90 days'])
    two = estimate_critical_power(recent) if recent else None
    three = pp.fit_cp3(recent)
    st.markdown(t("##### Critical power (from your last 90 days)"))
    c = st.columns(4)
    c[0].metric(t("CP (2-parameter)"), f"{two['cp']:.0f} W" if two else t("N/A"),
                help=t("Fitted to your best 3–20 min efforts. The power you can sustain for a long time "
                     "(close to FTP)."))
    c[1].metric("W′", f"{two['w_prime'] / 1000:.1f} kJ" if two else t("N/A"),
                help=t("Work capacity above CP: how much you can spend on surges before you must ease off."))
    c[2].metric(t("CP (3-parameter)"), f"{three['cp']:.0f} W" if three else t("N/A"),
                help=t("Morton's model, fitted to 1–30 min efforts. It also models the short end of the curve, "
                     "so it usually matches real efforts better."))
    c[3].metric(t("Pmax (model)"), f"{three['pmax']:.0f} W" if three else t("N/A"),
                help=t("The 3-parameter model's estimate of your maximal instantaneous power. "
                     "Reliable only if you have recent all-out sprints."))
    if not two and not three:
        st.caption(t("Needs maximal efforts between 3 and 20 minutes in the last 90 days."))

    hist = cp_history(recs, today=today)
    _chart(tp.plot_cp_history(hist, mode))

    st.markdown(t("##### Power profile"))
    allt = pp.as_dict(curves['All time'])
    prof = pp.power_profile(recent, weight, ftp).merge(
        pp.power_profile(allt, weight, ftp)[['duration', 'w_per_kg']].rename(columns={'w_per_kg': 'all_time'}),
        on='duration')
    c_dur, c_w90, c_kg90, c_ftp90, c_kgall = (t('Duration'), t('Watts (90 d)'), t('W/kg (90 d)'), t('× FTP (90 d)'),
                                              t('W/kg (all time)'))
    st.dataframe(prof.rename(columns={'duration': c_dur, 'watts': c_w90, 'w_per_kg': c_kg90, 'x_ftp': c_ftp90,
                                      'all_time': c_kgall}),
                 hide_index=True, use_container_width=True,
                 column_config={c_w90: st.column_config.NumberColumn(format="%.0f"),
                                c_kg90: st.column_config.NumberColumn(format="%.2f"),
                                c_ftp90: st.column_config.NumberColumn(format="%.2f"),
                                c_kgall: st.column_config.NumberColumn(format="%.2f")})
    st.caption(t("“× FTP” shows the shape of your curve: e.g. strong short efforts relative to FTP point to "
               "punch/sprint, a flat curve to a diesel/time-trial profile."))

    with st.expander(t("🏆 Records with dates")):
        best = curves['All time'].copy()
        r90 = curves['Last 90 days'].set_index('duration')['watts']
        c_dur, c_kg, c_set, c_90, c_w = t('Duration'), t('W/kg'), t('Set on'), t('Last 90 d'), t('Watts')
        best[c_dur] = best['duration'].map(_dur_label)
        best[c_kg] = (best['watts'] / weight).round(2)
        best[c_set] = best['date'].map(lambda d: fdate(d, '%d %b %Y'))
        best[c_90] = best['duration'].map(r90)
        st.dataframe(best[[c_dur, 'watts', c_kg, c_set, c_90]].rename(columns={'watts': c_w}),
                     hide_index=True, use_container_width=True,
                     column_config={c_w: st.column_config.NumberColumn(format="%.0f"),
                                    c_90: st.column_config.NumberColumn(format="%.0f")})

    _render_cp_profile(user_name)


def _render_cp_profile(user_name):
    """Critical-power model from ALL your best efforts, with buttons that store it in the profile."""
    best_power = get_user_best_power(user_name)
    if not best_power:
        return
    with st.expander(t("💾 Save critical power to your profile (all-time model)")):
        cp_results = estimate_critical_power(best_power)
        if not cp_results:
            st.info(t("Not enough data to model Critical Power. Upload maximal efforts between 3 and 20 minutes."))
            return
        c = st.columns(3)
        c[0].metric(t("Critical Power (CP)"), f"{cp_results['cp']:.0f} W",
                    help=t("The theoretical power you can sustain 'indefinitely' (aerobic ceiling). "
                         "Usually slightly higher than FTP."))
        c[1].metric(t("W' (Anaerobic Capacity)"), f"{cp_results['w_prime'] / 1000:.1f} kJ",
                    help=t("Battery size above CP. Typical values: 10kJ (Sprinter) to 25kJ (Pursuiter)."))
        c[2].metric(t("Model Fit (R²)"), f"{cp_results['r_squared']:.3f}",
                    help=t("How well your data fits the model. >0.95 is excellent. If low, you might need more "
                         "all-out efforts between 3 and 12 mins."))
        if cp_results['w_prime'] > 20000:
            st.caption(t("💡 You have a high anaerobic capacity: likely a **Puncheur** or **Sprinter**."))
        elif cp_results['w_prime'] < 12000:
            st.caption(t("💡 Your engine is more diesel: likely a **Time Trialist** or **Climber** "
                       "(or you need more intervals!)."))
        st.caption(t("Fitted to your best efforts of all time, so it can be higher than the last-90-days "
                   "figures above."))
        b1, b2 = st.columns([1, 2])
        if b1.button(t("💾 Save CP & W'"), key="save_cp"):
            update_user_physics(user_name, cp=cp_results['cp'], w_prime=cp_results['w_prime'])
            st.success(t("Updated {user_name}: CP={cp:.0f}W, W'={w_prime:.1f}kJ", user_name=user_name, cp=cp_results['cp'], w_prime=cp_results['w_prime'] / 1000))
            st.rerun()
        if b2.button(t("💾 Save CP & W' AND set FTP = CP"), key="save_cp_ftp"):
            update_user_physics(user_name, cp=cp_results['cp'], w_prime=cp_results['w_prime'], update_ftp=True)
            st.success(t('Full update! FTP is now {cp:.0f}W.', cp=cp_results['cp']))
            st.rerun()


# --- Durability ------------------------------------------------------------------

def _render_durability(user_name, today, mode):
    st.caption(t("How much of your fresh best power you can still produce after 1, 2 and 3 MJ of work "
               "(about 1, 2 and 3 hours at ~280 W). High retention = good durability, which decides "
               "the end of long races and gran fondos."))
    label = remembered(st.radio, "dur_period", "6 months", t("Period"), list(PERIODS), horizontal=True, format_func=tr)
    start = _period_start(label, today)
    table = du.durability_table(get_power_records(user_name, start, today),
                                get_kj_records(user_name, start, today), start, today)
    if table.empty:
        st.info(t("No rides in this period went beyond 1000 kJ (or they haven't been updated yet)."))
        return
    _chart(tp.plot_durability(table, mode))
    c_after, c_dur, c_fresh, c_best, c_kept = (t('After'), t('Duration'), t('Fresh best (W)'), t('Best after (W)'),
                                               t('Kept (%)'))
    show = table.assign(**{c_after: table['kj'].map(lambda k: f"{k / 1000:g} MJ"),
                           c_dur: table['duration'].map(_dur_label)})
    with st.expander(t("Table")):
        st.dataframe(show[[c_after, c_dur, 'fresh', 'fatigued', 'retention_pct']].rename(
            columns={'fresh': c_fresh, 'fatigued': c_best, 'retention_pct': c_kept}),
            hide_index=True, use_container_width=True,
            column_config={c_fresh: st.column_config.NumberColumn(format="%.0f"),
                           c_best: st.column_config.NumberColumn(format="%.0f"),
                           c_kept: st.column_config.NumberColumn(format="%.0f")})
    st.caption(t("A low value can also just mean you never went hard late in a long ride in this period."))


# --- Intensity ---------------------------------------------------------------------

def _render_intensity(user_name, today, mode):
    c1, c2, c3 = st.columns(3)
    kind = remembered(c1.radio, "tid_kind", "Power", t("Based on"), [mark("Power"), mark("Heart rate")], horizontal=True, format_func=tr)
    freq = remembered(c2.radio, "tid_freq", "Week", t("Group by"), [mark("Week"), mark("Month")], horizontal=True, format_func=tr)
    label = remembered(c3.selectbox, "tid_period", "3 months", t("Period"), list(PERIODS), format_func=tr)
    start = _period_start(label, today)
    rows = get_zone_rows(user_name, 'power' if kind == "Power" else 'hr', start, today)
    if rows.empty:
        st.info(t("No zone data in this period."))
        return

    tot = it.three_zone(rows)
    pi = it.polarization_index(tot['low'], tot['mid'], tot['high'])
    total = sum(tot.values())
    m = st.columns(4)
    m[0].metric(t("Easy (Z1–2)"), f"{tot['low'] / total:.0%}")
    m[1].metric(t("Moderate (Z3–4)"), f"{tot['mid'] / total:.0%}")
    m[2].metric(t("Hard (Z5+)"), f"{tot['high'] / total:.0%}")
    m[3].metric(t("Polarization index"), f"{pi:.2f}" if pi is not None else t("N/A"),
                delta=tid_label(tot['low'], tot['mid'], tot['high'], pi), delta_color="off",
                help=t("log10(easy / moderate × hard × 100) on time shares (Treff et al. 2019). "
                     "Above 2.0 = polarized (mostly easy, some hard, little in between). "
                     "Not defined when there is more hard than easy time."))

    dist = it.distribution(rows, 'W' if freq == "Week" else 'M')
    _chart(tp.plot_intensity_distribution(dist, mode))
    _chart(tp.plot_polarization(dist, mode))
    with st.expander(t("Table")):
        st.dataframe((dist.assign(low=dist['low'] * 100, mid=dist['mid'] * 100, high=dist['high'] * 100)
                      .rename(columns={'hours': t('Hours'), 'low': t('Easy %'), 'mid': t('Moderate %'),
                                       'high': t('Hard %'), 'pi': 'PI'}).rename_axis(tr(freq)).round(2)),
                     use_container_width=True)
    zones = t("Power zones (Coggan, from your FTP)") if kind == "Power" else t("Heart-rate zones (Friel, from your LTHR)")
    st.caption(t("{zones} grouped into three: Z1–2 easy, Z3–4 moderate, Z5+ hard. Moving time only.", zones=zones))


# --- Fuel ----------------------------------------------------------------------

def _render_fuel(user_name, user, today, mode):
    st.caption(t("Energy and carbohydrate burned on the bike, estimated from your power. No food is logged: "
                 "these are estimates of what the riding cost, not of what you ate."))
    label = remembered(st.radio, "fuel_period", "3 months", t("Show"), list(PERIODS), horizontal=True, format_func=tr)
    start = _period_start(label, today)
    stats = get_ride_stats(user_name, start, today)
    total = nut.totals(stats)
    if not total['rides']:
        st.info(t("No rides with power in this period."))
        return
    days = max(1, (today - (start if start is not None else pd.to_datetime(stats['date']).min().normalize())).days + 1)
    c = st.columns(4)
    c[0].metric(t("Energy burned"), f"{total['kcal']:,.0f} kcal",
                help=t("Work on the pedals converted to energy (about 24 % efficiency). Resting energy is not included."))
    c[1].metric(t("Per week"), f"{total['kcal'] / days * 7:,.0f} kcal")
    c[2].metric(t("Carbohydrate burned"), f"{total['carb_g']:,.0f} g" if total['carb_g'] else t("N/A"),
                help=t("From a typical fuel-mix curve: the harder the riding, the larger the carbohydrate share."))
    c[3].metric(t("Carbohydrate share"), f"{total['carb_pct']:.0f} %" if total['carb_pct'] is not None else t("N/A"),
                help=t("Share of the energy that came from carbohydrate rather than fat."))
    n_old = count_rides_needing_update(user_name)
    if n_old and total['carb_g'] is None:
        st.info(t("Carbohydrate figures need the ride history to be updated: run **Update ride history** in the User Corner."))
    freq = remembered(st.radio, "fuel_freq", "Week", t("Group by"), [mark("Week"), mark("Month")], horizontal=True, format_func=tr)
    per = nut.by_period(stats, 'W' if freq == "Week" else 'M')
    _chart(tp.plot_energy(per, t("Energy (kcal)"), mode))
    with st.expander(t("Table")):
        st.dataframe(per.rename(columns={'kcal': 'kcal', 'carb_g': 'carb (g)', 'fat_g': 'fat (g)',
                                         'carb_pct': 'carb (%)'}).round(1), use_container_width=True)
    heaviest = per['kcal'].idxmax()
    st.caption(t("Heaviest {freq}: {when:%d %b %Y}, {kcal:,.0f} kcal.", freq=tr(freq).lower(), when=heaviest, kcal=per['kcal'].max()))
    st.caption(t("Carbohydrate guide for riding: about 30–60 g/h on rides of 1–2 h, 60–90 g/h beyond 3 h, nothing extra "
                 "under 45 minutes. Guidance, not dietary advice. Each ride's own guide is on its Ride Summary."))
