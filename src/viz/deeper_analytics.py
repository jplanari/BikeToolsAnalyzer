"""Deeper analytics (Training page): power vs heart rate, power vs cadence, recovery and efficiency,
for any period with an optional comparison against the period before it."""
import streamlit as st
import pandas as pd
from src.viz.plots import (plot_joint_heatmap,
                           plot_y_by_power,
                           plot_trend,
                           plot_quadrants,
                           plot_cadence_by_zone)
from src.analysis import correlations as co
from src.data.db import update_user_crank
from src.data.cached import get_ride_stats, get_hist, period_coverage
from src.viz.sections import section_picker, remembered
from src.i18n import t, mark, tr


def _fmt_delta(a, b, fmt="{:+.1f}"):
    return fmt.format(a - b) if a is not None and b is not None else None


def render_deeper_analytics(user_name, user_data):
    today = pd.Timestamp.today().normalize()
    c1, c2 = st.columns([2, 1])
    period = remembered(c1.date_input, "da_period", ((today - pd.Timedelta(days=90)).date(), today.date()),
                        t("Period"))
    if not isinstance(period, (tuple, list)) or len(period) != 2:
        st.info(t("Pick a start and an end date."))
        return
    a0, a1 = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    compare = remembered(c2.checkbox, "da_compare", True, t("Compare with previous period"),
                         help=t("The same number of days immediately before the selected period."))
    span = (a1 - a0) + pd.Timedelta(days=1)
    b0, b1 = a0 - span, a0 - pd.Timedelta(days=1)
    lab_a = f"{a0:%d %b} – {a1:%d %b %Y}"
    lab_b = f"{b0:%d %b} – {b1:%d %b %Y}"

    ftp, lthr = user_data['ftp'], user_data['lthr']
    hr_a = get_hist(user_name, 'p_hr', a0, a1)
    cad_a = get_hist(user_name, 'p_cad', a0, a1)
    hr_b = get_hist(user_name, 'p_hr', b0, b1) if compare else None
    cad_b = get_hist(user_name, 'p_cad', b0, b1) if compare else None
    stats = get_ride_stats(user_name, b0 if compare else a0, a1)

    cov = period_coverage(user_name, a0, a1)
    if cov['needs_update']:
        st.warning(t("{needs_update} of the {rides} rides in this period were saved by an older version and aren't "
                        'included here yet. Run **Update ride history** in the User Corner to add them.', needs_update=cov['needs_update'], rides=cov['rides']))

    def _empty_reason(what):
        if cov['rides'] == 0:
            return "No rides in this period."
        if cov['needs_update'] == cov['rides']:
            return f"None of the {cov['rides']} rides in this period has been updated yet (run **Update ride history** in the User Corner)."
        return (f"None of the {cov['rides'] - cov['needs_update']} up-to-date rides in this period "
                f"has both power and {what} while pedalling.")

    sections = [mark("❤️ Power ↔ Heart rate"), mark("🦵 Power ↔ Cadence"), mark("🔁 Recovery & Efficiency")]
    section = section_picker(sections, key="da_section")

    # ---- Power vs HR ----
    if section == sections[0]:
        if hr_a.empty:
            st.info(_empty_reason("heart rate"))
        else:
            fit_a = co.linear_fit(hr_a)
            fit_b = co.linear_fit(hr_b) if compare else None
            ref_p = int(round(0.65 * ftp / 5) * 5) if ftp else 150
            hr_ref_a = co.y_at_power(hr_a, ref_p)
            hr_ref_b = co.y_at_power(hr_b, ref_p) if compare else None

            m1, m2, m3 = st.columns(3)
            m1.metric(t("Correlation (r)"), f"{fit_a['r']:.2f}" if fit_a else "N/A",
                      delta=_fmt_delta(fit_a and fit_a['r'], fit_b and fit_b['r'], "{:+.2f}"),
                      help=t("How tightly heart rate follows power (30 s averages, moving time). "
                           "Close to 1 = HR is a good proxy for effort; lower values usually mean "
                           "lots of short efforts, heat, fatigue or HR sensor noise."))
            m2.metric(t("HR rise per +10 W"), f"{fit_a['slope'] * 10:.1f} bpm" if fit_a else "N/A",
                      delta=_fmt_delta(fit_a and fit_a['slope'] * 10, fit_b and fit_b['slope'] * 10),
                      delta_color="inverse",
                      help=t("Slope of the HR-vs-power line. A flatter slope means each extra watt costs less heart rate."))
            hr_label = t("HR at {ref_p} W", ref_p=ref_p)
            m3.metric(hr_label, f"{hr_ref_a:.0f} bpm" if hr_ref_a else "N/A",
                      delta=_fmt_delta(hr_ref_a, hr_ref_b, "{:+.0f} bpm"), delta_color="inverse",
                      help=t('Average HR while riding at {ref_p} W ±15 W (65 % of FTP). Lower for the same power = better '
                                'aerobic fitness (or more fatigue/heat if it rises).', ref_p=ref_p))
            if compare:
                st.caption(t('Deltas compare with {lab_b}.', lab_b=lab_b))

            curve_a = co.mean_y_by_power(hr_a)
            st.plotly_chart(plot_joint_heatmap(co.aggregate(hr_a), 5, t("Heart rate (bpm, 30 s avg)"),
                                               t("Time at power × heart rate · {lab_a}", lab_a=lab_a), overlay=curve_a),
                            use_container_width=True)
            curves = {lab_a: curve_a}
            if compare and hr_b is not None and not hr_b.empty:
                curves[lab_b] = co.mean_y_by_power(hr_b)
            st.plotly_chart(plot_y_by_power(curves, t("Average HR (bpm)"), t("Heart rate at each power")),
                            use_container_width=True)
            st.caption(t("If this curve moves **down** between periods, you are producing the same power "
                       "at a lower heart rate: the clearest sign of improving aerobic fitness."))

            hr_all = pd.concat([x for x in [hr_b, hr_a] if x is not None and not x.empty])
            weekly_hr = co.weekly(hr_all, lambda g: co.y_at_power(g, ref_p, min_seconds=180))
            if len(weekly_hr) >= 2:
                st.plotly_chart(plot_trend({hr_label: weekly_hr}, t("bpm"),
                                           t("Weekly heart rate at {ref_p} W", ref_p=ref_p)), use_container_width=True)

    # ---- Power vs cadence ----
    if section == sections[1]:
        if cad_a.empty:
            st.info(_empty_reason("cadence"))
        else:
            crank_mm = float(user_data.get('crank_mm') or 172.5)
            new_crank = st.number_input(t("Crank length (mm)"), min_value=150.0, max_value=200.0,
                                        value=crank_mm, step=2.5, key="da_crank",
                                        help=t("Only affects the force/velocity axes of the quadrant chart."))
            if new_crank != crank_mm:
                update_user_crank(user_name, new_crank)
                crank_mm = new_crank

            ref_cad = co.weighted_median_y(cad_a)
            ref_cad_b = co.weighted_median_y(cad_b) if compare else None
            quads = co.quadrants(cad_a, ftp, ref_cad)
            quads_b = co.quadrants(cad_b, ftp, ref_cad) if compare else None

            m1, m2, m3 = st.columns(3)
            m1.metric(t("Median cadence"), f"{ref_cad:.0f} rpm" if ref_cad else "N/A",
                      delta=_fmt_delta(ref_cad, ref_cad_b, "{:+.0f} rpm"), delta_color="off",
                      help=t("Your typical cadence while pedalling in this period. Used as the quadrant reference."))
            m2.metric(t("High-force time (QI + QII)"),
                      f"{(quads['QI'] + quads['QII']):.0%}" if quads else "N/A",
                      delta=(f"{(quads['QI'] + quads['QII']) - (quads_b['QI'] + quads_b['QII']):+.0%}"
                             if quads and quads_b else None), delta_color="off",
                      help=t("Share of pedalling time pushing harder than the force that produces FTP at your "
                           "median cadence: climbing, accelerations, hard efforts."))
            fit_c = co.linear_fit(cad_a, y_bin=5)
            m3.metric(t("Cadence rise per +50 W"), f"{fit_c['slope'] * 50:.1f} rpm" if fit_c else "N/A",
                      help=t("How much you spin faster as power goes up. Near zero = you mostly push harder "
                           "at the same cadence (often seen on climbs or with a big gear)."))

            cc1, cc2 = st.columns(2)
            cc1.plotly_chart(plot_joint_heatmap(co.aggregate(cad_a), 5, t("Cadence (rpm)"),
                                                t("Time at power × cadence · {lab_a}", lab_a=lab_a),
                                                overlay=co.mean_y_by_power(cad_a, y_bin=5)),
                             use_container_width=True)
            cc2.plotly_chart(plot_cadence_by_zone(co.cadence_by_zone(cad_a, ftp)), use_container_width=True)

            if quads:
                st.plotly_chart(plot_quadrants(co.aggregate(cad_a), ftp, ref_cad, crank_mm / 1000),
                                use_container_width=True)
                rows = [{"Quadrant": q, "Meaning": tr(co.QUADRANT_NAMES[q]), lab_a: f"{quads[q]:.0%}"} for q in quads]
                if quads_b:
                    for r in rows:
                        r[lab_b] = f"{quads_b[r['Quadrant']]:.0%}"
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                st.caption(t('Lines at your median cadence ({ref_cad:.0f} rpm) and at the pedal force that gives your FTP '
                                '({ftp} W) at that cadence; the red curve is FTP. QII-heavy riding builds muscular force '
                                '(climbing, low-cadence work); QIV is fast, light spinning.', ref_cad=ref_cad, ftp=ftp))

            cad_all = pd.concat([x for x in [cad_b, cad_a] if x is not None and not x.empty])
            weekly_cad = co.weekly(cad_all, lambda g: co.weighted_median_y(g) if g['seconds'].sum() >= 600 else None)
            if len(weekly_cad) >= 2:
                st.plotly_chart(plot_trend({t("Median cadence"): weekly_cad}, t("rpm"), t("Weekly median cadence")),
                                use_container_width=True)

    # ---- Recovery & efficiency ----
    if section == sections[2]:
        in_a = (stats['date'] >= a0) & (stats['date'] < a1 + pd.Timedelta(days=1)) if not stats.empty else None
        hrr = stats.dropna(subset=['hrr60_bpm']) if not stats.empty else stats
        endu = co.endurance_rides(stats)

        m1, m2, m3 = st.columns(3)
        def med(df, col, mask_a=True):
            if df is None or df.empty:
                return None
            sel = df[col][in_a.reindex(df.index, fill_value=False)] if mask_a else \
                  df[col][~in_a.reindex(df.index, fill_value=False)]
            sel = sel.dropna()
            return float(sel.median()) if len(sel) else None
        hrr_a, hrr_b = med(hrr, 'hrr60_bpm'), (med(hrr, 'hrr60_bpm', False) if compare else None)
        ef_a, ef_b = med(endu, 'ef'), (med(endu, 'ef', False) if compare else None)
        dec_a, dec_b = med(endu, 'decoupling_pct'), (med(endu, 'decoupling_pct', False) if compare else None)
        m1.metric(t("HR recovery (60 s)"), f"{hrr_a:.0f} bpm" if hrr_a else "N/A",
                  delta=_fmt_delta(hrr_a, hrr_b, "{:+.0f} bpm"),
                  help=t("Median heart-rate drop in the minute after hard efforts (≥ 1 min above 105 % FTP "
                       "followed by easy riding). Faster recovery generally reflects fitness; a sudden drop "
                       "can be a sign of accumulated fatigue."))
        m2.metric(t("EF on endurance rides"), f"{ef_a:.2f}" if ef_a else "N/A",
                  delta=_fmt_delta(ef_a, ef_b, "{:+.2f}"),
                  help=t("Median NP / HR on rides of 60+ min with VI ≤ 1.10, so only comparable steady rides count."))
        m3.metric(t("Decoupling on endurance rides"), f"{dec_a:.1f} %" if dec_a is not None else "N/A",
                  delta=_fmt_delta(dec_a, dec_b, "{:+.1f} pp"), delta_color="inverse",
                  help=t("Median power:HR drift on the same rides. Below ~5 % suggests solid aerobic endurance."))

        if hrr is not None and not hrr.empty:
            st.plotly_chart(plot_trend({t("HRR60"): hrr.set_index('date')['hrr60_bpm']}, t("bpm drop in 60 s"),
                                       t("Heart-rate recovery after hard efforts"), rolling=5),
                            use_container_width=True)
        else:
            st.info(t("No hard efforts followed by easy riding found (needs power + HR)."))
        if endu is not None and not endu.empty:
            st.plotly_chart(plot_trend({"EF": endu.set_index('date')['ef']}, "NP / HR",
                                       t("Efficiency factor on endurance rides"), rolling=5),
                            use_container_width=True)
        else:
            st.info(t("No steady endurance rides (60+ min, VI ≤ 1.10, power + HR) in this period."))
