"""Per-climb performance estimator: change weight, CdA, rolling resistance and power, see the time."""
import numpy as np
import pandas as pd
import streamlit as st

from src.physics import climb_model as cm
from src.physics.aerodyn import get_air_density, get_avg_cda
from src.viz.formatting import format_duration

BIKE_MASS_KG = 8.0                # same bike weight the ride analysis assumes
DEFAULT_CRR = 0.005               # same as the ride analysis
DEFAULT_CDA = 0.32                # upright on the hoods; used when the ride has no usable CdA estimate
EFFICIENCY = 0.97                 # 3 % drivetrain loss, as in the ride analysis


def climb_baseline(df, climb, settings):
    """Inputs that reproduce how the climb was ridden: the profile, power, mass, CdA, Crr, air density."""
    seg = df.loc[climb['start_idx']:climb['end_idx']]
    ele_col = 'ele_smooth' if 'ele_smooth' in seg.columns else 'ele'
    lengths, grades = cm.climb_profile(seg['dist_m'], seg[ele_col])
    power = float(seg['power'].fillna(0).mean()) if 'power' in seg.columns and seg['power'].notna().any() else None
    cda = df.attrs.get('cda_est')
    if cda is None:
        cda = get_avg_cda(df['cda']) if 'cda' in df.columns else np.nan
    t = seg['time']
    return {
        'lengths': lengths, 'grades': grades,
        'actual_s': float((t.iloc[-1] - t.iloc[0]).total_seconds()) if len(t) > 1 else None,
        'power': power,
        'mass': float(settings['weight']) + BIKE_MASS_KG,
        'cda': float(cda) if pd.notna(cda) and 0.15 <= cda <= 0.6 else DEFAULT_CDA,
        'crr': DEFAULT_CRR,
        'rho': float(get_air_density(float(seg[ele_col].mean()),
                                     seg['temp_c'].mean() if 'temp_c' in seg.columns and seg['temp_c'].notna().any() else None,
                                     seg['relative_humidity'].mean() if 'relative_humidity' in seg.columns and seg['relative_humidity'].notna().any() else 0.0)),
    }


def _mmss(seconds):
    return format_duration(int(round(seconds)))


def _delta(seconds):
    sign = "-" if seconds < 0 else "+"
    s = int(round(abs(seconds)))
    return f"{sign}{s // 60}:{s % 60:02d}"


def render_climb_estimator(df, climb, index, settings, ride_id):
    base = climb_baseline(df, climb, settings)
    if len(base['lengths']) == 0:
        st.caption("This climb is too short to model.")
        return
    k = lambda name: f"clb_{ride_id}_{index}_{name}"          # per ride and climb, so values don't leak
    default_power = min(1500.0, max(20.0, base['power'] if base['power'] else float(settings.get('ftp') or 200) * 0.85))
    eff = EFFICIENCY

    st.markdown("**⏱️ Performance estimator**")
    st.caption("What would this climb take with a different weight, aerodynamics or power? Steady-state model "
               "on the climb's gradient profile, no wind. The *as ridden* line applies the model to what you "
               "actually did, so you can see how close it gets.")
    mode = st.radio("Estimate", ["Time for a power", "Power for a target time"], horizontal=True, key=k("mode"))

    c = st.columns(5)
    mass = c[0].number_input("System weight (kg)", 30.0, 200.0, min(200.0, max(30.0, round(base['mass'], 1))), 0.5, key=k("mass"),
                             help="Rider + bike + clothing + bottles. Default: your profile weight + 8 kg bike.")
    cda = c[1].number_input("CdA (m²)", 0.15, 0.70, round(base['cda'], 3), 0.005, format="%.3f", key=k("cda"),
                            help="Default: this ride's average estimate if it had one, else 0.32. "
                                 "Climbing at low speed, so it matters less than weight.")
    crr = c[2].number_input("Rolling resistance (Crr)", 0.001, 0.020, base['crr'], 0.0005, format="%.4f",
                            key=k("crr"), help="0.003–0.004 good road tyres, 0.005 typical, 0.008+ rough or gravel.")
    rho = c[3].number_input("Air density (kg/m³)", 0.7, 1.4, min(1.4, max(0.7, round(base['rho'], 3))), 0.01, format="%.3f",
                            key=k("rho"), help="From the climb's altitude; lower at altitude and in the heat.")

    if mode == "Time for a power":
        power = c[4].number_input("Power (W)", 20, 1500, int(round(default_power)), 5, key=k("power"),
                                  help="Default: your average power over the climb (zeros included)."
                                  if base['power'] else "No power data on this ride: defaults to 85 % of FTP.")
        scenario = cm.climb_time(base['lengths'], base['grades'], power, mass, cda, crr, rho, eff)
        s_time = scenario['time_s']
    else:
        target_min = c[4].number_input("Target time (min)", 1.0, 600.0,
                                       min(600.0, max(1.0, round((base['actual_s'] or 1200) / 60, 1))), 0.5, key=k("target"))
        power = cm.power_for_time(base['lengths'], base['grades'], target_min * 60, mass, cda, crr, rho, eff)
        if power is None:
            st.warning("That time can't be reached with these settings (it needs under 20 W or over 1500 W).")
            return
        scenario = cm.climb_time(base['lengths'], base['grades'], power, mass, cda, crr, rho, eff)
        s_time = scenario['time_s']

    asridden = None
    if base['power']:
        asridden = cm.climb_time(base['lengths'], base['grades'], base['power'], base['mass'], base['cda'],
                                 base['crr'], base['rho'], eff)

    m = st.columns(4)
    m[0].metric("Actual time", _mmss(base['actual_s']) if base['actual_s'] else "N/A",
                help="Elapsed time on the climb, stops included.")
    if asridden:
        m[1].metric("Model, as ridden", _mmss(asridden['time_s']),
                    delta=_delta(asridden['time_s'] - base['actual_s']) + " vs actual" if base['actual_s'] else None,
                    delta_color="off", help="The model with your actual power and the default weight, CdA and Crr.")
    m[2].metric("Scenario", _mmss(s_time),
                delta=(_delta(s_time - asridden['time_s']) + " vs as ridden") if asridden else None,
                delta_color="inverse")
    m[3].metric("Power needed" if mode != "Time for a power" else "Avg speed",
                f"{power:.0f} W ({power / max(mass - BIKE_MASS_KG, 1):.2f} W/kg)" if mode != "Time for a power"
                else f"{scenario['avg_speed_kmh']:.1f} km/h",
                help="W/kg uses system weight minus the 8 kg bike." if mode != "Time for a power" else None)
    st.caption(f"VAM {scenario['vam_m_h']:.0f} m/h · {power / max(mass - BIKE_MASS_KG, 1):.2f} W/kg"
               + (" · Model and actual differ because of wind, surface, drafting, pacing and gradient noise."
                  if asridden else ""))

    # What each change is worth at the chosen scenario
    if mode == "Time for a power":
        def t(**kw):
            a = dict(power_w=power, mass_kg=mass, cda=cda, crr=crr)
            a.update(kw)
            return cm.climb_time(base['lengths'], base['grades'], a['power_w'], a['mass_kg'], a['cda'], a['crr'],
                                 rho, eff)['time_s']
        rows = [("−1 kg system weight", t(mass_kg=mass - 1) - s_time),
                ("−0.01 m² CdA", t(cda=max(cda - 0.01, 0.05)) - s_time),
                ("−0.001 Crr", t(crr=max(crr - 0.001, 0.0005)) - s_time),
                ("+10 W", t(power_w=power + 10) - s_time)]
        st.dataframe(pd.DataFrame({"Change": [r[0] for r in rows],
                                   "Time": [f"{r[1]:+.0f} s" for r in rows]}),
                     hide_index=True, use_container_width=True)
    if st.button("Reset to ride values", key=k("reset")):
        for name in ("mass", "cda", "crr", "rho", "power", "target"):
            st.session_state.pop(k(name), None)
        st.rerun()
