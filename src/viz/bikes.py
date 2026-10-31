"""Bikes in the User Corner (frame, wheelset, groupset, tyres) and the bike picker of the ride page."""
import hashlib

import streamlit as st

from src.analysis import bikes as bk
from src.data import db
from src.data.cached import get_bikes

PART_HELP = {
    "frame": "Frame and fork (and seatpost, if you like).",
    "wheelset": "Both wheels with their skewers or axles, without tyres.",
    "groupset": "Cranks and chainrings, cassette, chain, derailleurs, shifters and brakes.",
    "tyres": "Both tyres, plus tubes or sealant if you use them.",
}
CRR_HELP = ("Rolling resistance of the tyres. About 0.003–0.004 for good road tyres, 0.005 typical, 0.008 and more for "
            "rough roads or gravel. Independent tyre tests give measured values.")
EXTRAS_HELP = "Whatever the four parts leave out: pedals, handlebar and stem, saddle, bottles, computer, lights."


def _bike_form(user_name, bike=None):
    """New bike (bike=None) or the edit form of an existing one. Saves and reruns."""
    tag = f"{user_name}_{bike['id'] if bike else 'new'}"
    b = bike or {}
    with st.form(f"bike_form_{tag}", clear_on_submit=not bike):
        name = st.text_input("Bike name", value=b.get('name', ""), placeholder="Road, Gravel, TT…",
                             key=f"bike_name_{tag}")
        parts = {}
        for part in bk.PARTS:
            c1, c2 = st.columns([3, 1])
            parts[f"{part}_name"] = c1.text_input(part.capitalize(), value=b.get(f"{part}_name") or "",
                                                  placeholder="make and model (optional)", help=PART_HELP[part],
                                                  key=f"bike_{part}_{tag}")
            parts[f"{part}_kg"] = c2.number_input("kg", value=b.get(f"{part}_kg"), min_value=0.0, max_value=20.0,
                                                  step=0.05, format="%.2f", key=f"kg_{part}_{tag}")
        c1, c2 = st.columns(2)
        parts['extras_kg'] = c1.number_input("Extras (kg) · optional", value=b.get('extras_kg'), min_value=0.0,
                                             max_value=10.0, step=0.05, format="%.2f", help=EXTRAS_HELP,
                                             key=f"extras_{tag}")
        parts['crr'] = c2.number_input("Crr (tyres)", value=float(b.get('crr') or 0.005), min_value=bk.CRR_RANGE[0],
                                       max_value=bk.CRR_RANGE[1], step=0.0005, format="%.4f", help=CRR_HELP,
                                       key=f"crr_{tag}")
        if st.form_submit_button("Save bike" if bike else "Add bike", type="primary"):
            ok, msg = db.update_bike(bike['id'], name, parts) if bike else db.save_bike(user_name, name, parts)
            if ok:
                st.rerun()
            else:
                st.error(msg)


def render_bikes(user_name):
    """User Corner section: the rider's bikes, each built from its parts."""
    st.subheader("🚲 Bikes")
    st.caption("Describe each bike by its parts. Their weights add up to the bike's weight and the tyres give the "
               "rolling resistance; you then choose the bike for each ride, and the ride analysis uses them. "
               "CdA is not entered here: it is computed from every ride.")
    bikes = get_bikes(user_name)
    for bike in bikes:
        weight, crr = bk.bike_physics(bike)
        star = " · default" if bike['is_default'] else ""
        with st.expander(f"{bike['name']}{star} — {weight:.2f} kg, Crr {crr:.4f}"):
            rows = [f"**{p.capitalize()}**: {bike.get(f'{p}_name') or '–'} · "
                    f"{bike[f'{p}_kg']:.2f} kg" if bike.get(f'{p}_kg') is not None else
                    f"**{p.capitalize()}**: {bike.get(f'{p}_name') or '–'}" for p in bk.PARTS]
            if bike.get('extras_kg'):
                rows.append(f"**Extras**: {bike['extras_kg']:.2f} kg")
            st.markdown("  \n".join(rows))
            if not bk.bike_weight_kg(bike) > 0 or bike.get('crr') is None:
                st.caption("Without weights the generic 8 kg is used; without a Crr, 0.005.")
            _bike_form(user_name, bike)
            a, b = st.columns(2)
            if not bike['is_default'] and a.button("Make it my default", key=f"bike_default_{bike['id']}",
                                                   help="New rides start with this bike."):
                db.set_default_bike(user_name, bike['id'])
                st.rerun()
            with b.popover("Delete bike"):
                st.warning("Rides done on it go back to your default bike. This cannot be undone.")
                if st.button(f"Yes, delete {bike['name']}", type="primary", key=f"bike_delete_{bike['id']}"):
                    db.delete_bike(bike['id'])
                    st.rerun()
    with st.expander("➕ Add a bike", expanded=not bikes):
        _bike_form(user_name)


def choose_ride_bike(user_name, file_bytes):
    """
    Bike picker at the top of a ride. Returns (bike, bike_id): the bike dict (None for the generic one) and what to
    store with the ride (a bike id, 0 for the generic bike, None when the rider has no bikes). The choice of a saved
    ride is kept; a new ride starts with the default bike.
    """
    bikes = get_bikes(user_name) if user_name else []
    if not bikes:
        if user_name:
            st.caption("🚲 Define your bikes in the User Corner to use their weight and rolling resistance; "
                       f"meanwhile the {bk.GENERIC_NAME.lower()} is used.")
        return None, None
    by_id = {b['id']: b for b in bikes}
    saved = db.get_ride_bike(user_name, file_bytes)
    saved_id = saved[1] if saved else None
    default = bk.default_bike(bikes)
    start = saved_id if saved_id == 0 or saved_id in by_id else (default['id'] if default else 0)

    options = [0] + [b['id'] for b in bikes]
    chosen = st.selectbox("🚲 Bike for this ride", options, index=options.index(start),
                          key=f"ride_bike_{hashlib.md5(file_bytes).hexdigest()}",
                          format_func=lambda i: bk.label(by_id.get(i)),
                          help="Sets the bike's weight and the tyres' Crr in the power balance. CdA is computed "
                               "from the ride, not taken from the bike.")
    if saved and saved_id != chosen:
        db.set_ride_bike(saved[0], chosen)
    return by_id.get(chosen), chosen
