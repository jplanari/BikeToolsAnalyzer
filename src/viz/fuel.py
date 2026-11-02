"""The energy and fuel panel of the Ride Summary (see src/analysis/nutrition.py for the model)."""
import streamlit as st

from src.analysis import nutrition as nut
from src.i18n import t, tr


def render_fuel_summary(summary, weight):
    """Energy burned and the fuelling that goes with it, estimated from the power."""
    st.subheader(t("Energy and fuel"))
    kcal, carb = nut.kcal_from_kj(summary['kj']), summary['carb_g']
    fat = nut.fat_grams(summary['kj'], carb)
    low, high = nut.target_total(summary['moving_time_s'], summary.get('if'))
    r_low, r_high = nut.target_rate(summary['moving_time_s'], summary.get('if'))
    f1, f2, f3, f4 = st.columns(4)
    f1.metric(t("Energy burned"), f"{kcal:,.0f} kcal",
              help=t("Estimated from the work done on the pedals (about 24 % efficiency, so kcal is close to kJ). "
                     "Resting energy while coasting or stopped is not included."))
    f2.metric(t("Carbohydrate burned"), f"{carb:,.0f} g",
              help=t("Estimated from your power relative to FTP with a typical fuel-mix curve: the harder you ride, "
                     "the bigger the carbohydrate share. An estimate; individuals differ."))
    f3.metric(t("Fat burned"), f"{fat:,.0f} g" if fat is not None else t("N/A"))
    if high:
        f4.metric(t("Fuelling guide"), f"{r_low:.0f}–{r_high:.0f} g/h" if r_low != r_high else f"{r_high:.0f} g/h",
                  help=t("Carbohydrate worth taking in per hour for a ride of this length and intensity "
                         "(sports-nutrition guidance, not medical advice)."))
    else:
        f4.metric(t("Fuelling guide"), t("Nothing needed"),
                  help=t("Rides under about 45 minutes do not need fuel beyond a normal meal."))
    used = nut.store_used(carb, weight)
    word = nut.load(used)
    if word and high:
        eaten = nut.store_used(carb, weight, eaten_g=low)
        if eaten < 0.05:
            st.caption(t("{word} carbohydrate load: about {used:.0%} of a full glycogen store (~{store:.0f} g) if nothing "
                         "is eaten; the guide ({low:.0f}–{high:.0f} g over the ride) covers it.",
                         word=tr(nut.LOADS[word]), used=used, store=nut.glycogen_store(weight), low=low, high=high))
        else:
            st.caption(t("{word} carbohydrate load: about {used:.0%} of a full glycogen store (~{store:.0f} g) if nothing is "
                         "eaten; taking in the guide, about {left:.0%}. In total, {low:.0f}–{high:.0f} g over the ride.",
                         word=tr(nut.LOADS[word]), used=used, store=nut.glycogen_store(weight), left=eaten,
                         low=low, high=high))
    elif word:
        st.caption(t("{word} carbohydrate load: about {used:.0%} of a full glycogen store (~{store:.0f} g).",
                     word=tr(nut.LOADS[word]), used=used, store=nut.glycogen_store(weight)))
