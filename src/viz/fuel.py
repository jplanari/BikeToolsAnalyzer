"""The energy and fuel panel of the Ride Summary (see src/analysis/nutrition.py for the model)."""
import streamlit as st

from src.analysis import nutrition as nut
from src.data import db
from src.i18n import t, tr


def render_fuel_summary(summary, weight, ride_id=None, user_name=None):
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
    if ride_id is not None:
        render_fuel_log(summary, weight, ride_id, user_name)
    elif user_name is None and summary.get('carb_g') is not None:
        st.caption(t("Choose a user to log what you ate on this ride."))


def render_fuel_log(summary, weight, ride_id, user_name):
    """Form to log the carbohydrate eaten and the fluid drunk on a saved ride, and what that implies."""
    eaten, fluid = db.get_ride_fuel(user_name, ride_id)
    with st.form(f"fuel_log_{ride_id}"):
        c1, c2, c3 = st.columns([2, 2, 1])
        new_eaten = c1.number_input(t("Carbohydrate eaten (g)"), value=eaten, min_value=0.0,
                                    max_value=db.rides.MAX_CARB_EATEN_G, step=10.0, placeholder=t("not logged"),
                                    help=t("Everything you ate and drank on the ride that has carbohydrate: gels, bars, "
                                           "sports drink, food. Leave empty if you did not log it; 0 means nothing."))
        new_fluid = c2.number_input(t("Fluid drunk (ml)"), value=fluid, min_value=0.0,
                                    max_value=db.rides.MAX_FLUID_ML, step=100.0, placeholder=t("not logged"))
        c3.write("")
        saved = c3.form_submit_button(t("Save"))
    if saved:
        ok, message = db.set_ride_fuel(user_name, ride_id, new_eaten, new_fluid)
        if not ok:
            st.error(t("Could not save: {message}", message=message))
            return
        eaten, fluid = new_eaten, new_fluid
    render_intake_result(summary, eaten, fluid, weight)


def render_intake_result(summary, eaten, fluid, weight):
    """What a ride's log says: intake per hour against the guide, how much of the burned carbohydrate it replaced."""
    if eaten is None and fluid is None:
        return
    r = nut.ride_intake(summary, eaten, fluid, weight)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(t("Carbohydrate eaten"), f"{eaten:.0f} g" if eaten is not None else t("N/A"))
    m2.metric(t("Intake rate"), f"{r['carb_rate']:.0f} g/h" if r['carb_rate'] is not None else t("N/A"),
              help=t("Carbohydrate eaten per hour of moving time."))
    m3.metric(t("Burned carbohydrate replaced"), f"{r['coverage']:.0%}" if r['coverage'] is not None else t("N/A"),
              help=t("Carbohydrate eaten as a share of the estimated amount burned. Under 100 % is normal: the rest "
                     "comes from your glycogen stores."))
    m4.metric(t("Fluid"), f"{r['fluid_rate']:.0f} ml/h" if r['fluid_rate'] is not None else t("N/A"))
    if r['status']:
        word = tr(nut.INTAKES[r['status']])
        left = r['store_used']
        if r['status'] == 'none':
            st.caption(t("{word}: a ride this short does not need fuel beyond a normal meal.", word=word))
        elif left is not None:
            st.caption(t("{word}. After what you ate, the ride used about {left:.0%} of a full glycogen store.",
                         word=word, left=left))
        else:
            st.caption(word)

