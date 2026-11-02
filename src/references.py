"""
The literature behind the app's numbers.

Physics and physiology are what this app stands on, so every model it uses is listed here with the work it comes
from and the part of the app that relies on it. The References page shows this list; `python -m src.references`
prints it as Markdown. When a new model, constant or threshold goes into the code, add its source here (a test checks
that the list stays well formed).

Each entry: key, topic, authors, year (None for standards and handbooks without one), title, source (journal with
volume and pages, or publisher), optional doi, and `used`: what in the app depends on it. Bibliographic fields are
not translated; topics and `used` are.
"""
from src.i18n import mark

TOPICS = [
    mark("Power and training load"),
    mark("Critical power and W′"),
    mark("Endurance and recovery"),
    mark("Aerodynamics and ride mechanics"),
    mark("Heat balance"),
    mark("Energy and fuel"),
    mark("Left / right balance"),
    mark("Data"),
]
POWER, CP, ENDURANCE, MECHANICS, HEAT, FUEL, BALANCE, DATA = TOPICS


def _ref(key, topic, authors, year, title, source, used, doi=None):
    return {'key': key, 'topic': topic, 'authors': authors, 'year': year, 'title': title, 'source': source,
            'doi': doi, 'used': used}


REFERENCES = [
    # --- power and training load ----------------------------------------------------------------------------------
    _ref('allen-coggan-2010', POWER, "Allen H, Coggan AR", 2010,
         "Training and Racing with a Power Meter, 2nd ed.", "VeloPress, Boulder",
         mark("Normalized power, intensity factor, TSS, the power zones, FTP from the best 20 minutes, the power "
              "profile durations, pedal force and velocity (quadrant analysis) and the performance management chart.")),
    _ref('banister-1975', POWER, "Banister EW, Calvert TW, Savage MV, Bach T", 1975,
         "A systems model of training for athletic performance", "Australian Journal of Sports Medicine 7:57–61",
         mark("The impulse-response idea behind fitness (CTL), fatigue (ATL) and form (TSB), with 42-day and 7-day "
              "exponentially weighted averages of the daily load.")),
    _ref('foster-1998', POWER, "Foster C", 1998,
         "Monitoring training in athletes with reference to overtraining syndrome",
         "Medicine & Science in Sports & Exercise 30(7):1164–1168",
         mark("Training monotony and strain over the last seven days.")),
    _ref('gabbett-2016', POWER, "Gabbett TJ", 2016,
         "The training–injury prevention paradox: should athletes be training smarter and harder?",
         "British Journal of Sports Medicine 50(5):273–280",
         mark("The acute:chronic workload ratio and the 0.8–1.3 range quoted next to it.")),
    _ref('impellizzeri-2020', POWER, "Impellizzeri FM, Tenan MS, Kempton T, Novak A, Coutts AJ", 2020,
         "Acute:chronic workload ratio: conceptual issues and fundamental pitfalls",
         "International Journal of Sports Physiology and Performance 15(6):907–913",
         mark("Why the acute:chronic ratio should be read as a rough guide: it is not a validated predictor of injury.")),
    _ref('friel-2018', POWER, "Friel J", 2018,
         "The Cyclist's Training Bible, 5th ed.", "VeloPress, Boulder",
         mark("Heart-rate zones from LTHR, LTHR as 95 % of the best 20-minute heart rate, the efficiency factor and "
              "aerobic decoupling (Pa:HR). The heart-rate based TSS is the app's own approximation.")),

    # --- critical power -------------------------------------------------------------------------------------------
    _ref('monod-scherrer-1965', CP, "Monod H, Scherrer J", 1965,
         "The work capacity of a synergic muscular group", "Ergonomics 8(3):329–338",
         mark("The critical power model: work done is linear in time, its slope is CP and its intercept W′.")),
    _ref('morton-1996', CP, "Morton RH", 1996,
         "A 3-parameter critical power model", "Ergonomics 39(4):611–619",
         mark("The three-parameter fit of the power-duration curve, which also describes the short end of the curve.")),
    _ref('jones-2010', CP, "Jones AM, Vanhatalo A, Burnley M, Morton RH, Poole DC", 2010,
         "Critical power: implications for determination of V̇O2max and exercise tolerance",
         "Medicine & Science in Sports & Exercise 42(10):1876–1890",
         mark("Fitting CP and W′ to maximal efforts of about 3 to 20 minutes, and what the parameters mean.")),
    _ref('poole-2016', CP, "Poole DC, Burnley M, Vanhatalo A, Rossiter HB, Jones AM", 2016,
         "Critical power: an important fatigue threshold in exercise physiology",
         "Medicine & Science in Sports & Exercise 48(11):2320–2334",
         mark("The physiological meaning of CP as a boundary between sustainable and unsustainable intensity.")),
    _ref('skiba-2012', CP, "Skiba PF, Chidnok W, Vanhatalo A, Jones AM", 2012,
         "Modeling the expenditure and reconstitution of work capacity above critical power",
         "Medicine & Science in Sports & Exercise 44(8):1526–1532",
         mark("The W′ balance during a ride, in its integral form, with the app's fixed recovery time constant "
              "of 546 s.")),

    # --- endurance and recovery -----------------------------------------------------------------------------------
    _ref('maunder-2021', ENDURANCE, "Maunder E, Seiler S, Mildenhall MJ, Kilding AE, Plews DJ", 2021,
         "The importance of ‘durability’ in the physiological profiling of endurance athletes",
         "Sports Medicine 51(8):1619–1628",
         mark("Durability: the power you can still produce after 1000, 2000 and 3000 kJ of work.")),
    _ref('seiler-2010', ENDURANCE, "Seiler S", 2010,
         "What is best practice for training intensity and duration distribution in endurance athletes?",
         "International Journal of Sports Physiology and Performance 5(3):276–291",
         mark("The three-zone view (easy, moderate, hard) of the training intensity distribution; the app groups its "
              "zones as Z1–2, Z3–4 and Z5+.")),
    _ref('treff-2019', ENDURANCE, "Treff G, Winkert K, Sareban M, Steinacker JM, Sperlich B", 2019,
         "The Polarization-Index: a simple calculation to distinguish polarized from non-polarized training "
         "intensity distributions", "Frontiers in Physiology 10:707",
         mark("The polarization index, with 2.0 as the limit for a polarized distribution."),
         doi="10.3389/fphys.2019.00707"),
    _ref('daanen-2012', ENDURANCE, "Daanen HAM, Lamberts RP, Kallen VL, Jin A, Van Meeteren NLU", 2012,
         "A systematic review on heart-rate recovery to monitor changes in training status in athletes",
         "International Journal of Sports Physiology and Performance 7(3):251–260",
         mark("Heart-rate recovery after hard efforts (HRR60) as a sign of training status.")),

    # --- aerodynamics and mechanics -------------------------------------------------------------------------------
    _ref('martin-1998', MECHANICS, "Martin JC, Milliken DL, Cobb JE, McFadden KL, Coggan AR", 1998,
         "Validation of a mathematical model for road cycling power", "Journal of Applied Biomechanics 14(3):276–291",
         mark("The power balance of aerodynamic drag, rolling resistance, gravity and acceleration, behind the CdA "
              "estimate, the power budget and the climb time model.")),
    _ref('wilson-2004', MECHANICS, "Wilson DG, Papadopoulos J", 2004,
         "Bicycling Science, 3rd ed.", "MIT Press, Cambridge MA",
         mark("Typical rolling resistance, drivetrain loss and drag values, and how they combine.")),
    _ref('iso-2533', MECHANICS, "International Organization for Standardization", 1975,
         "ISO 2533:1975 Standard Atmosphere", "ISO, Geneva",
         mark("Pressure and temperature with altitude, for the air density when the weather has no value.")),
    _ref('alduchov-1996', MECHANICS, "Alduchov OA, Eskridge RE", 1996,
         "Improved Magnus form approximation of saturation vapor pressure",
         "Journal of Applied Meteorology 35(4):601–609",
         mark("Saturation vapour pressure, for the effect of humidity on air density and on sweat evaporation.")),

    # --- heat balance ---------------------------------------------------------------------------------------------
    _ref('gagge-1971', HEAT, "Gagge AP, Stolwijk JAJ, Nishi Y", 1971,
         "An effective temperature scale based on a simple model of human physiological regulatory response",
         "ASHRAE Transactions 77(1):247–262",
         mark("The two-node (core and skin) model of the thermal analysis: skin conductance, sweating and "
              "vasodilation.")),
    _ref('iso-7730', HEAT, "International Organization for Standardization", 2005,
         "ISO 7730:2005 Ergonomics of the thermal environment", "ISO, Geneva",
         mark("Resting metabolic rate (1 met = 58.2 W/m²) and the heat lost through breathing.")),
    _ref('ashrae-fundamentals', HEAT, "ASHRAE", None,
         "ASHRAE Handbook – Fundamentals, chapter ‘Thermal comfort’", "ASHRAE, Peachtree Corners GA",
         mark("Convective and radiative heat transfer in still air.")),
    _ref('saltin-hermansen-1966', HEAT, "Saltin B, Hermansen L", 1966,
         "Esophageal, rectal, and muscle temperature during exercise", "Journal of Applied Physiology 21(6):1757–1762",
         mark("Core temperature at a steady state rises with relative exercise intensity, which moves the sweating "
              "and vasodilation thresholds in the model.")),
    _ref('ettema-2009', HEAT, "Ettema G, Lorås HW", 2009,
         "Efficiency in cycling: a review", "European Journal of Applied Physiology 106(1):1–14",
         mark("Gross efficiency of about 23–24 %, which turns pedal work into the heat the body must lose and into "
              "the energy burned.")),

    # --- energy and fuel ------------------------------------------------------------------------------------------
    _ref('brooks-mercier-1994', FUEL, "Brooks GA, Mercier J", 1994,
         "Balance of carbohydrate and lipid utilization during exercise: the ‘crossover’ concept",
         "Journal of Applied Physiology 76(6):2253–2261",
         mark("The share of energy from carbohydrate rising with intensity; the app's fuel-mix curve is a simplified "
              "version of it.")),
    _ref('romijn-1993', FUEL, "Romijn JA, Coyle EF, Sidossis LS, Gastaldelli A, Horowitz JF, Endert E, Wolfe RR", 1993,
         "Regulation of endogenous fat and carbohydrate metabolism in relation to exercise intensity and duration",
         "American Journal of Physiology 265(3):E380–E391",
         mark("Fat and carbohydrate use at low, moderate and high intensity, to shape the fuel-mix curve.")),
    _ref('burke-2011', FUEL, "Burke LM, Hawley JA, Wong SHS, Jeukendrup AE", 2011,
         "Carbohydrates for training and competition", "Journal of Sports Sciences 29(S1):S17–S27",
         mark("Carbohydrate intake during rides by duration; the glycogen store of 6 g per kg is the app's own "
              "approximation.")),
    _ref('jeukendrup-2014', FUEL, "Jeukendrup AE", 2014,
         "A step towards personalized sports nutrition: carbohydrate intake during exercise",
         "Sports Medicine 44(S1):S25–S33",
         mark("The shape of the fuelling guide: up to 30–60 g/h for rides to 2 hours, 60–90 g/h for longer ones.")),
    _ref('smith-2013', FUEL, "Smith JW, Pascoe DD, Passe DH, Ruby BC, Stewart LK, Baker LB, Zachwieja JJ", 2013,
         "Curvilinear dose-response relationship of carbohydrate (0–120 g·h⁻¹) and performance",
         "Medicine & Science in Sports & Exercise 45(2):336–341",
         mark("Performance in cyclists rises with carbohydrate intake and levels off at about 60–80 g/h, the "
              "background to comparing rides with and without enough fuel."),
         doi="10.1249/MSS.0b013e31827205d1"),
    _ref('thomas-2016', FUEL, "Thomas DT, Erdman KA, Burke LM", 2016,
         "Position of the Academy of Nutrition and Dietetics, Dietitians of Canada, and the American College of "
         "Sports Medicine: nutrition and athletic performance",
         "Journal of the Academy of Nutrition and Dietetics 116(3):501–528",
         mark("The consensus position the fuelling guide follows: guidance by session length, not a prescription.")),

    # --- left / right balance -------------------------------------------------------------------------------------
    _ref('carpes-2010', BALANCE, "Carpes FP, Mota CB, Faria IE", 2010,
         "On the bilateral asymmetry during running and cycling – a review considering leg preference",
         "Physical Therapy in Sport 11(4):136–142",
         mark("Some left / right asymmetry is normal in cyclists; the app treats 48–52 % as even, a rule of thumb "
              "of its own.")),
    _ref('bini-hume-2014', BALANCE, "Bini RR, Hume P", 2014,
         "Assessment of bilateral asymmetry in cycling using a commercial instrumented crank system and "
         "instrumented pedals", "International Journal of Sports Physiology and Performance 9(5):876–881",
         mark("What commercial pedal and crank power meters measure when they report the split between the legs."),
         doi="10.1123/ijspp.2013-0494"),
    _ref('garmin-fit', BALANCE, "Garmin International", None,
         "Flexible and Interoperable Data Transfer (FIT) protocol, SDK and profile",
         "Garmin Ltd., developer.garmin.com/fit",
         mark("The encoding of the left / right balance, torque effectiveness and pedal smoothness fields in FIT files.")),

    # --- data -----------------------------------------------------------------------------------------------------
    _ref('open-meteo', DATA, "Zippenfenig P", 2023,
         "Open-Meteo.com Weather API", "Zenodo",
         mark("Hourly wind, temperature, pressure and humidity along each ride."),
         doi="10.5281/zenodo.7970649"),
]


def by_topic():
    """[(topic, [reference, ...])] in the order of TOPICS, each topic's entries in the order of REFERENCES."""
    return [(topic, [r for r in REFERENCES if r['topic'] == topic]) for topic in TOPICS]


def citation(ref):
    """The reference as a line of Markdown: authors (year). Title. *Source*. DOI link."""
    year = f"({ref['year']})" if ref['year'] else "(n.d.)"
    title = ref['title'] if ref['title'][-1] in ".?!" else ref['title'] + "."
    text = f"{ref['authors']} {year}. {title} *{ref['source']}*."
    if ref.get('doi'):
        text += f" [doi:{ref['doi']}](https://doi.org/{ref['doi']})"
    return text


def to_markdown(translate=lambda s: s):
    """The whole list as Markdown, for documentation. `translate` renders topics and uses in another language."""
    lines = []
    for topic, refs in by_topic():
        lines += [f"## {translate(topic)}", ""]
        for r in refs:
            lines += [f"- {citation(r)}", f"  *{translate(r['used'])}*"]
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(to_markdown())
