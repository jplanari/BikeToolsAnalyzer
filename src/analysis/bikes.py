"""
What a bike contributes to the physics of a ride: its weight (the sum of its parts) and the rolling resistance of its
tyres. No Streamlit. CdA is deliberately absent: it is computed from each ride, never entered with a bike.
"""
from src.physics.aerodyn import BIKE_MASS_KG, CRR

GENERIC_NAME = "Generic bike"
PARTS = ("frame", "wheelset", "groupset", "tyres")
CRR_RANGE = (0.001, 0.020)


def bike_weight_kg(bike):
    """Frame + wheelset + groupset + tyres + extras (kg); parts left empty count as 0."""
    if not bike:
        return BIKE_MASS_KG
    return round(sum(float(bike.get(f"{p}_kg") or 0.0) for p in PARTS) + float(bike.get('extras_kg') or 0.0), 3)


def bike_crr(bike):
    c = (bike or {}).get('crr')
    return float(c) if c and CRR_RANGE[0] <= float(c) <= CRR_RANGE[1] else CRR


def bike_physics(bike):
    """(bike mass in kg, Crr) for the power balance; the generic values for None or a bike with no weights yet."""
    if not bike:
        return BIKE_MASS_KG, CRR
    w = bike_weight_kg(bike)
    return (w if w > 0 else BIKE_MASS_KG), bike_crr(bike)


def default_bike(bikes):
    return next((b for b in bikes if b.get('is_default')), None)


def label(bike):
    if not bike:
        return f"{GENERIC_NAME} ({BIKE_MASS_KG:.1f} kg, Crr {CRR:.4f})"
    m, c = bike_physics(bike)
    return f"{bike['name']} ({m:.1f} kg, Crr {c:.4f})"
