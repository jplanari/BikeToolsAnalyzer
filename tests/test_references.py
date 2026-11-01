"""The References page: the list is well formed, and the numbers it quotes are the ones the code uses."""
import inspect
import re

import pytest

from src import references as refs
from src.analysis import durability, nutrition, power, training_load
from src.i18n import tr, use_language


def test_every_reference_is_complete_and_unique():
    keys = [r['key'] for r in refs.REFERENCES]
    assert len(keys) == len(set(keys)) and len(keys) >= 30
    for r in refs.REFERENCES:
        for field in ('authors', 'title', 'source', 'used', 'topic'):
            assert r[field] and r[field].strip(), (r['key'], field)
        assert r['year'] is None or 1900 < r['year'] <= 2026, r['key']
        assert r['topic'] in refs.TOPICS, r['key']
        if r['doi']:
            assert re.fullmatch(r'10\.\d{4,9}/\S+', r['doi']), r['key']


def test_every_topic_has_entries_and_by_topic_keeps_the_order():
    grouped = refs.by_topic()
    assert [t for t, _ in grouped] == refs.TOPICS and all(items for _, items in grouped)
    assert sum(len(items) for _, items in grouped) == len(refs.REFERENCES)


def test_citation_has_no_doubled_punctuation_and_links_the_doi():
    for r in refs.REFERENCES:
        line = refs.citation(r)
        assert not re.search(r'[.?!]\.', line) and '  ' not in line, line
    with_doi = refs.citation(next(r for r in refs.REFERENCES if r['key'] == 'treff-2019'))
    assert "(https://doi.org/10.3389/fphys.2019.00707)" in with_doi and "(2019)" in with_doi
    assert "(n.d.)" in refs.citation(next(r for r in refs.REFERENCES if r['key'] == 'garmin-fit'))


def test_markdown_lists_every_reference_and_topic():
    md = refs.to_markdown()
    assert all(topic in md for topic in refs.TOPICS)
    assert all(r['title'].rstrip('.?!') in md for r in refs.REFERENCES)


def test_topics_and_uses_are_translated():
    with use_language('es'):
        assert tr(refs.TOPICS[0]) != refs.TOPICS[0]
        assert all(tr(r['used']) != r['used'] for r in refs.REFERENCES)
    with use_language('ca'):
        assert all(tr(r['used']) != r['used'] for r in refs.REFERENCES)


# --- the numbers the page quotes are the ones in the code -------------------------------------------------------------

def used(key):
    return next(r['used'] for r in refs.REFERENCES if r['key'] == key)


def test_w_prime_time_constant_is_546_seconds():
    assert "546 s" in used('skiba-2012')
    assert inspect.signature(power.w_prime_expended).parameters['tau'].default == 546
    assert inspect.signature(power.calculate_w_prime_balance).parameters['tau'].default == 546


def test_fitness_and_fatigue_windows_are_42_and_7_days():
    assert "42-day and 7-day" in used('banister-1975')
    assert (training_load.CTL_DAYS, training_load.ATL_DAYS) == (42, 7)


def test_durability_work_thresholds_are_the_ones_listed():
    assert "1000, 2000 and 3000 kJ" in used('maunder-2021')
    assert durability.KJ_THRESHOLDS == (1000, 2000, 3000)


def test_energy_efficiency_is_within_the_quoted_range():
    assert "23–24 %" in used('ettema-2009')
    from src.physics import thermal
    assert 0.23 <= nutrition.GROSS_EFFICIENCY <= 0.24 and 0.23 <= thermal.EFFICIENCY <= 0.24


def test_resting_metabolic_rate_is_the_iso_value():
    from src.physics import thermal
    assert "58.2 W/m²" in used('iso-7730') and thermal.M_REST_W_PER_M2 == 58.2


def test_glycogen_store_and_fuel_bands_match_the_text():
    assert "6 g per kg" in used('burke-2011') and nutrition.GLYCOGEN_G_PER_KG == 6.0
    assert "30–60 g/h" in used('jeukendrup-2014') and "60–90 g/h" in used('jeukendrup-2014')
    assert nutrition.target_rate(90 * 60) == (30, 60) and nutrition.target_rate(4 * 3600) == (60, 90)


def test_balance_tolerance_is_48_to_52():
    from src.analysis import balance
    assert "48–52 %" in used('carpes-2010') and balance.BALANCED_TOLERANCE_PCT == 2.0


def test_critical_power_window_is_three_to_twenty_minutes():
    assert "3 to 20 minutes" in used('jones-2010')
    curve = {120: 400.0, 180: 350.0, 600: 300.0, 1200: 280.0, 1800: 270.0}
    fit = power.estimate_critical_power(curve)
    assert fit and fit['r_squared'] > 0.9          # only the 180-1200 s points enter the fit


# --- the page and the navigation ---------------------------------------------------------------------------------

class Recorder:
    def __init__(self):
        self.shown = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.shown.append((name, args, kwargs))
            return self
        return call


def test_page_shows_every_reference(monkeypatch):
    from src.viz import references_page as page
    rec = Recorder()
    monkeypatch.setattr(page, "st", rec)
    page.render_references()
    bullets = [a[0] for n, a, k in rec.shown if n == "markdown" and a[0].startswith("- ")]
    assert len(bullets) == len(refs.REFERENCES)
    assert [a[0] for n, a, k in rec.shown if n == "subheader"] == refs.TOPICS
    assert sum(1 for n, a, k in rec.shown if n == "caption") == len(refs.REFERENCES) + 1


def test_app_has_a_references_page():
    src = open("app.py", encoding="utf-8").read()
    side = open("src/viz/sidebar.py", encoding="utf-8").read()
    assert 'app_mode == "References"' in src and 'mark("References")' in side
