"""The phone app's translations are generated from the desktop catalogs: they must be complete and current."""
from mobile import build_locales as bl


def test_every_phone_string_is_translated_with_the_same_placeholders():
    bad, missing = bl.problems()
    assert bad == []
    assert missing == {"es": [], "ca": []}


def test_generated_file_is_up_to_date():
    assert bl.OUT.read_text(encoding="utf-8") == bl.render(), "run: python -m mobile.build_locales"


def test_strings_are_found_in_the_code():
    keys = bl.wanted()
    assert "Energy and fuel" in keys and "✅ Fresh" in keys and "Season {year}" in keys
