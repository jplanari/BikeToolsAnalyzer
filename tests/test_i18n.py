"""Translation foundation: lookup and fallbacks, language detection, the saved language, the scanner and the catalogs."""
import json
import threading
import types

import pytest

from src import i18n
from src.i18n import catalog, detect, scan


@pytest.fixture
def locales(tmp_path, monkeypatch):
    """An isolated catalog folder with a Spanish and a French catalog (French: 0 and 1 are singular)."""
    monkeypatch.setattr(catalog, "LOCALES_DIR", tmp_path)
    monkeypatch.setattr(catalog, "_cache", {})
    monkeypatch.delenv("BIKETOOLS_I18N_PREVIEW", raising=False)

    def write(code, strings, released=True, name=None):
        (tmp_path / f"{code}.json").write_text(json.dumps(
            {"_meta": {"name": name or code, "released": released}, "strings": strings}, ensure_ascii=False))
        catalog._cache.pop(code, None)
    write("es", {"Hello {name}": "Hola {name}", "{n} ride": {"one": "{n} salida", "other": "{n} salidas"},
                 "Empty": "", "Broken {x}": "Roto {y}"}, name="Español")
    write("fr", {"{n} ride": {"one": "{n} sortie", "other": "{n} sorties"}}, name="Français")
    i18n.set_language("en")
    yield write
    i18n.set_language("en")


# --- lookup ---------------------------------------------------------------------------------------------------------

def test_english_and_unknown_text_pass_through_and_placeholders_fill(locales):
    assert i18n.t("Hello {name}", name="Pep") == "Hello Pep"
    assert i18n.t("not in any catalog") == "not in any catalog"
    assert i18n.t("100 % {not a placeholder}") == "100 % {not a placeholder}"      # no values: no formatting


def test_translation_is_used_and_gaps_fall_back_to_english(locales):
    i18n.set_language("es")
    assert i18n.t("Hello {name}", name="Pep") == "Hola Pep"
    assert i18n.t("Empty") == "Empty"                      # an empty entry means not translated yet
    assert i18n.t("Missing") == "Missing"


def test_a_broken_translation_never_breaks_the_page(locales):
    i18n.set_language("es")
    assert i18n.t("Broken {x}", x=3) == "Broken 3"         # the translation uses a name that does not exist


def test_plurals_follow_the_language(locales):
    assert [i18n.tn("{n} ride", "{n} rides", n) for n in (0, 1, 2)] == ["0 rides", "1 ride", "2 rides"]
    i18n.set_language("es")
    assert [i18n.tn("{n} ride", "{n} rides", n) for n in (0, 1, 2)] == ["0 salidas", "1 salida", "2 salidas"]
    i18n.set_language("fr")
    assert [i18n.tn("{n} ride", "{n} rides", n) for n in (0, 1, 2)] == ["0 sortie", "1 sortie", "2 sorties"]


def test_unknown_language_means_english_and_the_context_manager_restores(locales):
    assert i18n.set_language("zz") == "en"
    i18n.set_language("es")
    with i18n.use_language("fr"):
        assert i18n.current_language() == "fr"
        with i18n.use_language("nonsense"):
            assert i18n.current_language() == "en"
    assert i18n.current_language() == "es"


def test_each_thread_has_its_own_language(locales):
    i18n.set_language("es")
    seen = {}

    def other():
        seen["start"] = i18n.current_language()
        i18n.set_language("fr")
        seen["end"] = i18n.current_language()
    th = threading.Thread(target=other)
    th.start()
    th.join()
    assert seen == {"start": "en", "end": "fr"} and i18n.current_language() == "es"


def test_unreleased_languages_stay_hidden_until_previewed(locales, monkeypatch):
    locales("ca", {"Hello {name}": "Hola {name}"}, released=False, name="Català")
    assert i18n.available_languages() == ["en", "es", "fr"]
    assert i18n.available_languages(include_unreleased=True) == ["en", "ca", "es", "fr"]
    monkeypatch.setenv("BIKETOOLS_I18N_PREVIEW", "1")
    assert "ca" in i18n.available_languages()
    assert i18n.language_name("ca") == "Català" and i18n.language_name("en") == "English"


def test_a_damaged_catalog_is_ignored(locales, tmp_path):
    (tmp_path / "es.json").write_text("{not json")
    catalog._cache.clear()
    i18n.set_language("es")                                # unreleased now (no meta): still selectable by code
    assert i18n.t("Hello {name}", name="Pep") == "Hello Pep"


# --- detection ------------------------------------------------------------------------------------------------------

def test_accept_language_is_read_in_order_of_preference():
    assert detect.parse_accept_language("es-ES,es;q=0.9,en;q=0.8") == ["es", "en"]
    assert detect.parse_accept_language("en;q=0.5, ca-ES;q=0.9, es;q=0.7") == ["ca", "es", "en"]
    assert detect.parse_accept_language("fr;q=0, de") == ["de"]
    assert detect.parse_accept_language("*") == [] and detect.parse_accept_language(None) == []


def test_detection_order_browser_then_system_then_english():
    supported = ["en", "es", "ca"]
    assert detect.detect_language(supported, "ca-ES,ca;q=0.9", {"LANG": "es_ES.UTF-8"}) == ("ca", "browser")
    assert detect.detect_language(supported, "fr-FR,fr;q=0.9", {"LANG": "es_ES.UTF-8"}) == ("es", "system")
    assert detect.detect_language(supported, None, {"LANGUAGE": "ca:es", "LANG": "en_US.UTF-8"}) == ("ca", "system")
    assert detect.detect_language(supported, "de", {"LANG": "C"})[0] in ("en", "es", "ca")   # C/POSIX are not a language


def test_posix_codes_skip_c_and_strip_encoding():
    assert detect._posix_codes({"LANG": "C.UTF-8"}) == [] and detect._posix_codes({"LANG": "POSIX"}) == []
    assert detect._posix_codes({"LC_ALL": "pt_BR.UTF-8", "LANG": "es_ES@euro"}) == ["pt", "es"]


# --- saved language and the session --------------------------------------------------------------------------------

def test_the_riders_language_is_saved_and_unknown_ones_are_dropped(tmp_path, monkeypatch, locales):
    from src.data import db
    monkeypatch.setattr(db, "DB_FILE", str(tmp_path / "t.db"))
    db.init_db()
    db.save_user("Ana")
    assert db.get_user_data("Ana")['language'] is None
    db.set_user_language("Ana", "es")
    assert db.get_user_data("Ana")['language'] == "es"
    db.set_user_language("Ana", "klingon")
    assert db.get_user_data("Ana")['language'] is None


def test_an_old_database_gains_the_language_column(tmp_path, monkeypatch):
    import sqlite3
    from src.data import db
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT UNIQUE, ftp INTEGER, lthr INTEGER, weight REAL, height REAL)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(db, "DB_FILE", str(path))
    db.init_db()
    conn = sqlite3.connect(path)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(users)")]
    conn.close()
    assert "language" in cols


class _State(dict):
    __getattr__ = dict.get


def _fake_streamlit(monkeypatch, header=None, shown=None):
    from src.i18n import session
    headers = {"Accept-Language": header} if header else {}
    fake = types.SimpleNamespace(session_state=_State(), context=types.SimpleNamespace(headers=headers),
                                 sidebar=types.SimpleNamespace(selectbox=lambda *a, **k: shown.append((a, k)) if shown is not None else None))
    monkeypatch.setattr(session, "st", fake)
    return session, fake


def test_session_starts_in_the_browsers_language_and_a_saved_one_wins(locales, monkeypatch):
    session, st = _fake_streamlit(monkeypatch, header="es-ES,es;q=0.9,en;q=0.8")
    assert session.init_session() == "es" and st.session_state["lang_source"] == "browser"
    assert session.apply_user_language("Ana", "fr") is True            # saved language differs: run again
    assert session.init_session() == "fr" and st.session_state["lang_source"] == "profile"
    assert session.apply_user_language("Ana", "es") is False           # once per rider: a later choice is kept
    assert session.apply_user_language("Ana", None) is False


def test_session_without_a_header_or_a_match_is_english(locales, monkeypatch):
    monkeypatch.setattr(detect, "os_languages", lambda environ=None: ["ja"])
    session, st = _fake_streamlit(monkeypatch, header="de-DE")
    assert session.init_session() == "en" and st.session_state["lang_source"] == "default"


def test_the_picker_is_hidden_while_english_is_the_only_language(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "LOCALES_DIR", tmp_path)
    monkeypatch.setattr(catalog, "_cache", {})
    shown = []
    session, _ = _fake_streamlit(monkeypatch, shown=shown)
    session.picker("Ana")
    assert shown == []


def test_the_picker_offers_the_released_languages(locales, monkeypatch):
    shown = []
    session, st = _fake_streamlit(monkeypatch, shown=shown)
    session.init_session()
    session.picker(None)
    assert shown and shown[0][0][1] == ["en", "es", "fr"]


# --- scanner and catalogs --------------------------------------------------------------------------------------------

def _scan_source(tmp_path, code):
    f = tmp_path / "page.py"
    f.write_text(code)
    return scan.scan_file(f)


def test_scanner_separates_translated_from_english_text(tmp_path):
    translated, untranslated, problems = _scan_source(tmp_path, '''
import streamlit as st
from src.i18n import t, tn
st.header(t("Profile"))
st.caption("Rides are stored in UTC.")
st.metric("Weight", 70, help=t("In kg"))
st.write(f"Hello {name}")
st.write(tn("{n} ride", "{n} rides", 3))
st.button("OK")
''')
    assert [(m, p) for _, m, p in translated] == [("Profile", None), ("In kg", None), ("{n} ride", "{n} rides")]
    kinds = sorted((k, x[:14]) for _, k, x in untranslated)
    assert ("text", "Rides are stor") in kinds and ("text", "Weight") in kinds and any(k == "f-string" for k, _ in kinds)
    assert not problems and all("OK" not in x for _, _, x in untranslated)       # "OK" is too short to be text


def test_scanner_reports_text_it_cannot_read_and_a_hidden_t(tmp_path):
    _, _, problems = _scan_source(tmp_path, '''
from src.i18n import t
label = "x"
st.write(t(label))
def t(**kw): pass
''')
    assert len(problems) == 2


def test_a_t_in_a_file_that_does_not_import_it_is_not_a_translation(tmp_path):
    translated, _, problems = _scan_source(tmp_path, 'x = t("a local function")\n')
    assert translated == [] and problems == []


def test_sync_adds_new_strings_and_drops_stale_ones(locales, monkeypatch, tmp_path):
    src = tmp_path / "src_tree"
    src.mkdir()
    (src / "page.py").write_text('from src.i18n import t, tn\nt("New text")\ntn("{n} ride", "{n} rides", 2)\n')
    monkeypatch.setattr(scan, "ROOT", tmp_path)
    monkeypatch.setattr(scan, "SOURCES", ["src_tree"])
    lines = scan.sync()
    assert any(l.startswith("es: 1 added, 3 removed") for l in lines)
    es = catalog.load("es")["strings"]
    assert es["New text"] == "" and es["{n} ride"]["one"] == "{n} salida" and "Hello {name}" not in es
    assert scan.check_catalog("es", scan.used_messages(scan.scan_all())) == []


def test_catalog_check_finds_bad_placeholders_and_stale_keys(locales):
    locales("es", {"Hello {name}": "Hola {nombre}", "Old": "Viejo", "{n} ride": {"one": "{n}", "few": "x"}})
    problems = scan.check_catalog("es", {"Hello {name}": None, "{n} ride": "{n} rides"})
    text = "\n".join(problems)
    assert "placeholders" in text and "'Old' is not used" in text and "unknown plural forms" in text


def test_the_repositorys_own_translations_are_consistent():
    results = scan.scan_all()
    messages = scan.used_messages(results)
    bad = [f"{f}:{ln}: {why}" for f, (_, _, probs) in results.items() for ln, why in probs]
    for code in catalog.codes():
        bad += scan.check_catalog(code, messages)
        assert catalog.meta(code).get("name"), code
    assert not bad, bad
