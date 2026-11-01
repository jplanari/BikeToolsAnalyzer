"""
Translation of the app's own text.

    from src.i18n import t, tn
    st.caption(t("Rides are stored in UTC."))
    st.write(t("{name} has {n} rides", name=user, n=count))
    st.write(tn("{n} ride", "{n} rides", count))            # plural: picks the form for the language

The English text IS the key, so the code stays readable and a string with no translation shows in English; nothing
can break by a missing entry. Catalogs are JSON files in `locales/` (one per language, see `catalog.py`); English
needs none. Placeholders use `{name}`: they must be the same in every language, and the tests check that.

The language of the current run is set with `set_language` (the Streamlit glue does it at the start of every run,
see `session.py`) or, for code with no UI such as the weekly e-mail, `with use_language("es"): ...`.

Options of a radio or select box whose value the code compares are marked, not translated: `mark("Week")` in the list
and `format_func=tr` on the widget. The widget returns the English value; only the label changes.

Two rules for code that calls `t`:
- the argument must be a plain string (or a literal split over lines): the scanner (`python -m src.i18n.scan`)
  finds strings by reading the source, so a variable cannot be extracted;
- a function cached with `st.cache_data` that returns translated text must take the language as an argument
  (use `current_language()`), or the first language to run it is cached for everybody.
"""
import contextvars
from contextlib import contextmanager

from . import catalog, detect

DEFAULT = "en"
_language = contextvars.ContextVar("biketools_language", default=DEFAULT)

# languages whose plural rule differs from "1 is singular"
_ONE_FOR_0_AND_1 = {"fr", "pt"}


def available_languages(include_unreleased=False):
    """Codes of the languages the app can show, English first. Unreleased catalogs (still being translated) are left
    out unless asked for, or unless BIKETOOLS_I18N_PREVIEW is set (to try them)."""
    return [DEFAULT] + catalog.languages(include_unreleased or catalog.preview_enabled())


def language_name(code):
    """The language's own name ('Español'), for the language picker."""
    return catalog.name(code)


def current_language():
    return _language.get()


def set_language(code):
    """Makes `code` the language of the current run; an unknown code means English. Returns the code used."""
    code = code if code in available_languages(include_unreleased=True) else DEFAULT
    _language.set(code)
    return code


@contextmanager
def use_language(code):
    """`with use_language('es'):` translates inside the block only (weekly e-mail, reports built for another rider)."""
    token = _language.set(code if code in available_languages(include_unreleased=True) else DEFAULT)
    try:
        yield
    finally:
        _language.reset(token)


def _fill(text, fallback, kw):
    if not kw:
        return text
    try:
        return text.format(**kw)
    except (KeyError, IndexError, ValueError):
        try:
            return fallback.format(**kw)         # a broken translation never breaks the page
        except (KeyError, IndexError, ValueError):
            return fallback


def t(message, /, **values):
    """`message` in the current language; `{name}` placeholders are filled from `values`."""
    found = catalog.strings(_language.get()).get(message)
    text = found if isinstance(found, str) and found else message
    return _fill(text, message, values)


def mark(text):
    """Marks text that is translated later, where it is shown: `options = [mark("Week"), mark("Month")]` and then
    `st.radio(..., options, format_func=tr)`. It returns the text unchanged (the option keeps its English value, so
    comparisons in the code still work); the scanner reads it like t()."""
    return text


def tr(text):
    """Translates text that is only known when it is shown (an option marked with mark()). Not for new messages:
    use t() so the scanner can find them."""
    return t(text)


def plural_form(code, n):
    """'one' or 'other' (the two forms the catalogs use for now)."""
    if code in _ONE_FOR_0_AND_1:
        return "one" if n in (0, 1) else "other"
    return "one" if n == 1 else "other"


def tn(one, other, n, /, **values):
    """The singular or the plural of a message, as the current language needs it. `{n}` is filled in too.
    The catalog entry is keyed by the singular: {"one": "...", "other": "..."}."""
    values = dict(values, n=n)
    code = _language.get()
    found = catalog.strings(code).get(one)
    if isinstance(found, dict) and found.get(plural_form(code, n)):
        text = found[plural_form(code, n)]
    else:
        text = one if n == 1 else other
    return _fill(text, one if n == 1 else other, values)
