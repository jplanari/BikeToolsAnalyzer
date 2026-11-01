# Translations

Stage 1: the mechanism. A few strings of the sidebar are wrapped as a pilot; everything else is still English.

## How it works
- `t("English text")` returns the text in the current language; `tn("{n} ride", "{n} rides", n)` picks singular or plural.
  The English text is the key, so a missing translation shows English and nothing can break.
- Catalogs are `locales/<code>.json`. An empty value means "not translated yet".
- The language of a run: the rider's saved language, else the browser's (Accept-Language), else the computer's, else English
  (`detect.py`, `session.py`). The picker (sidebar) saves the choice in the rider's profile.
- A language only shows in the picker once its catalog has `"released": true`. To try an unreleased one, start the app with
  `BIKETOOLS_I18N_PREVIEW=1`.

## Working with the text
```
python -m src.i18n.scan             # how much of the UI is wrapped, per file
python -m src.i18n.scan --list      # every widget string still to wrap, with file and line
python -m src.i18n.scan --sync      # add new strings to every catalog, drop the ones no code uses
python -m src.i18n.scan --status    # translated strings per language
python -m src.i18n.scan --check     # what the tests check: readable t() calls, no stale keys, same placeholders
```

## Rules for code
- Pass a plain string to `t`: `t("Weight (kg)")`. For values use placeholders, not an f-string:
  `t("{name} has {n} rides", name=user, n=count)`. A translation may reorder the placeholders.
- Do not name a variable or argument `t` or `tn` in a file that imports them (the scanner reports it).
- A function cached with `st.cache_data` that returns translated text must take `current_language()` as an argument.
- Strings used as identifiers (page names compared with `==`, dictionary keys) must not be translated; show a translated
  label with `format_func` and keep the English value.
- Keep sports and physics acronyms (FTP, TSS, NP, W′, CdA) as they are unless the reviewer says otherwise.

## Adding a language
1. Create `locales/<code>.json` with `{"_meta": {"name": "Deutsch", "released": false}, "strings": {}}`.
2. `python -m src.i18n.scan --sync` fills it with every string, empty.
3. Translate (a draft by Claude is fine), have a native speaker review it, then set `"released": true`.

## Option lists and dynamic text

Widget options whose English value is compared in code (radio/select lists, page and section names, period
labels) keep the English string as the value and are shown translated:

```python
sections = [mark("📊 Summary"), mark("⚡ Power")]       # mark(): extracted by the scanner, returns the text unchanged
st.radio(t("Section"), sections, format_func=tr)        # tr(): translates at display time
```

Spanish and Catalan are complete (every string translated) but still `"released": false` until a native
speaker has reviewed them; set `BIKETOOLS_I18N_PREVIEW=1` to try them. Not yet translated: chart titles and
axis labels, PDFs, the weekly e-mail body, and month/day names produced by `strftime`.
