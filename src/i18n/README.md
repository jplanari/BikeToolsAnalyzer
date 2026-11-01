# Translations

The whole interface is wrapped for translation: pages, chart titles and labels, tables, the PDF reports, the weekly
e-mail, and the messages of the data layer and the providers.

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

Spanish and Catalan are complete (every string translated), reviewed by the author and released. New languages
stay `"released": false` until reviewed; set `BIKETOOLS_I18N_PREVIEW=1` to try them. Chart titles, axis titles, trace names and
annotations are covered (the scanner checks Plotly calls). So are the PDF reports and the weekly e-mail (written
in the rider's saved language, also when it is sent in the background).

Month and day names come from the catalog too: use `fdate(value, "%a %d %b")` instead of `strftime`, or pass the
date to `t()` as a placeholder (`t("Week of {monday:%d %b}", monday=monday)`); both translate `%a %A %b %B`.

Code must never compare translated text. When a message drives a decision, give it a type or a helper that works in
every language (`is_duplicate_message`, `PolarNoData`, `StravaNoData`, `WahooNoData`), as the importers do.

Still English: developer-facing log lines and warnings, file and column names, and unit symbols.
