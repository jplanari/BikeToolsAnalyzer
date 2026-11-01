"""
Translation catalogs: one JSON file per language in `locales/` (`es.json`).

    {"_meta": {"name": "Español", "released": false},
     "strings": {"Current User": "Usuario actual",
                 "{n} ride": {"one": "{n} salida", "other": "{n} salidas"}}}

- The key is the English text exactly as written in the code. An empty value means "not translated yet" and falls
  back to English, so `python -m src.i18n.scan --sync` can add every new string at once.
- `released` decides whether the language shows in the picker. A catalog is released when a native speaker has
  reviewed it; until then it can be tried with the environment variable BIKETOOLS_I18N_PREVIEW=1.
"""
import json
import os
from pathlib import Path

LOCALES_DIR = Path(__file__).parent / "locales"
_cache = {}


def preview_enabled():
    return os.environ.get("BIKETOOLS_I18N_PREVIEW", "").strip().lower() in ("1", "true", "yes")


def _load(code):
    path = LOCALES_DIR / f"{code}.json"
    stamp = path.stat().st_mtime_ns if path.exists() else None
    hit = _cache.get(code)
    if hit and hit[0] == stamp:
        return hit[1]
    data = {"_meta": {}, "strings": {}}
    if stamp is not None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {"_meta": {}, "strings": {}}        # a damaged catalog must not stop the app: English shows
    _cache[code] = (stamp, data)
    return data


def strings(code):
    return _load(code).get("strings", {}) if code != "en" else {}


def meta(code):
    return _load(code).get("_meta", {})


def name(code):
    return "English" if code == "en" else meta(code).get("name") or code


def codes():
    """Every catalog file, whether released or not."""
    return sorted(p.stem for p in LOCALES_DIR.glob("*.json") if p.stem != "en")


def languages(include_unreleased=False):
    return [c for c in codes() if include_unreleased or meta(c).get("released")]


def save(code, data):
    """Writes a catalog with its keys sorted (stable diffs)."""
    data = {"_meta": data.get("_meta", {}), "strings": dict(sorted(data.get("strings", {}).items()))}
    (LOCALES_DIR / f"{code}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _cache.pop(code, None)


def load(code):
    return _load(code)
