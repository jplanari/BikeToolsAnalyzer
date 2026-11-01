"""
Which language the person uses, without asking: the browser's Accept-Language header first, then this computer's
own language. No Streamlit here, so it also works for the desktop launcher and for tests.

Order used by the app (see session.py): the language saved in the rider's profile, then the browser, then the
computer, then English.
"""
import ctypes
import locale
import os
import re
import subprocess
import sys


def parse_accept_language(header):
    """Language codes of an Accept-Language header, most preferred first and without the region:
    'es-ES,es;q=0.9,en;q=0.8' -> ['es', 'en']."""
    scored = []
    for n, part in enumerate((header or "").split(",")):
        bits = part.strip().split(";")
        tag = bits[0].strip().lower()
        if not tag or tag == "*":
            continue
        q = 1.0
        for extra in bits[1:]:
            m = re.match(r"\s*q\s*=\s*([0-9.]+)", extra)
            if m:
                try:
                    q = float(m.group(1))
                except ValueError:
                    q = 0.0
        if q > 0:
            scored.append((-q, n, re.split(r"[-_]", tag)[0]))
    out = []
    for _, _, code in sorted(scored):
        if code not in out:
            out.append(code)
    return out


def _posix_codes(environ):
    out = []
    for var in ("LC_ALL", "LC_MESSAGES", "LANGUAGE", "LANG"):
        for value in (environ.get(var) or "").split(":"):
            value = value.strip()
            if value and value not in ("C", "POSIX") and not value.startswith(("C.", "POSIX.")):
                out.append(re.split(r"[-_.@]", value.lower())[0])
    return out


def _windows_code():
    try:
        lcid = ctypes.windll.kernel32.GetUserDefaultUILanguage()          # type: ignore[attr-defined]
        return locale.windows_locale.get(lcid, "").split("_")[0].lower() or None
    except Exception:
        return None


def _mac_codes():
    try:
        out = subprocess.run(["defaults", "read", "-g", "AppleLanguages"], capture_output=True, text=True,
                             timeout=2).stdout
        return [re.split(r"[-_]", m.lower())[0] for m in re.findall(r'"?([A-Za-z]{2,3}(?:[-_][A-Za-z0-9]+)*)"?,?', out)]
    except Exception:
        return []


def os_languages(environ=None):
    """The computer's languages, most preferred first (environment variables, then Windows or macOS settings)."""
    environ = os.environ if environ is None else environ
    codes = _posix_codes(environ)
    if sys.platform.startswith("win"):
        codes.append(_windows_code())
    elif sys.platform == "darwin" and environ is os.environ:
        codes += _mac_codes()
    try:
        codes.append((locale.getlocale()[0] or "").split("_")[0].lower())
    except Exception:
        pass
    out = []
    for c in codes:
        if c and c not in out:
            out.append(c)
    return out


def first_supported(candidates, supported):
    """The first candidate the app has a language for, or None."""
    return next((c for c in candidates if c in supported), None)


def detect_language(supported, accept_language=None, environ=None):
    """(code, source) with source 'browser', 'system' or 'default'. English when nothing matches."""
    found = first_supported(parse_accept_language(accept_language), supported)
    if found:
        return found, "browser"
    found = first_supported(os_languages(environ), supported)
    if found:
        return found, "system"
    return "en", "default"
