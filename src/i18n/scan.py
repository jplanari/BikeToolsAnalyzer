"""
Finds the text of the app that is translated and the text that is not, and keeps the catalogs in step with the code.

    python -m src.i18n.scan                 summary: translated / still English literals, per file
    python -m src.i18n.scan --list          every string that still needs wrapping in t(), with file and line
    python -m src.i18n.scan --sync          add new strings to every catalog (empty = not translated yet),
                                            report the ones no code uses any more
    python -m src.i18n.scan --check         exit 1 on a t() that cannot be read from the source, a stale or
                                            inconsistent catalog entry (used by the tests)
    python -m src.i18n.scan --status        per language: how many strings are translated

It reads the source (it does not run the app). "Text of the app" is: the label, help, caption, placeholder or first
argument of a Streamlit widget or message call. Text built elsewhere (chart titles, the PDF writers, the e-mail,
sentences produced in src/analysis) is not found automatically yet; wrap it by hand and the scanner will pick it up.
"""
import argparse
import ast
import string
import sys
from pathlib import Path

from . import catalog

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ["app.py", "src"]
SKIP_PARTS = {"__pycache__"}

WIDGETS = {"markdown", "write", "caption", "subheader", "header", "title", "info", "warning", "error", "success",
           "button", "metric", "selectbox", "radio", "number_input", "text_input", "checkbox", "expander", "tabs",
           "spinner", "download_button", "toggle", "slider", "multiselect", "date_input", "file_uploader",
           "form_submit_button", "text_area", "popover", "toast", "segmented_control", "pills", "select_slider",
           "link_button", "page_link", "sidebar_header", "set_page_config"}
TEXT_KEYWORDS = {"help", "label", "placeholder", "caption", "title"}
TRANSLATE = {"t", "tn"}


def placeholders(text):
    """The {names} in a message."""
    try:
        return sorted({name for _, name, _, _ in string.Formatter().parse(text) if name})
    except ValueError:
        return None


def _is_alpha_text(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and any(c.isalpha() for c in node.value) \
        and len(node.value.strip()) > 2


def _call_name(node):
    return getattr(node.func, "attr", getattr(node.func, "id", ""))


def python_files():
    for entry in SOURCES:
        path = ROOT / entry
        files = [path] if path.is_file() else sorted(path.rglob("*.py"))
        for f in files:
            if not (SKIP_PARTS & set(f.relative_to(ROOT).parts)):
                yield f


def scan_file(path):
    """(translated, untranslated, problems): translated = [(line, singular, plural or None)];
    untranslated = [(line, kind, text)]; problems = [(line, text)]."""
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except SyntaxError:
        return [], [], []
    translated, untranslated, problems = [], [], []
    wrapped_nodes = set()
    imports_t = any(isinstance(n, ast.ImportFrom) and any(a.name in TRANSLATE for a in n.names)
                    and ((n.module or "").startswith("src.i18n") or (n.level >= 1 and not n.module)) for n in ast.walk(tree))
    if imports_t:                                  # t / tn must mean the translation function in this file
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in TRANSLATE:
                problems.append((n.lineno, f"'{n.name}' is redefined here and hides the translation function"))
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id in TRANSLATE:
                problems.append((n.lineno, f"'{n.id}' is assigned here and hides the translation function"))
            elif isinstance(n, ast.arg) and n.arg in TRANSLATE:
                problems.append((n.lineno, f"'{n.arg}' is an argument here and hides the translation function"))
    for node in ast.walk(tree):
        if imports_t and isinstance(node, ast.Call) and _call_name(node) in TRANSLATE and isinstance(node.func, ast.Name):
            args = node.args
            need = 2 if _call_name(node) == "tn" else 1
            if len(args) < need or not all(isinstance(a, ast.Constant) and isinstance(a.value, str) for a in args[:need]):
                problems.append((node.lineno, "t()/tn() needs plain string arguments to be found by the scanner"))
                continue
            translated.append((node.lineno, args[0].value, args[1].value if need == 2 else None))
            for a in args[:need]:
                wrapped_nodes.add(id(a))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) not in WIDGETS or _call_name(node) in TRANSLATE:
            continue
        candidates = list(node.args[:1]) + [k.value for k in node.keywords if k.arg in TEXT_KEYWORDS]
        for arg in candidates:
            for sub in ast.walk(arg):
                if isinstance(sub, ast.Call) and _call_name(sub) in TRANSLATE:
                    break
            else:
                if _is_alpha_text(arg):
                    untranslated.append((arg.lineno, "text", arg.value))
                elif isinstance(arg, ast.JoinedStr):
                    untranslated.append((arg.lineno, "f-string", ast.unparse(arg)))
    return sorted(translated), sorted(untranslated), sorted(problems)


def scan_all():
    out = {}
    for f in python_files():
        out[str(f.relative_to(ROOT))] = scan_file(f)
    return out


def used_messages(results):
    """{singular: plural or None} of every translated message in the code."""
    msgs = {}
    for translated, _, _ in results.values():
        for _, one, other in translated:
            msgs[one] = other
    return msgs


def check_catalog(code, messages):
    """Problems of one catalog: stale keys, placeholders that differ, a plural entry that is not {one, other}."""
    out = []
    strings = catalog.strings(code)
    for key, value in strings.items():
        if key not in messages:
            out.append(f"{code}: '{key}' is not used by any t() call")
            continue
        forms = {"": value} if isinstance(value, str) else (value if isinstance(value, dict) else None)
        if forms is None:
            out.append(f"{code}: '{key}' must be text or {{one, other}}")
            continue
        if isinstance(value, dict) and set(value) - {"one", "other"}:
            out.append(f"{code}: '{key}' has unknown plural forms {sorted(set(value))}")
        want = set(placeholders(key) or []) | set(placeholders(messages[key]) if messages[key] else [])
        for form, text in forms.items():
            if text and set(placeholders(text) or []) != want:
                out.append(f"{code}: '{key}' {form}: placeholders {sorted(placeholders(text) or [])} != {sorted(want)}")
    return out


def sync():
    results = scan_all()
    messages = used_messages(results)
    lines = []
    for code in catalog.codes():
        data = catalog.load(code)
        strings = dict(data.get("strings", {}))
        added = [m for m in messages if m not in strings]
        for m in added:
            strings[m] = {"one": "", "other": ""} if messages[m] else ""
        stale = [k for k in strings if k not in messages]
        for k in stale:
            del strings[k]
        catalog.save(code, {"_meta": data.get("_meta", {}), "strings": strings})
        lines.append(f"{code}: {len(added)} added, {len(stale)} removed, {len(strings)} in the catalog")
    return lines


def status():
    messages = used_messages(scan_all())
    lines = []
    for code in catalog.codes():
        strings = catalog.strings(code)

        def done(v):
            return bool(v) if isinstance(v, str) else bool(v and v.get("one") and v.get("other"))
        n = sum(1 for m in messages if done(strings.get(m)))
        flag = "released" if catalog.meta(code).get("released") else "not released"
        lines.append(f"{code} ({catalog.name(code)}): {n} of {len(messages)} translated, {flag}")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args(argv)
    results = scan_all()
    if a.sync:
        print("\n".join(sync()))
        return 0
    if a.status:
        print("\n".join(status()))
        return 0
    if a.check:
        messages = used_messages(results)
        bad = [f"{f}:{ln}: {why}" for f, (_, _, probs) in results.items() for ln, why in probs]
        for code in catalog.codes():
            bad += check_catalog(code, messages)
        print("\n".join(bad) if bad else "ok")
        return 1 if bad else 0
    if a.list:
        for f, (_, untranslated, _) in results.items():
            for ln, kind, text in untranslated:
                print(f"{f}:{ln}: [{kind}] {text[:110]!r}")
        return 0
    total_t = total_u = 0
    print(f"{'file':<38}{'translated':>11}{'English':>9}")
    for f, (translated, untranslated, _) in sorted(results.items(), key=lambda kv: -len(kv[1][1])):
        if translated or untranslated:
            print(f"{f:<38}{len(translated):>11}{len(untranslated):>9}")
        total_t += len(translated)
        total_u += len(untranslated)
    print(f"{'total':<38}{total_t:>11}{total_u:>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
