"""Static check that no module uses a global name that is never defined or imported
(Streamlit pages only fail at run time, and only in the section that uses the name)."""
import ast
import builtins
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"


def _defined(tree):
    names = set(dir(builtins)) | {"__file__", "__name__"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if not isinstance(node, ast.ClassDef):
                a = node.args
                names.update(x.arg for x in a.posonlyargs + a.args + a.kwonlyargs)
                names.update(x.arg for x in (a.vararg, a.kwarg) if x)
        elif isinstance(node, ast.Lambda):
            a = node.args
            names.update(x.arg for x in a.posonlyargs + a.args + a.kwonlyargs)
            names.update(x.arg for x in (a.vararg, a.kwarg) if x)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((x.asname or x.name).split(".")[0] for x in node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: str(p.relative_to(SRC)))
def test_no_undefined_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defined = _defined(tree)
    used = {n.id: n.lineno for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    missing = {k: v for k, v in used.items() if k not in defined}
    assert not missing, f"undefined names (name: line): {missing}"
