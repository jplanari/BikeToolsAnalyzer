"""The looks: CSS packs, the shipped Streamlit theme, and the desktop app carrying it along."""
import importlib.util
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
toml = pytest.importorskip("tomllib") if sys.version_info >= (3, 11) else pytest.importorskip("tomli")

from src.viz import theme


def _config():
    with open(os.path.join(ROOT, ".streamlit", "config.toml"), "rb") as f:
        return toml.load(f)


def test_every_look_has_css_except_classic():
    assert set(theme.LOOKS) == set(theme.CSS)
    assert theme.css_for("Classic") == "" and theme.css_for("nonsense") == ""
    for look in ("Dashboard", "Compact"):
        css = theme.css_for(look)
        assert "stMetric" in css and css.count("{") == css.count("}")


def test_css_adapts_to_light_and_dark_instead_of_fixing_colours():
    css = theme.css_for("Dashboard")
    assert "currentColor" in css
    assert not re.findall(r"background(?:-color)?:\s*#(?:fff|ffffff|000|000000)\b", css, re.I)


def test_accent_matches_the_streamlit_theme():
    cfg = _config()
    assert cfg["theme"]["primaryColor"].lower() == theme.ACCENT.lower()


def test_theme_has_light_and_dark_palettes_and_safe_chart_colours():
    cfg = _config()
    assert cfg["theme"]["light"]["backgroundColor"] != cfg["theme"]["dark"]["backgroundColor"]
    colours = cfg["theme"]["chartCategoricalColors"]
    assert len(colours) == len(set(colours)) == 8 and all(re.fullmatch(r"#[0-9A-Fa-f]{6}", c) for c in colours)


def test_desktop_app_installs_the_theme_next_to_the_data(tmp_path):
    spec = importlib.util.spec_from_file_location("launcher", os.path.join(ROOT, "desktop", "launcher.py"))
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    (tmp_path / ".streamlit").mkdir()
    (tmp_path / ".streamlit" / "secrets.toml").write_text("keep = 1")
    launcher.install_theme(str(tmp_path))
    assert (tmp_path / ".streamlit" / "config.toml").read_text() == open(
        os.path.join(ROOT, ".streamlit", "config.toml")).read()
    assert (tmp_path / ".streamlit" / "secrets.toml").read_text() == "keep = 1"


def test_pyinstaller_spec_ships_the_theme():
    assert "config.toml" in open(os.path.join(ROOT, "desktop", "biketools.spec")).read()
