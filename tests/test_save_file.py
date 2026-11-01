import sys
from types import SimpleNamespace

import pytest

from src.viz import save_file as sf


class FakeSt(SimpleNamespace):
    pass


def make_st(pressed):
    calls = []
    st = FakeSt(session_state={}, calls=calls)
    st.download_button = lambda *a, **k: calls.append(("download", k.get("file_name")))
    st.button = lambda *a, **k: pressed
    st.success = lambda m: calls.append(("success", m))
    st.error = lambda m: calls.append(("error", m))
    return st


def test_browser_keeps_download_button(monkeypatch):
    monkeypatch.delenv(sf.DESKTOP_ENV, raising=False)
    st = make_st(False)
    monkeypatch.setattr(sf, "st", st)
    assert sf.file_button(b"x", "a.pdf", "k") is None
    assert st.calls == [("download", "a.pdf")]


def test_desktop_saves_where_chosen(monkeypatch, tmp_path):
    monkeypatch.setenv(sf.DESKTOP_ENV, "1")
    monkeypatch.setattr(sf, "dialog_available", lambda: True)
    st = make_st(True)
    monkeypatch.setattr(sf, "st", st)
    target = tmp_path / "out.pdf"
    assert sf.file_button(b"%PDF", "a.pdf", "k", choose=lambda name: str(target)) == str(target)
    assert target.read_bytes() == b"%PDF"
    assert st.calls[-1][0] == "success"


def test_desktop_cancelled_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv(sf.DESKTOP_ENV, "1")
    monkeypatch.setattr(sf, "dialog_available", lambda: True)
    st = make_st(True)
    monkeypatch.setattr(sf, "st", st)
    assert sf.file_button(b"x", "a.pdf", "k", choose=lambda name: None) is None
    assert st.calls == []


def test_desktop_write_error_is_reported(monkeypatch, tmp_path):
    monkeypatch.setenv(sf.DESKTOP_ENV, "1")
    monkeypatch.setattr(sf, "dialog_available", lambda: True)
    st = make_st(True)
    monkeypatch.setattr(sf, "st", st)
    sf.file_button(b"x", "a.pdf", "k", choose=lambda name: str(tmp_path / "missing" / "a.pdf"))
    assert st.calls[-1][0] == "error"


def test_desktop_without_dialog_falls_back(monkeypatch):
    monkeypatch.setenv(sf.DESKTOP_ENV, "1")
    monkeypatch.setattr(sf, "dialog_available", lambda: False)
    st = make_st(False)
    monkeypatch.setattr(sf, "st", st)
    sf.file_button(b"x", "a.pdf", "k")
    assert st.calls == [("download", "a.pdf")]


def test_windows_filter_is_double_null_terminated():
    f = sf.windows_filter("pdf")
    assert f.startswith("PDF (*.pdf)\0*.pdf\0") and f.endswith("\0\0")


def test_applescript_escapes_quotes():
    cmd = sf.applescript_command('a"b.pdf', "/tmp", "Save")
    assert cmd[0] == "osascript" and 'a\\"b.pdf' in cmd[2]


def test_linux_command_prefers_zenity(monkeypatch):
    monkeypatch.setattr(sf.shutil, "which", lambda n: "/usr/bin/" + n if n in ("zenity", "kdialog") else None)
    assert sf.linux_command("a.pdf", "/tmp", "Save")[0] == "zenity"
    monkeypatch.setattr(sf.shutil, "which", lambda n: None)
    assert sf.linux_command("a.pdf", "/tmp", "Save") is None


def test_choose_save_path_swallows_errors(monkeypatch):
    monkeypatch.setattr(sf.shutil, "which", lambda n: None)
    monkeypatch.setattr(sys, "platform", "linux")
    assert sf.choose_save_path("a.pdf") is None
