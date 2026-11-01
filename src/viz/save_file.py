"""Saving a generated file (the PDF reports).

In a browser, st.download_button is right: the browser decides where the file goes. In the desktop app the page
lives in a program's own window, where "download" is out of place (no download bar, files land in a folder the rider
did not choose), so there the button opens the system's *Save as* dialog and writes the file where the rider says.
The dialog is opened by the server process (the one running the app), in a native way per system: the Win32 common
dialog, AppleScript on macOS, zenity / kdialog on Linux. Where none is available the plain download button is kept.
"""
import ctypes
import os
import shutil
import subprocess
import sys
from ctypes import wintypes

import streamlit as st

from src.i18n import t

DESKTOP_ENV = "BIKETOOLS_DESKTOP"          # set by the desktop launcher in the server process


def is_desktop():
    return bool(os.environ.get(DESKTOP_ENV))


def start_folder():
    """Where the dialog opens: the Downloads folder if there is one, else the home folder."""
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    return downloads if os.path.isdir(downloads) else os.path.expanduser("~")


# --- Windows: the common Save dialog ------------------------------------------------------------------------------

class OPENFILENAMEW(ctypes.Structure):
    _fields_ = [("lStructSize", wintypes.DWORD), ("hwndOwner", wintypes.HWND), ("hInstance", wintypes.HINSTANCE),
                ("lpstrFilter", wintypes.LPCWSTR), ("lpstrCustomFilter", wintypes.LPWSTR),
                ("nMaxCustFilter", wintypes.DWORD), ("nFilterIndex", wintypes.DWORD),
                ("lpstrFile", wintypes.LPWSTR), ("nMaxFile", wintypes.DWORD),
                ("lpstrFileTitle", wintypes.LPWSTR), ("nMaxFileTitle", wintypes.DWORD),
                ("lpstrInitialDir", wintypes.LPCWSTR), ("lpstrTitle", wintypes.LPCWSTR),
                ("Flags", wintypes.DWORD), ("nFileOffset", wintypes.WORD), ("nFileExtension", wintypes.WORD),
                ("lpstrDefExt", wintypes.LPCWSTR), ("lCustData", wintypes.LPARAM), ("lpfnHook", ctypes.c_void_p),
                ("lpTemplateName", wintypes.LPCWSTR), ("pvReserved", ctypes.c_void_p),
                ("dwReserved", wintypes.DWORD), ("FlagsEx", wintypes.DWORD)]


OFN_OVERWRITEPROMPT, OFN_PATHMUSTEXIST, OFN_NOCHANGEDIR, OFN_EXPLORER = 0x2, 0x800, 0x8, 0x80000


def windows_filter(extension):
    """The file-type list the Win32 dialog wants: 'PDF (*.pdf)\\0*.pdf\\0\\0'."""
    return f"{extension.upper()} (*.{extension})\0*.{extension}\0All files (*.*)\0*.*\0\0"


def _save_dialog_windows(default_name, folder, title):
    buffer = ctypes.create_unicode_buffer(default_name, 1024)
    extension = os.path.splitext(default_name)[1].lstrip(".") or "pdf"
    ofn = OPENFILENAMEW()
    ofn.lStructSize = ctypes.sizeof(OPENFILENAMEW)
    ofn.hwndOwner = ctypes.windll.user32.GetForegroundWindow()
    ofn.lpstrFilter = windows_filter(extension)
    ofn.lpstrFile = ctypes.cast(buffer, wintypes.LPWSTR)
    ofn.nMaxFile = len(buffer)
    ofn.lpstrInitialDir = folder
    ofn.lpstrTitle = title
    ofn.lpstrDefExt = extension
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_EXPLORER
    return buffer.value if ctypes.windll.comdlg32.GetSaveFileNameW(ctypes.byref(ofn)) else None


# --- macOS and Linux ----------------------------------------------------------------------------------------------

def applescript_command(default_name, folder, title):
    escape = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')
    script = (f'POSIX path of (choose file name with prompt "{escape(title)}" default name "{escape(default_name)}" '
              f'default location (POSIX file "{escape(folder)}"))')
    return ["osascript", "-e", script]


def linux_command(default_name, folder, title):
    """zenity or kdialog, whichever exists; None when neither does."""
    start = os.path.join(folder, default_name)
    if shutil.which("zenity"):
        return ["zenity", "--file-selection", "--save", "--confirm-overwrite", f"--filename={start}",
                f"--title={title}"]
    if shutil.which("kdialog"):
        return ["kdialog", "--getsavefilename", start]
    return None


def _run_dialog(command):
    try:
        done = subprocess.run(command, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    except OSError:
        return None
    path = done.stdout.strip()
    return path if done.returncode == 0 and path else None


def dialog_available():
    return sys.platform in ("win32", "darwin") or bool(shutil.which("zenity") or shutil.which("kdialog"))


def choose_save_path(default_name, title=None, folder=None):
    """The path the rider picked in the system's Save dialog, or None (cancelled, or no dialog on this system)."""
    title = title or t("Save as")
    folder = folder or start_folder()
    try:
        if sys.platform == "win32":
            return _save_dialog_windows(default_name, folder, title)
        if sys.platform == "darwin":
            return _run_dialog(applescript_command(default_name, folder, title))
        command = linux_command(default_name, folder, title)
        return _run_dialog(command) if command else None
    except Exception:
        return None


# --- the button -----------------------------------------------------------------------------------------------------

def file_button(data, file_name, key, mime="application/pdf", choose=choose_save_path):
    """A download button in a browser; a 'Save as…' button that writes the file where the rider chooses in the
    desktop app. Returns the saved path (this run or an earlier one for the same file) or None."""
    if not (is_desktop() and dialog_available()):
        st.download_button(t("⬇️ Download PDF"), data, mime=mime, file_name=file_name, key=key)
        return None
    state = f"_saved_{key}"
    if st.button(t("💾 Save as…"), key=key):
        path = choose(file_name)
        if path:
            try:
                with open(path, "wb") as fh:
                    fh.write(data)
                st.session_state[state] = (file_name, len(data), path)
            except OSError as e:
                st.error(t("Could not save the file: {e}", e=e))
    saved = st.session_state.get(state)
    if saved and saved[:2] == (file_name, len(data)):
        st.success(t("Saved to {path}", path=saved[2]))
        return saved[2]
    return None
