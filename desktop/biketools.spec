# PyInstaller spec: pyinstaller desktop/biketools.spec  (run from the repository root)
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

# PyInstaller resolves relative paths against the spec's folder (SPECPATH), so anchor everything to the repo root
ROOT = os.path.abspath(os.path.join(SPECPATH, '..'))

datas, binaries, hiddenimports = [(os.path.join(ROOT, 'app.py'), '.'), (os.path.join(ROOT, 'src'), 'src'),
                                    (os.path.join(ROOT, '.streamlit', 'config.toml'), '.streamlit')], [], []
for pkg in ('streamlit', 'streamlit_folium', 'folium', 'plotly', 'fitparse', 'gpxpy', 'geographiclib',
            'matplotlib', 'altair'):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h
# app.py and src/ travel as data files, so PyInstaller does not see their imports: list them here
for pkg in ('scipy', 'pandas', 'numpy', 'requests', 'sqlite3', 'PIL'):
    hiddenimports += collect_submodules(pkg)
hiddenimports += ['webview']
for dist in ('streamlit', 'pandas', 'numpy', 'plotly', 'requests', 'tornado', 'altair', 'pyarrow'):
    try:
        datas += copy_metadata(dist)
    except Exception:
        pass

a = Analysis([os.path.join(ROOT, 'desktop', 'launcher.py')], pathex=[ROOT], datas=datas, binaries=binaries, hiddenimports=hiddenimports,
             excludes=['pytest', 'tkinter'])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='BikeToolsAnalyzer', console=False)
coll = COLLECT(exe, a.binaries, a.datas, name='BikeToolsAnalyzer')
