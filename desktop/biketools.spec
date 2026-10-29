# PyInstaller spec: pyinstaller desktop/biketools.spec  (run from the repository root)
from PyInstaller.utils.hooks import collect_all, copy_metadata

datas, binaries, hiddenimports = [('app.py', '.'), ('src', 'src')], [], []
for pkg in ('streamlit', 'streamlit_folium', 'folium', 'plotly', 'fitparse', 'gpxpy', 'geographiclib',
            'matplotlib', 'altair'):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h
hiddenimports += ['webview']
for dist in ('streamlit', 'pandas', 'numpy', 'plotly', 'requests', 'tornado', 'altair', 'pyarrow'):
    try:
        datas += copy_metadata(dist)
    except Exception:
        pass

a = Analysis(['desktop/launcher.py'], datas=datas, binaries=binaries, hiddenimports=hiddenimports,
             excludes=['pytest', 'tkinter'])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='BikeToolsAnalyzer', console=False)
coll = COLLECT(exe, a.binaries, a.datas, name='BikeToolsAnalyzer')
