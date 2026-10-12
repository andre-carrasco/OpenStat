# PyInstaller spec for OpenStat.  Build with:  pyinstaller packaging/openstat.spec --noconfirm
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent

datas = [(str(ROOT / "openstat" / "i18n" / "pt_BR.json"), "openstat/i18n"),
         (str(ROOT / "openstat" / "i18n" / "en.json"), "openstat/i18n")]
datas += collect_data_files("statsmodels") + collect_data_files("lifelines") + collect_data_files("pyDOE3")
datas += collect_data_files("docx")

# analyses are imported dynamically by the registry, and several libraries load submodules lazily
hiddenimports = collect_submodules("openstat") + collect_submodules("sklearn") + collect_submodules("statsmodels")
hiddenimports += collect_submodules("pyDOE3") + ["scipy.special._cdflib", "pyarrow", "openpyxl"]

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "PyQt5", "PyQt6", "IPython", "notebook", "pytest", "matplotlib.tests", "numpy.tests",
              "scipy.tests", "sklearn.tests", "statsmodels.tests", "pandas.tests"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="OpenStat", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="OpenStat")
