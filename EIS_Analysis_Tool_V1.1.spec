# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[('C:\\ProgramData\\anaconda3\\Library\\bin\\ffi-8.dll', '.'), ('C:\\ProgramData\\anaconda3\\Library\\bin\\libcrypto-3-x64.dll', '.'), ('C:\\ProgramData\\anaconda3\\Library\\bin\\libssl-3-x64.dll', '.'), ('C:\\ProgramData\\anaconda3\\Library\\bin\\liblzma.dll', '.'), ('C:\\ProgramData\\anaconda3\\Library\\bin\\LIBBZ2.dll', '.'), ('C:\\ProgramData\\anaconda3\\Library\\bin\\libexpat.dll', '.'), ('C:\\ProgramData\\anaconda3\\Library\\bin\\sqlite3.dll', '.')],
    datas=[('piture', 'piture'), ('QRcode', 'QRcode'), ('resources', 'resources'), ('tests', 'tests'), ('docs', 'docs')],
    hiddenimports=['openpyxl', 'cvxopt'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='EIS_Analysis_Tool_V1.1',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='EIS_Analysis_Tool_V1.1',
)
