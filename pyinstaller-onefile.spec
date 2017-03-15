# -*- mode: python -*-

#__author__ = "Alberto Pettarin"
#__email__ = "aeneas@readbeyond.it"
#__copyright__ = """
#    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
#    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
#    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
#"""
#__license__ = "GNU AGPL 3"
#__status__ = "Production"
#__version__ = "1.7.3"

datas = [
    # required
    ("aeneas/res/*",            "aeneas/res"),
    ("aeneas/tools/res/*",      "aeneas/tools/res"),
    # optional, copy files
    ("aeneas/extra/*.py",       "aeneas/extra"),
    # optional, create output directory
    ("aeneas/extra/.gitignore", "output"),
]

block_cipher = None

a = Analysis(
    ['pyinstaller-aeneas-cli.py'],
    pathex=[],
    binaries=None,
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher
)
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='aeneas-cli',
    debug=False,
    strip=False,
    upx=True,
    console=True
)
