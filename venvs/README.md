# Virtual Environments

This directory contains Bash scripts automating the creation
of virtual environments ("venv"s) using ``virtualenv``
for testing purposes.


## Requirements

1. The OS requirements of ``aeneas`` (``espeak``, ``ffmpeg``, etc.)
2. ``virtualenv``, version 15.0.3 or later
3. Your choice of Python interpreters, available on the command line (e.g., ``python2.7``, ``python3.5``, ``pypy``, etc.)

## Usage

In the examples, ``python3.5`` is assumed to be the target interpreter.

### Uninstall existing venv

```bash
$ bash manage_venvs.sh python3.5 uninstall
```

### Install new venv

```bash
$ bash manage_venvs.sh python3.5 install
```

### Install Python dependencies inside the venv

```bash
$ bash manage_venvs.sh python3.5 deps
```

Note: the above will perform a ``pip install -U ...`` (upgrade).

### Install new version of aeneas from sdist tar.gz

```bash
$ bash manage_venvs.sh python3.5 install
$ cd venv_python3.5
$ source bin/activate
$ bash install_aeneas_from_sdist_tar_gz.sh
$ deactivate
``` 

This procedure will:

1. create the venv, if not already existing
2. activate the venv
3. uninstall ``aeneas``, if already installed
4. copy the tar.gz and install it
5. perform a couple of quick tests
6. deactivate the venv

Note: specify ``bash install_aeneas_from_sdist_tar_gz.sh --remove``
to just uninstall ``aeneas`` from the venv and exit.

### Install aeneas for full (local) testing

```bash
$ bash manage_venvs.sh python3.5 install
$ bash manage_venvs.sh python3.5 deps
$ bash manage_venvs.sh python3.5 tests
$ cd venv_python3.5
$ source bin/activate
$ cd tests
$ python run_all_unit_tests.py
$ cd ../..
$ deactivate
```

