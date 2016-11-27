# Virtual Environments

This directory contains Bash scripts automating the creation
of virtual environments ("venv"s) using ``virtualenv``
for testing purposes.


## Requirements

1. The OS requirements of ``aeneas`` (``espeak``, ``ffmpeg``, ``festival``, etc.)
2. ``virtualenv``, version 15.0.3 or later
3. Your choice of Python interpreters (e.g., ``python2.7``, ``python3.5``, ``pypy``, etc.), which should be available on the command line (i.e., be in your ``PATH`` environment variable).

## Usage

In the examples, ``python3.5`` is the target interpreter.
For example, on Debian:

```bash
$ which python3.5
/usr/bin/python3.5
```

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

### Install new version of aeneas from the sdist archive

To install from ``../dist/aeneas*.tar.gz``:

```bash
$ bash manage_venvs.sh python3.5 install
$ bash manage_venvs.sh python3.5 sdist
```

To install from file ``/tmp/foo.tar.gz``:

```bash
$ bash manage_venvs.sh python3.5 install
$ bash manage_venvs.sh python3.5 sdist /tmp/foo.tar.gz
```

This procedure will:

1. create the venv
2. activate the venv
3. uninstall ``aeneas``, if already installed
4. copy the sdist archive inside the venv and install it
5. delete the sdist archive inside the venv
6. perform the diagnostic tests
7. deactivate the venv

To uninstall ``aeneas`` from the venv:

```bash
$ bash manage_venvs.sh python3.5 sdist --remove
```

To force building the ``cfw`` extension:

```bash
$ bash manage_venvs.sh python3.5 install
$ AENEAS_FORCE_CFW=True bash manage_venvs.sh python3.5 sdist
```

### Manual testing

```bash
$ bash manage_venvs.sh python3.5 full
$ cd venv_python3.5
$ source bin/activate
$ cd tests

$ # do your testing here
$ # for example:
$ python run_all_unit_tests.py
$ python run_all_unit_tests.py -l -v
$ python run_all_unit_tests.py task
$ python run_all_unit_tests.py configuration -v
$ # etc ...

$ cd ..
$ deactivate
$ cd ..
```

Note: ``full`` is equivalent to ``install`` + ``deps`` + ``tests``.

### Automated testing

First, do a full install of the venv:

```bash
$ bash manage_venvs.sh python3.5 full
```

Then, each time you want to run the tests:

```bash
$ # all
$ bash run_tests.sh python3.5 all

$ # fast
$ bash run_tests.sh python3.5 fast

$ # tool
$ bash run_tests.sh python3.5 tool

$ # long
$ bash run_tests.sh python3.5 long

$ # bench
$ bash run_tests.sh python3.5 bench

$ # net
$ bash run_tests.sh python3.5 net

$ # all, except net
$ bash run_tests.sh python3.5 nonet
```
