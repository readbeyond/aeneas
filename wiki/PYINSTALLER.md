# Compiling And Distributing aeneas As A Single Executable

The root directory of the **aeneas** repository contains
two [PyInstaller](http://www.pyinstaller.org/) configuration files
that can be used to compile **aeneas** into an executable
to be distributed to other users,
who will not need to install Python,
Python dependencies, or the Python C extension compiler.

(The end users will still need to install **eSpeak** and **FFmpeg** separately
before running the compiled **aeneas** executable.
If you are looking for an "all-in-one" installer, please consult
the [INSTALL](INSTALL.md) file.)

You can choose to compile **aeneas** to either:

* a single executable file (use ``pyinstaller-onefile.spec``), or
* a single directory (use ``pyinstaller-onedir.spec``).

The single directory approach is slightly faster at run time,
since it does not need to unpack the support libraries and files
to a temporary location.
On the other hand, the single executable file is easier
to distribute to your end users.
If you compile to a single executable,
installing the [UPX](http://upx.sourceforge.net/) compressor
is strongly suggested, and
PyInstall will run it automatically if found.

Important notes (for both cases):

1. PyInstaller will package the hydra script
   ``pyinstaller-aeneas-cli.py``, which allows the user to access
   all the ``aeneas.tools.*`` programs.
   Run it without arguments to get its manual.
2. The resulting compiled executable will run only
   on the same platform (OS type, 32/64 bit) where it was created.
   For example,
   if you compile the executable on Linux,
   the latter will not work on Mac OS X or Windows.
   However, if you create a Windows 32 bit executable,
   the latter might run on Windows 64 bit in compatibility mode (not tested).
   Since all recent Macs are 64 bit machines,
   on Mac OS X there should be no issues.
3. PyInstall will package any Python C extensions compiled
   in the ``aeneas`` source directory,
   hence make sure you have them compiled correctly.
4. PyInstall will take care only of the "Python part" of **aeneas**
   (the Python source code, the C extensions, and their Python dependencies).
   You will still need to install **eSpeak** and **FFmpeg** separately,
   in order to use **aeneas**.
5. You might want to use PyInstaller v3.1.1,
   since, at the time of writing, v3.2 seems buggy on Mac OS X and Windows.

## Usage

1. (Optional but recommended)
   Install the [UPX](http://upx.sourceforge.net/) compressor,
   and add its path to your ``PATH`` environment variable,
   so that PyInstall can invoke it.

2. Install PyInstaller:

    ```bash
    $ pip install pyinstaller==3.1.1
    ```

3. Set **aeneas** up locally:

    ```bash
    $ python setup.py build_ext --inplace
    ```

4. Delete the ``build`` and ``dist`` directories, to avoid caching effects:

    ```bash
    $ rm -r build/ dist/
    ```

5. Compile the single file executable using the provided configuration:

    ```bash
    $ pyinstaller pyinstaller-onefile.spec
    ```

    or, if you prefer the directory output:

    ```bash
    $ pyinstaller pyinstaller-onedir.spec
    ```

6. Distribute the file(s) created in the ``dist/`` directory.



