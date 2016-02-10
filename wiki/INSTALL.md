# Installing aeneas

## OS Independent, via pip

1. Make sure you have
    `ffmpeg`, `ffprobe` (usually provided by the `ffmpeg` package),
    and `espeak` installed and available on your command line.
    You also need Python and its "developer" package
    containing the C headers (`python-dev` or similar).

2. Install `aeneas` system-wise with `pip`:
    
    ```bash
    $ sudo pip install numpy
    $ sudo pip install aeneas
    (Optional: $ sudo pip install pafy)
    ```

    **Note**: you must install `numpy` before `aeneas`,
    otherwise the setup process will fail.

    **Note**: you can install `aeneas` via `pip`
    in a virtual environment (e.g. created by `virtualenv`).

## Linux

1. If you are a user of a `deb`-based Linux distribution
(e.g., Debian or Ubuntu),
you can install all the dependencies by downloading and running
[the provided install_dependencies.sh script](https://raw.githubusercontent.com/readbeyond/aeneas/master/install_dependencies.sh)

    ```bash
    $ wget https://raw.githubusercontent.com/readbeyond/aeneas/master/install_dependencies.sh
    $ sudo bash install_dependencies.sh
    ```

    If you have another Linux distribution,
    just make sure you have
    `ffmpeg`, `ffprobe` (usually provided by the `ffmpeg` package),
    and `espeak` installed and available on your command line.
    You also need Python and its "developer" package
    containing the C headers (`python-dev` or similar).

2. Clone the `aeneas` repo, install Python dependencies, and compile C extensions:

    ```bash
    $ git clone https://github.com/ReadBeyond/aeneas.git
    $ cd aeneas
    $ sudo pip install -r requirements.txt
    (Optional: $ sudo pip install pafy)
    $ python setup.py build_ext --inplace
    $ python aeneas_check_setup.py
    ```

    If the last command prints a success message,
    you have all the required dependencies installed
    and you can confidently run **aeneas** in production.

3. In alternative to the previous point, you can install `aeneas` system-wise with `pip`:
    
    ```bash
    $ sudo pip install numpy
    $ sudo pip install aeneas
    (Optional: $ sudo pip install pafy)
    ```

## Windows

Please follow the installation instructions
contained in the
["Using aeneas for Audio-Text Synchronization" PDF](http://software.sil.org/scriptureappbuilder/resources/),
based on
[these directions](https://groups.google.com/d/msg/aeneas-forced-alignment/p9cb1FA0X0I/8phzUgIqBAAJ),
written by Richard Margetts.

Please note that on Windows it is recommended to run **aeneas**
with Python 2.7, since compiling the C extensions on Python 3.4 or 3.5
requires [a complex setup process](http://stackoverflow.com/questions/29909330/microsoft-visual-c-compiler-for-python-3-4).

## Mac OS X

Feel free to jump to step 9 if you already have
`python`, `ffmpeg`/`ffprobe` and `espeak` installed.

1. Install the Xcode command line tools:

    ```bash
    $ xcode-select --install
    ```

    Follow the instructions appearing on screen.

2. Install the `brew` packet manager:

    ```bash
    $ ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    ```

3. Update `brew`:

    ```bash
    $ brew update
    ```

4. Install `espeak` and `ffmpeg` (which also provides `ffprobe`)  via `brew`:

    ```bash
    $ brew install espeak
    $ brew install ffmpeg
    ```

5. Install Python:

    ```bash
    $ brew install python
    ```

6. Replace the default (Apple's) Python distribution with the Python installed by `brew`,
   by adding the following line at the end of your `~/.bash_profile`:

    ```bash
    export PATH=/usr/local/bin:/usr/local/sbin:~/bin:$PATH
    ```

7. Open a new terminal window. (This step is IMPORTANT!
   If you do not, you will still use Apple's Python,
   and everything in the Universe will go wrong!)

8. Check that you are running the new `python`:

    ```bash
    $ which python
    /usr/local/bin/python
    
    $ python --version
    Python 2.7.10 (or later)
    ```

9. Clone the `aeneas` repo, install Python dependencies, and compile C extensions:

    ```bash
    $ git clone https://github.com/ReadBeyond/aeneas.git
    $ cd aeneas
    $ sudo pip install -r requirements.txt
    (Optional: $ sudo pip install pafy)
    $ python setup.py build_ext --inplace
    $ python aeneas_check_setup.py
    ```

    If the last command prints a success message,
    you have all the required dependencies installed
    and you can confidently run **aeneas** in production.

10. In alternative to the previous point, you can install `aeneas` system-wise with `pip`:
    
    ```bash
    $ sudo pip install numpy
    $ sudo pip install aeneas
    (Optional: $ sudo pip install pafy)
    ```


