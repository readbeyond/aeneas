# aeneas extras 

This Python module (directory) contains
a collection of extra tools for aeneas,
mainly custom TTS engine wrappers.



## `ctw_espeak.py`

A wrapper for the `eSpeak` TTS engine
that executes `eSpeak` via `subprocess`.

This file is an example to illustrate
how to write a custom TTS wrapper,
and how to use it at runtime:

1. Copy the `ctw_espeak.py` file to `/tmp/ctw_espeak.py`
   (or any other directory you like).

2. Run any `aeneas.tools.*` with the following options:

    ```
    -r="tts=custom|tts_path=/tmp/ctw_espeak.py"
    ```

   For example:

    ```bash
    python -m aeneas.tools.execute_task --example-srt -r="tts=custom|tts_path=/tmp/ctw_espeak.py"
    ```

For details, please inspect the `ctw_espeak.py` file,
which is heavily commented and it should help you
create a new wrapper for your own TTS engine.

Note: if you want to use `eSpeak` as your TTS engine
in a production environment,
do NOT use the `ctw_espeak.py` wrapper!
`eSpeak` is the default TTS engine of `aeneas`,
and the `aeneas.espeakwrapper` in the main library
is faster than the `ctw_espeak.py` wrapper.



## `ctw_speect.py`

A wrapper for the `Speect` TTS engine
that synthesizes text via Python calls
to the `speect` Python module.

To use it, do the following:

1. Install `Speect` and compile the Python module `speect`:
see [http://speect.sourceforge.net/](http://speect.sourceforge.net/) for details.

2. Download a voice for `Speect`, for example the `Speect CMU Arctic slt` voice
(file `cmu_arctic_slt-1.0.tar.gz`
from [http://hlt.mirror.ac.za/TTS/Speect/](http://hlt.mirror.ac.za/TTS/Speect/)),
and decompress it to `/tmp/cmu_arctic_slt/`
(or any other directory you like).

3. Copy the `ctw_speect.py` file to `/tmp/cmu_arctic_slt/ctw_speect.py`
   (or any other directory you like).

4. Run any `aeneas.tools.*` with the following options:

    ```
    -r="tts=custom|tts_path=/tmp/cmu_arctic_slt/ctw_speect.py"
    ```

   For example:

    ```bash
    python -m aeneas.tools.execute_task --example-srt -r="tts=custom|tts_path=/tmp/cmu_arctic_slt/ctw_speect.py"
    ```

For details, please inspect the `ctw_speect.py` file.



