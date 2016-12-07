.. _clitutorial:

aeneas Built-in Command Line Tools Tutorial
===========================================

This tutorial explains how to process tasks and jobs with
the command line tools ``aeneas.tools.execute_task`` and ``aeneas.tools.execute_job``.

(If you are interested in using ``aeneas``
as a Python package in your own application,
please consult the :ref:`libtutorial`.)

Processing Tasks
~~~~~~~~~~~~~~~~

First, we need some definitions:

.. topic:: Audio File

    An audio file is a file on disk containing audio data,
    usually text narrated by a human being.
    The audio format can be any of those supported by ``ffprobe`` and ``ffmpeg``,
    including: FLAC, MP3, MP4/AAC, OGG, WAVE, etc.

    Example: ``/home/rb/audio.mp3``

.. topic:: Text File

    A text file is a file on disk containing the textual data
    to be aligned with a matching audio file.
    The format of the text file can be any format listed in :data:`~aeneas.textfile.TextFileFormat.ALLOWED_VALUES`.
    The contents of the text file define, explicitly or implicity,
    a segmentation of the entire text into fragments,
    which can have arbitrary granularity (paragraph, sentence, sub-sentence, word, etc.),
    can be nested in a hierarchical structure,
    can consist of multiple lines,
    and can be associated to unique identifiers.
    Certain input formats require the user to specify
    additional parameters to parse the input file.

    Example of a text file ``/home/rb/text.txt`` in :data:`~aeneas.textfile.TextFileFormat.PLAIN` format, with three fragments::

        Text of the first fragment
        Text of the second fragment
        Text of the third fragment    

.. topic:: Sync Map File

    A sync map file is a file on disk
    which expresses the correspondence between an audio file and a text file.
    Specifically, for each fragment in the text file,
    it declares a time interval in the audio file where
    the text of the fragment is spoken.
    The actual format of the sync map file depends on the intended application.
    Available formats are listed in :data:`~aeneas.syncmap.SyncMapFormat.ALLOWED_VALUES`.
    Text fragments can be represented by the full text and/or by their unique idenfiers.

    Example of a sync map file in :data:`~aeneas.syncmap.SyncMapFormat.CSV` format::

        f001,0.000,1.234,First fragment text
        f002,1.234,5.678,Second fragment text
        f003,5.678,7.890,Third fragment text

.. topic:: Task

    A Task is a triple ``(audio file, text file, parameters)``.
    When a task is processed (executed), a sync map is computed
    for the given audio and text files.
    The ``parameters`` control how the alignment is computed, for example:

    * specifying the language and the format of the input text;
    * setting the format of the sync map file to be output;
    * excluding the head/tail of the audio file because they contain speech not present in the text;
    * modifying the time step of the aligner;
    * etc.

    Example (continued):

        * audio file: ``/home/rb/audio.mp3``
        * text file: ``/home/rb/text.txt``
        * parameters:
            * text in PLAIN format
            * language is ENGLISH
            * output in JSON format

The ``aeneas.tools.execute_task`` tool processes a Task
and writes the corresponding sync map to file.
Therefore, it requires at least four arguments:

* the path of the input audio file;
* the path of the input text file;
* the parameters, formatted as a ``key1=value1|key2=value2|...|keyN=valueN`` string;
* the path of the sync map to be created.

Showing Help Messages
---------------------

If you execute the program without arguments,
it will print the following help message:

.. literalinclude:: _static/execute_task_help.txt
    :language: text

If you pass the ``--help`` argument,
it will print a slightly more verbose version:

.. literalinclude:: _static/execute_task_help_arg.txt
    :language: text

Showing And Running Built-In Examples
-------------------------------------

aeneas includes some example input files which cover common use cases,
enabling the user to run live examples.
To list them, pass the ``--examples`` switch:

.. literalinclude:: _static/execute_task_examples.txt
    :language: text

Similarly, the ``--examples-all`` switch prints a list
of more than twenty built-in examples,
covering more specific input/output/parameter combinations.

.. literalinclude:: _static/execute_task_examples_all.txt
    :language: text

Running a built-in example can help learning quickly all the options/parameters
available in ``aeneas``.

For example, passing the ``--example-json`` switch will produce:

.. literalinclude:: _static/execute_task_example_json.txt
    :language: text

.. warning::

    If the above command generates an error, be sure to have
    a directory named ``output`` in your current working directory.
    If one does not exist, create it.

As you can see in the example above, built-in examples
will print the command line arguments they shortcut.
Therefore, the example above is essentially equivalent to:

.. literalinclude:: _static/execute_task_example_json_2.txt
    :language: text

.. note::

    There is a formal difference: when running an example,
    no validation of the input files and parameters is performed.
    In fact, by default they are validated using a
    :class:`~aeneas.validator.Validator` object,
    created and run automatically for you.
    If a validation error occurs,
    the execution of the Task does not begin.
    You can override this safety check with the ``--skip-validator`` switch.

In both cases, a new file ``output/sonnet.json`` is created,
containing the sync map in JSON format:

.. literalinclude:: _static/execute_task_example_json_output.txt
    :language: json

for the input file:

.. literalinclude:: _static/execute_task_example_json_input.txt
    :language: text

Verbose Output And Logging To File
----------------------------------

If you want more verbose output, you can pass the ``-v`` or ``--verbose`` switch:

.. literalinclude:: _static/execute_task_example_json_verbose.txt
    :language: text

There is also a ``-vv`` or ``--very-verbose`` switch
to increase the verbosity of the output.

Sometimes it is easier to dump the log to file, and then inspect it
with a text editor. To do so, just specify the ``-l`` switch:

.. literalinclude:: _static/execute_task_example_json_log.txt
    :language: text

The path of the log file will be printed.
By default, the log file will be created in the temporary directory of your OS.
If you want your log file to be created at a specific path,
use ``--log=/path/to/your.log`` instead of ``-l``.

Note that you can specify both ``-v``/``-vv`` and ``-l``/``--log``.

Input Text Formats
------------------

``aeneas`` is able to read several text file formats, listed in
:class:`~aeneas.textfile.TextFileFormat`:

#. :data:`~aeneas.textfile.TextFileFormat.PLAIN`,
   one fragment per line
   (example: ``--example-json``):

   .. code-block:: text
    
    Text of the first fragment
    Text of the second fragment
    Text of the third fragment
   
#. :data:`~aeneas.textfile.TextFileFormat.PARSED`,
   one fragment per line, starting with an explicit identifier
   (example: ``--example-tsv``):
    
   .. code-block:: text
    
    f001|Text of the first fragment
    f002|Text of the second fragment
    f003|Text of the third fragment
   

#. :data:`~aeneas.textfile.TextFileFormat.SUBTITLES`,
   fragments separated by a blank line, each fragment
   might span multiple lines. This format is suitable
   for creating subtitle sync map files
   (example: ``--example-srt``):

   .. code-block:: text
    
    Fragment on a single row

    Fragment on two rows
    because it is quite long

    Another one liner

    Another fragment
    on two rows

#. :data:`~aeneas.textfile.TextFileFormat.UNPARSED`,
   XML file from which text fragments will be extracted
   by matching ``id`` and/or ``class`` attributes
   (example: ``--example-smil``):

   .. literalinclude:: _static/unparsed.xhtml
    :language: xml

#. :data:`~aeneas.textfile.TextFileFormat.MPLAIN`,
   the multilevel equivalent to PLAIN,
   with paragraphs separated by a blank line,
   one sentence per line,
   and words separated by blank spaces
   (example: ``--example-mplain-json``):

   .. code-block:: text
    
    First sentence of Paragraph One.
    Second sentence of Paragraph One.

    First sentence of Paragraph Two.

    First sentence of Paragraph Three.
    Second sentence of Paragraph Three.
    Third sentence of Paragraph Three.

#. :data:`~aeneas.textfile.TextFileFormat.MUNPARSED`,
   the multilevel equivalent to UNPARSED
   (example: ``--example-munparsed-json``):

   .. literalinclude:: _static/munparsed.xhtml
    :language: xml

If you use :data:`~aeneas.textfile.TextFileFormat.UNPARSED` files,
you need to provide the following additional parameters:

* at least one of :data:`~aeneas.globalconstants.PPN_TASK_IS_TEXT_UNPARSED_ID_REGEX`
  and :data:`~aeneas.globalconstants.PPN_TASK_IS_TEXT_UNPARSED_CLASS_REGEX`,
  to select the elements from which text will be considered;
* :data:`~aeneas.globalconstants.PPN_TASK_IS_TEXT_UNPARSED_ID_SORT`
  to specify how extracted elements should be sorted, based on their ``id`` attributes.

.. literalinclude:: _static/execute_task_example_smil.txt
    :language: text

.. note::
    Even if you only specify the
    :data:`~aeneas.globalconstants.PPN_TASK_IS_TEXT_UNPARSED_CLASS_REGEX`
    regex, your XML elements still need to have ``id`` attributes.
    This is required for e.g. SMIL output to make sense.
    (Although the EPUB 3 Media Overlays specification allows you
    to specify an EPUB CFI instead of an ``id`` value,
    it is recommended to use ``id`` values
    for maximum reading system compatibility,
    and hence ``aeneas`` only outputs SMIL files with ``id`` references.)

Similarly, for :data:`~aeneas.textfile.TextFileFormat.MUNPARSED` files
you need to provide the following additional parameters:

* :data:`~aeneas.globalconstants.PPN_TASK_IS_TEXT_MUNPARSED_L1_ID_REGEX`, 
* :data:`~aeneas.globalconstants.PPN_TASK_IS_TEXT_MUNPARSED_L2_ID_REGEX`, and
* :data:`~aeneas.globalconstants.PPN_TASK_IS_TEXT_MUNPARSED_L3_ID_REGEX`.

.. literalinclude:: _static/execute_task_example_munparsed.txt
    :language: text

.. note::
    If you are interested in synchronizing at **word granularity**,
    it is highly suggested to use:
   
    1. MFCC nonspeech masking;
    2. a **multilevel text format**,
       even if you are going to use only the timings for the finer granularity;
    3. better TTS engines, like Festival or AWS/Nuance TTS API;

    as they generally yield more accurate timings.

    (If you do not want the output sync map file to contain
    the multilevel tree hierarchy for the timings,
    you might "flatten" the output sync map file,
    retaining only the word-level timings,
    by using the configuration parameter
    :data:`~aeneas.globalconstants.PPN_TASK_OS_FILE_LEVELS`
    with value ``3``).

    Since ``aeneas`` v1.7.0,
    the ``aeneas.tools.execute_task`` has a switch ``--presets-word``
    that enables MFCC nonspeech masking for single level tasks or
    MFCC nonspeech masking on level 3 (word) for multilevel tasks.
    For example::

        $ python -m aeneas.tools.execute_task --example-words
        $ python -m aeneas.tools.execute_task --example-words --presets-word
        $ python -m aeneas.tools.execute_task --example-words-multilevel
        $ python -m aeneas.tools.execute_task --example-words-multilevel --presets-word

    The other default settings should be fine for most users,
    however if you need finer control, feel free to experiment
    with the following parameters.

    Starting with ``aeneas`` v1.5.1,
    you can specify different MFCC parameters for each level, see:

    * :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.MFCC_WINDOW_LENGTH_L1`,
    * :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.MFCC_WINDOW_SHIFT_L1`,
    * :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.MFCC_WINDOW_LENGTH_L2`,
    * :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.MFCC_WINDOW_SHIFT_L2`,
    * :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.MFCC_WINDOW_LENGTH_L3`,
    * :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.MFCC_WINDOW_SHIFT_L3`.
    
    Starting with ``aeneas`` v1.6.0,
    you can also specify a different TTS engine for each level, see:

    * :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.TTS_L1`,
    * :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.TTS_L2`,
    * :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.TTS_L3`.

    Starting with ``aeneas`` v1.7.0,
    you can specify the MFCC nonspeech masking, for both
    single level tasks and multilevel tasks.
    In the latter case, you can apply it to each level separately, see:

    * :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.MFCC_MASK_NONSPEECH`,
    * :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.MFCC_MASK_NONSPEECH_L1`,
    * :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.MFCC_MASK_NONSPEECH_L2`,
    * :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.MFCC_MASK_NONSPEECH_L3`.
    
    If you are using a multilevel text format,
    you might want to enable MFCC masking only for level 3 (word),
    as enabling it for level 1 and 2 does not seem to yield significantly
    better results.

    The ``aeneas`` mailing list contains some interesting threads
    about using aeneas for word-level synchronization.

Output Sync Map Formats
-----------------------

``aeneas`` is able to write the sync map into several formats, listed in
:class:`~aeneas.syncmap.SyncMapFormat`.

As for the input text, certain output sync map formats
require the user to specify additional parameters
to correctly create the output file.
For example,
:data:`~aeneas.syncmap.SyncMapFormat.SMIL`
requires:

* :data:`~aeneas.globalconstants.PPN_TASK_OS_FILE_SMIL_AUDIO_REF` and
* :data:`~aeneas.globalconstants.PPN_TASK_OS_FILE_SMIL_PAGE_REF`.

Example:

.. literalinclude:: _static/execute_task_example_smil.txt
    :language: text

Listing Parameter Names And Values
----------------------------------

Since there are dozens of parameter names and values,
it is easy to forget their correct spelling.
You can use the ``--list-parameters`` switch to print
the list of parameter names that you can use in the configuration string.

.. literalinclude:: _static/execute_task_list_parameters.txt
    :language: text

For parameters that accept a restricted set of values,
you can list the allowed values with ``--list-values=PARAM``.
For example:

.. literalinclude:: _static/execute_task_list_values.txt
    :language: text

Downloading Audio From YouTube
------------------------------

``aeneas`` can download the audio stream from a YouTube video.
Instead of the audio file path, you provide the YouTube URL,
and add the ``-y`` switch at the end:

.. literalinclude:: _static/execute_task_youtube.txt
    :language: text

.. warning::

    The download feature is experimental,
    and it might be unavailable in the future,
    for example if YouTube disables API access
    to audio/video contents.
    Also note that sometimes the download fails
    for network/backend reasons: just wait a few seconds
    and try executing again.

The Runtime Configuration
-------------------------

Although the default settings should be fine for most users,
sometimes it might be useful to modify certain internal parameters
affecting the processing of tasks, for example
changing the directory where temporary files are created,
modifying processing parameters like the time resolution, etc.

To do so, the user can use the ``-r`` or ``--runtime-configuration`` switch,
providing a suitable configuration string as its value.

.. warning::

    Using the runtime configuration switch is advisable
    only to expert users or if explicitly suggested by expert users,
    since there are (almost) no sanity checks on the values provided
    this way, and setting wrong values might lead to erratic behaviors
    of the aligner.

The available paramenter names are listed in
:class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`.

Examples:

#. disable checks on the language codes:

    .. code-block:: text

        python -m aeneas.tools.execute_task --example-json -r="allow_unlisted_languages=True"

#. disable the Python C/C++ extensions, running the pure Python code:

    .. code-block:: text

        python -m aeneas.tools.execute_task --example-json -r="c_extensions=False"

#. disable only the ``cew`` Python C/C++ extension, while ``cdtw`` and ``cmfcc`` will still run (if compiled):

    .. code-block:: text

        python -m aeneas.tools.execute_task --example-json -r="cew=False"

#. set the DTW margin to ``10.000`` seconds:

    .. code-block:: text

        python -m aeneas.tools.execute_task --example-json -r="dtw_margin=10"

#. specify the path to the ``ffprobe`` and ``ffmpeg`` executables:

    .. code-block:: text

        python -m aeneas.tools.execute_task --example-json -r="ffmpeg_path=/path/to/my/ffmpeg|ffprobe_path=/path/to/my/ffprobe"

#. set the time resolution of the aligner to ``0.050`` seconds:

    .. code-block:: text

        python -m aeneas.tools.execute_task --example-json -r="mfcc_window_length=0.150|mfcc_window_shift=0.050"

#. use the eSpeak-ng TTS, via the ``espeak-ng`` executable available on ``$PATH``, instead of eSpeak:

    .. code-block:: text

        python -m aeneas.tools.execute_task --example-json -r="tts=espeak-ng"

#. use the eSpeak-ng TTS, via the ``espeak-ng`` executable at a custom location, instead of eSpeak:

    .. code-block:: text

        python -m aeneas.tools.execute_task --example-json -r="tts=espeak-ng|tts_path=/path/to/espeak-ng"

#. use the Festival TTS, via the ``text2wave`` executable available on ``$PATH``, instead of eSpeak:

    .. code-block:: text

        python -m aeneas.tools.execute_task --example-json -r="tts=festival"

#. use the Festival TTS, via the ``text2wave`` executable at a custom location, instead of eSpeak:

    .. code-block:: text

        python -m aeneas.tools.execute_task --example-json -r="tts=festival|tts_path=/path/to/text2wave"

#. use the AWS Polly TTS API instead of eSpeak (with TTS caching enabled):

    .. code-block:: text

        python -m aeneas.tools.execute_task --example-json -r="tts=aws|tts_cache=True"

#. use the Nuance TTS API instead of eSpeak (with TTS caching enabled):

    .. code-block:: text

        python -m aeneas.tools.execute_task --example-json -r="tts=nuance|nuance_tts_api_id=YOUR_NUANCE_API_ID|nuance_tts_api_key=YOUR_NUANCE_API_KEY|tts_cache=True"

#. use a custom TTS wrapper located at ``/path/to/your/wrapper.py`` (see the ``aeneas/extra/`` directory for examples):

    .. code-block:: text

        python -m aeneas.tools.execute_task --example-json -r="tts=custom|tts_path=/path/to/your/wrapper.py"

#. set the temporary directory:

    .. code-block:: text

        python -m aeneas.tools.execute_task --example-json -r="tmp_path=/path/to/tmp/"

#. allow processing tasks with audio files at most 1 hour (= 3600 seconds) long:

    .. code-block:: text

        python -m aeneas.tools.execute_task --example-json -r="task_max_audio_length=3600"

Miscellanea
-----------

#. ``--example-head-tail``: ignore the first ``0.400`` seconds and
   the last ``0.500`` seconds of the audio file for alignment purposes
#. ``--example-no-zero``: ensure that no fragment in the output sync map has zero length
#. ``--example-percent``: adjust the output sync map,
   setting each boundary between adjacent fragments to the middle of the nonspeech interval,
   using the :data:`~aeneas.adjustboundaryalgorithm.AdjustBoundaryAlgorithm.PERCENT` algorithm
   with value ``50`` (i.e., ``50%``)
#. ``--example-rate``: adjust the output sync map, trying to ensure that no fragment has
   a rate of more than ``14`` character/s,
   using the :data:`~aeneas.adjustboundaryalgorithm.AdjustBoundaryAlgorithm.RATE` algorithm
#. ``--example-sd``: detect the audio head/tail, each at most ``10.000`` seconds long
#. ``--example-multilevel-tts``: use different TTS engines for different levels
   (``mplain`` multilevel input text)

Processing Jobs
~~~~~~~~~~~~~~~

If you have several Tasks sharing the same parameters (configuration strings)
and changing only in their audio/text files,
you can either write your own Bash/BAT script, or
you might want to create a Job:

.. topic:: Job

    A Job is a container (compressed file or uncompressed directory),
    containing:
    
    * one or more pairs audio/text files, and
    * a configuration file (``config.txt`` or ``config.xml``)
      specifying parameters to locate each Task assets inside the Job,
      to process each Task, and to create the output container
      containing the output sync map files.

    Example: ``/home/rb/job.zip``, containing the following files,
    corresponding to three Tasks:

    .. code-block:: text

        .
        ├── config.txt
        └── OEBPS
            └── Resources
                ├── sonnet001.mp3
                ├── sonnet001.txt
                ├── sonnet002.mp3
                ├── sonnet002.txt
                ├── sonnet003.mp3
                └── sonnet003.txt

The ``aeneas.tools.execute_job`` tool processes a Job
and writes the corresponding output container to file.
Therefore, it requires at least two arguments:

* the path of the input job container;
* the path of an existing directory where the output container should be created.

The ``--help``, ``-v``, ``-l``, and ``-r`` switches
have the same meaning for ``aeneas.tools.execute_job``
as described above. For example, the help message reads:

.. literalinclude:: _static/execute_job_help.txt
    :language: text

Currently ``aeneas.tools.execute_job`` does not have
built-in examples shortcuts (``--example-*``),
but you can run a built-in example:

.. literalinclude:: _static/execute_job_example.txt
    :language: text

TXT Config File (``config.txt``)
--------------------------------

A ZIP container with the following files:

.. code-block:: text

    .
    ├── config.txt
    └── OEBPS
        └── Resources
            ├── sonnet001.mp3
            ├── sonnet001.txt
            ├── sonnet002.mp3
            ├── sonnet002.txt
            ├── sonnet003.mp3
            └── sonnet003.txt

where the ``config.txt`` config file reads:

.. literalinclude:: _static/execute_job_config.txt

will generate three tasks (``sonnet001``, ``sonnet002`` and ``sonnet003``),
output a SMIL file for each of them,
finally compress them in a ZIP file with the following structure:

.. code-block:: text

    .
    └── OEBPS
        └── Resources
            ├── sonnet001.smil
            ├── sonnet002.smil
            └── sonnet003.smil

Note that the paths in ``config.txt`` are relative to
(the directory containing) the ``config.txt`` file,
and that you can use the :data:`~aeneas.globalconstants.PPV_OS_TASK_PREFIX`
placeholder (``$PREFIX``) that will be replaced with the Task id.

XML Config File (``config.xml``)
--------------------------------
            
While ``config.txt`` is concise and easy to write,
it constraints all the tasks of the job to share the same
execution settings (language, output format, and so on).

If you need to specify different values for execution parameters
of different tasks, you must use an XML config file,
named ``config.xml``.

The following ``config.xml`` is equivalent to the example above:

.. literalinclude:: _static/execute_job_config_xml_1.txt
    :language: xml

Now note that ``config.xml`` allows you to bundle together
Tasks with different languages, output formats, etc.:

.. literalinclude:: _static/execute_job_config_xml_2.txt
    :language: xml



