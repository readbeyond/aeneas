Package ``aeneas``
==================

.. automodule:: aeneas 

Goal
----

**aeneas** automatically generates a **synchronization map**
between a list of text fragments
and an audio file containing the narration of the (same) text.
In computer science this task is known as (automatically computing a) **forced alignment**.

For example, given the verses and a ``53.280s``-long audio recording
of *Sonnet I* by William Shakespeare,
**aeneas** might compute a map like the following::

    1                                                     => [00:00:00.000, 00:00:02.680]
    From fairest creatures we desire increase,            => [00:00:02.680, 00:00:05.480]
    That thereby beauty's rose might never die,           => [00:00:05.480, 00:00:08.640]
    But as the riper should by time decease,              => [00:00:08.640, 00:00:11.960]
    His tender heir might bear his memory:                => [00:00:11.960, 00:00:15.280]
    But thou contracted to thine own bright eyes,         => [00:00:15.280, 00:00:18.520]
    Feed'st thy light's flame with self-substantial fuel, => [00:00:18.520, 00:00:22.760]
    Making a famine where abundance lies,                 => [00:00:22.760, 00:00:25.720]
    Thy self thy foe, to thy sweet self too cruel:        => [00:00:25.720, 00:00:31.240]
    Thou that art now the world's fresh ornament,         => [00:00:31.240, 00:00:34.280]
    And only herald to the gaudy spring,                  => [00:00:34.280, 00:00:36.960]
    Within thine own bud buriest thy content,             => [00:00:36.960, 00:00:40.640]
    And tender churl mak'st waste in niggarding:          => [00:00:40.640, 00:00:43.600]
    Pity the world, or else this glutton be,              => [00:00:43.600, 00:00:48.000]
    To eat the world's due, by the grave and thee.        => [00:00:48.000, 00:00:53.280]

The above map is just an abstract representation of a sync map.
In practice, the sync map will be output to a file with a precise syntax.
Currently, the following formats are supported:

#. SMIL for EPUB 3 ebooks with Media Overlays,
#. SRT/TTML/VTT for closed captioning,
#. JSON for consumption on the Web, and
#. "raw" CSV/SSV/TSV/TXT/XML for further processing.


Quick Start
-----------

Tasks
~~~~~

A **task** is the minimum work unit of ``aeneas``.
It represents a pair **(text fragments, audio file)**,
for which **a single sync map** should be computed.

The audio file might be encoded in any format that can be read by ``ffmpeg``,
such as MP3, WAV, AAC, OGG, FLAC, etc.

The text fragments could have **arbitrary granularity**
(single word, subsentence, sentence, entire paragraph),
and they could be provided either as a list of ``unicode`` strings
(when using ``aeneas`` via code) or
as **a single text file**
(when using ``aeneas.tools.execute_task`` as a program).

The text file, encoded in **UTF-8**,
must be one of the following four formats:

#. **plain**: each text fragment is on a separated line of the file::

    1
    From fairest creatures we desire increase,
    That thereby beauty's rose might never die,
    But as the riper should by time decease,
    His tender heir might bear his memory:
    But thou contracted to thine own bright eyes,
    Feed'st thy light's flame with self-substantial fuel,
    Making a famine where abundance lies,
    Thy self thy foe, to thy sweet self too cruel:
    Thou that art now the world's fresh ornament,
    And only herald to the gaudy spring,
    Within thine own bud buriest thy content,
    And tender churl mak'st waste in niggarding:
    Pity the world, or else this glutton be,
    To eat the world's due, by the grave and thee.

#. **subtitles**: each text fragment is contained in
   one or more consecutive lines, separated by a blank line.
   Use this format if you want to output to SRT/TTML/VTT
   and you want to keep multilines in the output file::

    1
    
    From fairest creatures
    we desire increase,
    
    That thereby beauty's rose
    might never die,
    
    But as the riper should by time decease,
    
    His tender heir might bear his memory:
    
    But thou contracted to thine own bright eyes,
    
    Feed'st thy light's flame
    with self-substantial fuel,
    
    Making a famine
    where abundance lies,
    
    Thy self thy foe, to thy sweet self
    too cruel:
    
    Thou that art now the world's fresh ornament,
    
    And only herald
    to the gaudy spring,
    
    Within thine own bud buriest thy content,
    
    And tender churl mak'st waste in niggarding:
    
    Pity the world,
    or else this glutton be,
    
    To eat the world's due,
    by the grave and thee.

#. **parsed**: each line of the file contains the fragment identifier
   and the corresponding text, separated by a ``|`` character::

    f001|1
    f002|From fairest creatures we desire increase,
    f003|That thereby beauty's rose might never die,
    f004|But as the riper should by time decease,
    f005|His tender heir might bear his memory:
    f006|But thou contracted to thine own bright eyes,
    f007|Feed'st thy light's flame with self-substantial fuel,
    f008|Making a famine where abundance lies,
    f009|Thy self thy foe, to thy sweet self too cruel:
    f010|Thou that art now the world's fresh ornament,
    f011|And only herald to the gaudy spring,
    f012|Within thine own bud buriest thy content,
    f013|And tender churl mak'st waste in niggarding:
    f014|Pity the world, or else this glutton be,
    f015|To eat the world's due, by the grave and thee.

#. **unparsed**: an XML (e.g., XHTML) file, where text fragments
   can be parsed out of elements with
   ``id`` and/or ``class`` attributes matching a given regular expression
   (``id=f[0-9]+`` in the example)::

    <?xml version="1.0" encoding="UTF-8"?>
    <html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="en" xml:lang="en">
     <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=768,height=1024"/>
      <link rel="stylesheet" href="../Styles/style.css" type="text/css"/>
      <title>Sonnet I</title>
     </head>
     <body>
      <div id="divTitle">
       <h1><span class="ra" id="f001">I</span></h1>
      </div>
      <div id="divSonnet"> 
       <p>
        <span class="ra" id="f002">From fairest creatures we desire increase,</span><br/>
        <span class="ra" id="f003">That thereby beauty’s rose might never die,</span><br/>
        <span class="ra" id="f004">But as the riper should by time decease,</span><br/>
        <span class="ra" id="f005">His tender heir might bear his memory:</span><br/>
        <span class="ra" id="f006">But thou contracted to thine own bright eyes,</span><br/>
        <span class="ra" id="f007">Feed’st thy light’s flame with self-substantial fuel,</span><br/>
        <span class="ra" id="f008">Making a famine where abundance lies,</span><br/>
        <span class="ra" id="f009">Thy self thy foe, to thy sweet self too cruel:</span><br/>
        <span class="ra" id="f010">Thou that art now the world’s fresh ornament,</span><br/>
        <span class="ra" id="f011">And only herald to the gaudy spring,</span><br/>
        <span class="ra" id="f012">Within thine own bud buriest thy content,</span><br/>
        <span class="ra" id="f013">And tender churl mak’st waste in niggarding:</span><br/>
        <span class="ra" id="f014">Pity the world, or else this glutton be,</span><br/>
        <span class="ra" id="f015">To eat the world’s due, by the grave and thee.</span>
       </p>
      </div>
     </body>
    </html>

For ``aeneas`` to be able to execute a task, that is,
to compute the sync map between its text fragments and its audio file,
the user must specify **some properties** of the task.
The most important are:

#. the format of the input text file,
#. in case of ``unparsed`` text, the regex to extract the text fragments
   and the rule for sorting them (e.g., by lexicographically sorting
   their ``id`` values),
#. the language of the text,
#. the desired format for the output sync map, along with additional parameters
   (like ``src`` values for ``<audio>`` and ``<text>`` for SMIL sync maps),
#. the name of the output sync map file.

To do so, the user provides a configuration string, which
is a ``|``-separated list of ``key=value`` pairs that looks like::

    task_language=en|os_task_file_format=smil|os_task_file_smil_audio_ref=p001.mp3|os_task_file_smil_page_ref=p001.xhtml|is_text_type=unparsed|is_text_unparsed_id_regex=f[0-9]+|is_text_unparsed_id_sort=numeric

Breaking it down into pairs::

    task_language=en
    os_task_file_format=smil
    os_task_file_smil_audio_ref=p001.mp3
    os_task_file_smil_page_ref=p001.xhtml
    is_text_type=unparsed
    is_text_unparsed_id_regex=f[0-9]+
    is_text_unparsed_id_sort=numeric

its meaning appears clearer: this task is made
by audio file ``p001.mp3`` and text file ``p001.xhtml``.
The latter is ``unparsed``, and the text is contained in elements
with ``id`` matching the regex ``f[0-9]+``, which should be parsed out
and sorted according to the value of the numerical part of their ``id``
(in the example, lexicographic and numeric order coincides).
The output sync map must be in ``smil`` format, with
``<audio src="p001.mp3" ... >`` and ``<text src="p001.xhtml#..." ... >``.

The resulting sync map will be::

    <smil xmlns="http://www.w3.org/ns/SMIL" xmlns:epub="http://www.idpf.org/2007/ops" version="3.0">
     <body>
      <seq id="s000001" epub:textref="p001.xhtml">
       <par id="p000001">
        <text src="p001.xhtml#f001"/>
        <audio clipBegin="00:00:00.000" clipEnd="00:00:02.680" src="p001.mp3"/>
       </par>
       <par id="p000002">
        <text src="p001.xhtml#f002"/>
        <audio clipBegin="00:00:02.680" clipEnd="00:00:05.480" src="p001.mp3"/>
       </par>
       <par id="p000003">
        <text src="p001.xhtml#f003"/>
        <audio clipBegin="00:00:05.480" clipEnd="00:00:08.640" src="p001.mp3"/>
       </par>
       <par id="p000004">
        <text src="p001.xhtml#f004"/>
        <audio clipBegin="00:00:08.640" clipEnd="00:00:11.960" src="p001.mp3"/>
       </par>
       <par id="p000005">
        <text src="p001.xhtml#f005"/>
        <audio clipBegin="00:00:11.960" clipEnd="00:00:14.320" src="p001.mp3"/>
       </par>
       <par id="p000006">
        <text src="p001.xhtml#f006"/>
        <audio clipBegin="00:00:14.320" clipEnd="00:00:18.840" src="p001.mp3"/>
       </par>
       <par id="p000007">
        <text src="p001.xhtml#f007"/>
        <audio clipBegin="00:00:18.840" clipEnd="00:00:22.760" src="p001.mp3"/>
       </par>
       <par id="p000008">
        <text src="p001.xhtml#f008"/>
        <audio clipBegin="00:00:22.760" clipEnd="00:00:25.320" src="p001.mp3"/>
       </par>
       <par id="p000009">
        <text src="p001.xhtml#f009"/>
        <audio clipBegin="00:00:25.320" clipEnd="00:00:31.240" src="p001.mp3"/>
       </par>
       <par id="p000010">
        <text src="p001.xhtml#f010"/>
        <audio clipBegin="00:00:31.240" clipEnd="00:00:34.280" src="p001.mp3"/>
       </par>
       <par id="p000011">
        <text src="p001.xhtml#f011"/>
        <audio clipBegin="00:00:34.280" clipEnd="00:00:36.480" src="p001.mp3"/>
       </par>
       <par id="p000012">
        <text src="p001.xhtml#f012"/>
        <audio clipBegin="00:00:36.480" clipEnd="00:00:40.640" src="p001.mp3"/>
       </par>
       <par id="p000013">
        <text src="p001.xhtml#f013"/>
        <audio clipBegin="00:00:40.640" clipEnd="00:00:43.600" src="p001.mp3"/>
       </par>
       <par id="p000014">
        <text src="p001.xhtml#f014"/>
        <audio clipBegin="00:00:43.600" clipEnd="00:00:48.000" src="p001.mp3"/>
       </par>
       <par id="p000015">
        <text src="p001.xhtml#f015"/>
        <audio clipBegin="00:00:48.000" clipEnd="00:00:53.240" src="p001.mp3"/>
       </par>
      </seq>
     </body>
    </smil>

For simpler I/O files, configuration strings might be as short as::

    task_language=en|os_task_file_format=srt|is_text_type=parsed

which yields::

    1
    00:00:00,000 --> 00:00:02,680
    1

    2
    00:00:02,680 --> 00:00:05,480
    From fairest creatures we desire increase,

    3
    00:00:05,480 --> 00:00:08,640
    That thereby beauty's rose might never die,

    4
    00:00:08,640 --> 00:00:11,960
    But as the riper should by time decease,

    5
    00:00:11,960 --> 00:00:14,320
    His tender heir might bear his memory:

    6
    00:00:14,320 --> 00:00:18,560
    But thou contracted to thine own bright eyes,

    7
    00:00:18,560 --> 00:00:22,760
    Feed'st thy light's flame with self-substantial fuel,

    8
    00:00:22,760 --> 00:00:25,320
    Making a famine where abundance lies,

    9
    00:00:25,320 --> 00:00:31,240
    Thy self thy foe, to thy sweet self too cruel:

    10
    00:00:31,240 --> 00:00:34,280
    Thou that art now the world's fresh ornament,

    11
    00:00:34,280 --> 00:00:36,480
    And only herald to the gaudy spring,

    12
    00:00:36,480 --> 00:00:40,640
    Within thine own bud buriest thy content,

    13
    00:00:40,640 --> 00:00:43,600
    And tender churl mak'st waste in niggarding:

    14
    00:00:43,600 --> 00:00:48,000
    Pity the world, or else this glutton be,

    15
    00:00:48,000 --> 00:00:53,240
    To eat the world's due, by the grave and thee.

The list of available parameter keys can be found
in :class:`aeneas.globalconstants`.
It is higly recommended to look at the provided examples
in ``aeneas/tests/res/example_jobs``.

Job
~~~

``aeneas`` offers functionalities for **batch processing**
several related tasks, which is a convenient way
of processing, for example, SMIL files for EPUB 3 FXL or EPUB 3 Audio-eBooks,
which have several pages/chapters, all with the same I/O structure.

A group of tasks is called a **job**.

The job assets, that are, the audio and text files
of the tasks children of the job,
can be provided by the user as a **container**,
for example a ZIP file or an uncompressed directory
with a certain structure.

Similarly to a task, a job has **properties**
that must be specified by the user, most importantly:

#. the language of the text
#. the structure of the input container, that is,
   where to look for the task text/audio files
#. the format of the sync map files to be output,
   with any additional parameters (same as for a task)
#. the structure of the output container, that is,
   where to place the sync map files in the directory hierarchy
   of the output container

When executing a job as a container,
these configuration properties can be either 
be given as a **configuration string** (as above),
or written into a **TXT config file ``config.txt``**
or an **XML config file ``config.xml``**,
placed inside the job container.

Ideally, the config file should be in the **root directory
of the container**, however ``aeneas`` will look inside subdirectories
until it finds a ``config.txt`` or ``config.xml`` file,
and consider that directory as the **logical root** of the container.

If both ``config.txt`` and ``config.xml`` are present
in a container, ``aeneas`` will discard the TXT config file
and it will consider the **XML config file** only.

The list of **available parameter keys** can be found
in :class:`aeneas.globalconstants`.
It is highly recommended to look at the provided **examples**
in ``aeneas/tests/res/example_jobs``.

TXT Config File (``config.txt``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For example, a ZIP container with the following files::

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

where the ``config.txt`` config file reads::

    is_hierarchy_type=flat
    is_hierarchy_prefix=OEBPS/Resources/
    is_text_file_relative_path=.
    is_text_file_name_regex=.*\.txt
    is_text_type=parsed
    is_audio_file_relative_path=.
    is_audio_file_name_regex=.*\.mp3

    os_job_file_name=output_example1
    os_job_file_container=zip
    os_job_file_hierarchy_type=flat
    os_job_file_hierarchy_prefix=OEBPS/Resources/
    os_task_file_name=$PREFIX.smil
    os_task_file_format=smil
    os_task_file_smil_page_ref=$PREFIX.xhtml
    os_task_file_smil_audio_ref=$PREFIX.mp3

    job_language=en
    job_description=Example 1 (flat hierarchy, parsed text files)

will generate three tasks (``sonnet001``, ``sonnet002`` and ``sonnet003``),
output a SMIL file for each of them,
finally compress them in a ZIP file with the following structure::

    .
    └── OEBPS
        └── Resources
            ├── sonnet001.smil
            ├── sonnet002.smil
            └── sonnet003.smil

Note that the paths in ``config.txt`` are relative to
(the directory containing) the ``config.txt`` file.

XML Config File (``config.xml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
While ``config.txt`` is concise and easy to write,
it constraints all the tasks of the job to share the same
execution settings (language, output format, and so on).

If you need to specify **different values** for execution parameters
of **different tasks**, you must use an **XML config file**,
named ``config.xml``, with `this DTD <_static/configuration.dtd>`_ .

The following ``config.xml`` is equivalent to the example above::

    <?xml version = "1.0" encoding="UTF-8" standalone="no"?>
    <job>
        <job_language>en</job_language>
        <job_description>Example 4 (XML, flat hierarchy, parsed text files)</job_description>
        <tasks>
            <task>
                <task_language>en</task_language>
                <task_description>Sonnet 1</task_description>
                <task_custom_id>sonnet001</task_custom_id>
                <is_text_file>OEBPS/Resources/sonnet001.txt</is_text_file>
                <is_text_type>parsed</is_text_type>
                <is_audio_file>OEBPS/Resources/sonnet001.mp3</is_audio_file>
                <os_task_file_name>sonnet001.smil</os_task_file_name>
                <os_task_file_format>smil</os_task_file_format>
                <os_task_file_smil_page_ref>sonnet001.xhtml</os_task_file_smil_page_ref>
                <os_task_file_smil_audio_ref>sonnet001.mp3</os_task_file_smil_audio_ref>
            </task>
            <task>
                <task_language>en</task_language>
                <task_description>Sonnet 2</task_description>
                <task_custom_id>sonnet002</task_custom_id>
                <is_text_file>OEBPS/Resources/sonnet002.txt</is_text_file>
                <is_text_type>parsed</is_text_type>
                <is_audio_file>OEBPS/Resources/sonnet002.mp3</is_audio_file>
                <os_task_file_name>sonnet002.smil</os_task_file_name>
                <os_task_file_format>smil</os_task_file_format>
                <os_task_file_smil_page_ref>sonnet002.xhtml</os_task_file_smil_page_ref>
                <os_task_file_smil_audio_ref>sonnet002.mp3</os_task_file_smil_audio_ref>
            </task>
            <task>
                <task_language>en</task_language>
                <task_description>Sonnet 3</task_description>
                <task_custom_id>sonnet003</task_custom_id>
                <is_text_file>OEBPS/Resources/sonnet003.txt</is_text_file>
                <is_text_type>parsed</is_text_type>
                <is_audio_file>OEBPS/Resources/sonnet003.mp3</is_audio_file>
                <os_task_file_name>sonnet003.smil</os_task_file_name>
                <os_task_file_format>smil</os_task_file_format>
                <os_task_file_smil_page_ref>sonnet003.xhtml</os_task_file_smil_page_ref>
                <os_task_file_smil_audio_ref>sonnet003.mp3</os_task_file_smil_audio_ref>
            </task>
        </tasks>
        <os_job_file_name>output_example4</os_job_file_name>
        <os_job_file_container>zip</os_job_file_container>
        <os_job_file_hierarchy_type>flat</os_job_file_hierarchy_type>
        <os_job_file_hierarchy_prefix>OEBPS/Resources/</os_job_file_hierarchy_prefix>
    </job>

Package ``aeneas.tools``
------------------------

This package contains the two main tools:

#. ``aeneas.tools.execute_task``
#. ``aeneas.tools.execute_job``

The ``aeneas.tools`` package also contains other programs
useful for debugging:

#. ``aeneas.tools.convert_syncmap``: convert a sync map from a format to another
#. ``aeneas.tools.download``: download a file from a Web resource (currently, audio from a YouTube video)
#. ``aeneas.tools.espeak_wrapper``: a wrapper around ``espeak``
#. ``aeneas.tools.extract_mfcc``: extract MFCCs from a monoaural wav file
#. ``aeneas.tools.ffmpeg_wrapper``: a wrapper around ``ffmpeg``
#. ``aeneas.tools.ffprobe_wrapper``: a wrapper around ``ffprobe``
#. ``aeneas.tools.read_audio``: read the properties of an audio file
#. ``aeneas.tools.read_text``: read a text file and show the extracted text fragments
#. ``aeneas.tools.run_sd``: read an audio file and the corresponding text file and detect the audio head/tail
#. ``aeneas.tools.run_vad``: read an audio file and compute speech/nonspeech time intervals
#. ``aeneas.tools.synthesize_text``: synthesize several text fragments read from file into a single wav file
#. ``aeneas.tools.validate``: validate a job container or configuration strings/files

Run each program without arguments
to get its help manual and usage examples.

Executing a Task with ``aeneas.tools.execute_task``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This program **executes a task** and outputs the corresponding **sync map file**::

    $ python -m aeneas.tools.execute_task audio.mp3 text.txt config_string output.smil

Run the above command without arguments to get its help manual
with working examples::

    $ python -m aeneas.tools.execute_task

    Usage:
      $ python -m aeneas.tools.execute_task path/to/audio.mp3 path/to/text.txt config_string /path/to/output/file.smil

    Example 1 (input: parsed text, output: SRT)
      $ DIR="aeneas/tests/res/example_jobs/example1/OEBPS/Resources"
      $ CONFIG_STRING="task_language=en|os_task_file_format=srt|is_text_type=parsed"
      $ python -m aeneas.tools.execute_task $DIR/sonnet001.mp3 $DIR/sonnet001.txt "$CONFIG_STRING" /tmp/sonnet001.srt

    Example 2 (input: unparsed text, output: SMIL)
      $ DIR="aeneas/tests/res/container/job/assets"
      $ CONFIG_STRING="task_language=en|os_task_file_format=smil|os_task_file_smil_audio_ref=p001.mp3|os_task_file_smil_page_ref=p001.xhtml|is_text_type=unparsed|is_text_unparsed_id_regex=f[0-9]+|is_text_unparsed_id_sort=numeric"
      $ python -m aeneas.tools.execute_task $DIR/p001.mp3 $DIR/p001.xhtml "$CONFIG_STRING" /tmp/p001.smil


Executing a Job with ``aeneas.tools.execute_job``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This program **executes a job** and creates the corresponding **output container**,
which contains the sync map files for all the tasks in the job::

    $ python -m aeneas.tools.execute_job job.zip path/to/output/dir/

You can also create a **directory** with all your task assets,
instead of creating a compressed archive::

    $ python -m aeneas.tools.execute_job /path/to/dir/job path/to/output/dir/

Again, run the above command without arguments to get its help manual::
    
    $ python -m aeneas.tools.execute_job
    
    Usage:
      $ python -m aeneas.tools.execute_job /path/to/container [config_string] /path/to/output/dir

    Example:
      $ python -m aeneas.tools.execute_job aeneas/tests/res/container/job.zip /tmp/


Package ``aeneas.tests``
------------------------

This package contains the **unit testing** files for ``aeneas``.

Resources needed to run the tests,
for example audio and text files,
are located in the ``aeneas/tests/res/`` subdirectory.


Package ``aeneas``
------------------

The ``aeneas`` package contains the following modules:

.. toctree::
    :maxdepth: 3

    adjustboundaryalgorithm
    analyzecontainer
    audiofile
    container
    downloader
    dtw
    espeakwrapper
    executejob
    executetask
    ffmpegwrapper
    ffprobewrapper
    hierarchytype
    idsortingalgorithm
    job
    language
    logger
    sd
    syncmap
    synthesizer
    task
    textfile
    vad
    validator
    globalconstants
    globalfunctions



Changelog
---------

See :doc:`changelog`



Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
