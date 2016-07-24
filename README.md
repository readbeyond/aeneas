# aeneas 

**aeneas** is a Python/C library and a set of tools to automagically synchronize audio and text (aka forced alignment).

* Version: 1.5.1.0
* Date: 2016-07-25
* Developed by: [ReadBeyond](http://www.readbeyond.it/)
* Lead Developer: [Alberto Pettarin](http://www.albertopettarin.it/)
* License: the GNU Affero General Public License Version 3 (AGPL v3)
* Contact: [aeneas@readbeyond.it](mailto:aeneas@readbeyond.it)
* Quick Links: [Home](http://www.readbeyond.it/aeneas/) - [GitHub](https://github.com/readbeyond/aeneas/) - [PyPI](https://pypi.python.org/pypi/aeneas/) - [Docs](http://www.readbeyond.it/aeneas/docs/) - [Tutorial](http://www.readbeyond.it/aeneas/docs/clitutorial.html) - [Benchmark](https://readbeyond.github.io/aeneas-benchmark/) - [Mailing List](https://groups.google.com/d/forum/aeneas-forced-alignment) - [Web App](http://aeneasweb.org)

 
## Goal

**aeneas** automatically generates a **synchronization map**
between a list of text fragments
and an audio file containing the narration of the text.
In computer science this task is known as
(automatically computing a) **forced alignment**.

For example, given
[this text file](https://raw.githubusercontent.com/readbeyond/aeneas/master/aeneas/tests/res/container/job/assets/p001.xhtml)
and
[this audio file](https://raw.githubusercontent.com/readbeyond/aeneas/master/aeneas/tests/res/container/job/assets/p001.mp3),
**aeneas** determines, for each fragment, the corresponding time interval in the audio file:

```
1                                                     => [00:00:00.000, 00:00:02.640]
From fairest creatures we desire increase,            => [00:00:02.640, 00:00:05.880]
That thereby beauty's rose might never die,           => [00:00:05.880, 00:00:09.240]
But as the riper should by time decease,              => [00:00:09.240, 00:00:11.920]
His tender heir might bear his memory:                => [00:00:11.920, 00:00:15.280]
But thou contracted to thine own bright eyes,         => [00:00:15.280, 00:00:18.800]
Feed'st thy light's flame with self-substantial fuel, => [00:00:18.800, 00:00:22.760]
Making a famine where abundance lies,                 => [00:00:22.760, 00:00:25.680]
Thy self thy foe, to thy sweet self too cruel:        => [00:00:25.680, 00:00:31.240]
Thou that art now the world's fresh ornament,         => [00:00:31.240, 00:00:34.400]
And only herald to the gaudy spring,                  => [00:00:34.400, 00:00:36.920]
Within thine own bud buriest thy content,             => [00:00:36.920, 00:00:40.640]
And tender churl mak'st waste in niggarding:          => [00:00:40.640, 00:00:43.640]
Pity the world, or else this glutton be,              => [00:00:43.640, 00:00:48.080]
To eat the world's due, by the grave and thee.        => [00:00:48.080, 00:00:53.240]
```

![Waveform with aligned labels, detail](wiki/align.png)

This synchronization map can be output to file in several formats:
EAF for research purposes,
SMIL for EPUB 3,
SBV/SRT/SUB/TTML/VTT for closed captioning,
JSON for Web usage,
or raw AUD/CSV/SSV/TSV/TXT/XML for further processing.


## System Requirements, Supported Platforms and Installation

### System Requirements

1. a reasonably recent machine (recommended 4 GB RAM, 2 GHz 64bit CPU)
2. [Python](https://python.org/) 2.7 (Linux, OS X, Windows) or 3.4 or later (Linux, OS X)
3. [FFmpeg](https://www.ffmpeg.org/)
4. [eSpeak](http://espeak.sourceforge.net/)
5. Python modules `BeautifulSoup4`, `lxml`, and `numpy`
6. Python C headers to compile the Python C extensions (optional but strongly recommended)
7. A shell supporting UTF-8 (optional but strongly recommended)

### Supported Platforms

**aeneas** has been developed and tested on **Debian 64bit**,
which is the **only supported OS** at the moment.
Nevertheless, **aeneas** has been confirmed to work on
other Linux distributions, OS X, and Windows.
See the
[PLATFORMS file](https://github.com/readbeyond/aeneas/blob/master/wiki/PLATFORMS.md)
for the details.

If installing **aeneas** natively on your OS proves difficult,
you are strongly encouraged to use
[aeneas-vagrant](https://github.com/readbeyond/aeneas-vagrant),
which provides **aeneas** inside a virtualized Debian image
running under
[VirtualBox](https://www.virtualbox.org/)
and
[Vagrant](http://www.vagrantup.com/),
which can be installed on any modern OS (Linux, Mac OS X, Windows).

### Installation

All-in-one installers are available for Mac OS X and Windows,
and a Bash script for deb-based Linux distributions (Debian, Ubuntu)
is provided in this repository.
It is also possible to download a VirtualBox+Vagrant virtual machine.
Please see the
[INSTALL file](https://github.com/readbeyond/aeneas/blob/master/wiki/INSTALL.md)
for detailed, step-by-step installation procedures for different operating systems.

The generic OS-independent procedure is simple:

1. Install
   [Python](https://python.org/) (2.7.x preferred),
   [FFmpeg](https://www.ffmpeg.org/), and
   [eSpeak](http://espeak.sourceforge.net/)

2. Make sure the following executables can be called from your shell:
   `espeak`, `ffmpeg`, `ffprobe`, `pip`, and `python`

3. First install `numpy` with `pip` and then `aeneas`:
    
    ```bash
    pip install numpy
    pip install aeneas
    ```

4. To **check** whether you installed **aeneas** correctly, run:

   ```bash
    python -m aeneas.diagnostics
    ```


## Usage

1. Run without arguments to get the **usage message**:

    ```bash
    python -m aeneas.tools.execute_task
    python -m aeneas.tools.execute_job
    ```

    You can also get a list of **live examples**
    that you can immediately run on your machine
    thanks to the included files:

    ```bash
    python -m aeneas.tools.execute_task --examples
    python -m aeneas.tools.execute_task --examples-all
    ```

2. To **compute a synchronization map** `map.json` for a pair
   (`audio.mp3`, `text.txt` in
   [plain](http://www.readbeyond.it/aeneas/docs/textfile.html#aeneas.textfile.TextFileFormat.PLAIN)
   text format), you can run:

    ```bash
    python -m aeneas.tools.execute_task \
        audio.mp3 \
        text.txt \
        "task_language=eng|os_task_file_format=json|is_text_type=plain" \
        map.json
    ```

   (The command has been split into lines with `\` for visual clarity;
   in production you can have the entire command on a single line
   and/or you can use shell variables.)

   To **compute a synchronization map** `map.smil` for a pair
   (`audio.mp3`,
   [page.xhtml](http://www.readbeyond.it/aeneas/docs/textfile.html#aeneas.textfile.TextFileFormat.UNPARSED)
   containing fragments marked by `id` attributes like `f001`),
   you can run:

    ```bash
    python -m aeneas.tools.execute_task \
        audio.mp3 \
        page.xhtml \
        "task_language=eng|os_task_file_format=smil|os_task_file_smil_audio_ref=audio.mp3|os_task_file_smil_page_ref=page.xhtml|is_text_type=unparsed|is_text_unparsed_id_regex=f[0-9]+|is_text_unparsed_id_sort=numeric" \
        map.smil
    ```

   As you can see, the third argument (the _configuration string_)
   specifies the parameters controlling the I/O formats
   and the processing options for the task.
   Consult the
   [documentation](http://www.readbeyond.it/aeneas/docs/)
   for details.

3. If you have several tasks to process,
   you can create a **job container**
   to batch process them:

    ```bash
    python -m aeneas.tools.execute_job job.zip output_directory
    ```
    
   File `job.zip` should contain a `config.txt` or `config.xml`
   configuration file, providing **aeneas**
   with all the information needed to parse the input assets
   and format the output sync map files.
   Consult the
   [documentation](http://www.readbeyond.it/aeneas/docs/)
   for details.

The
[documentation](http://www.readbeyond.it/aeneas/docs/)
contains a highly suggested
[tutorial](http://www.readbeyond.it/aeneas/docs/clitutorial.html)
which explains how to use the built-in command line tools.


## Documentation and Support

* Documentation:
  [http://www.readbeyond.it/aeneas/docs/](http://www.readbeyond.it/aeneas/docs/)
* Command line tools tutorial:
  [http://www.readbeyond.it/aeneas/docs/clitutorial.html](http://www.readbeyond.it/aeneas/docs/clitutorial.html)
* Library tutorial:
  [http://www.readbeyond.it/aeneas/docs/libtutorial.html](http://www.readbeyond.it/aeneas/docs/libtutorial.html)
* Old, verbose tutorial:
  [A Practical Introduction To The aeneas Package](http://www.albertopettarin.it/blog/2015/05/21/a-practical-introduction-to-the-aeneas-package.html)
* Mailing list:
  [https://groups.google.com/d/forum/aeneas-forced-alignment](https://groups.google.com/d/forum/aeneas-forced-alignment)
* Changelog:
  [http://www.readbeyond.it/aeneas/docs/changelog.html](http://www.readbeyond.it/aeneas/docs/changelog.html)
* High level description of how aeneas works:
  [HOWITWORKS](https://github.com/readbeyond/aeneas/blob/master/wiki/HOWITWORKS.md)
* Development history:
  [HISTORY](https://github.com/readbeyond/aeneas/blob/master/wiki/HISTORY.md)
* Benchmark suite:
  [https://readbeyond.github.io/aeneas-benchmark/](https://readbeyond.github.io/aeneas-benchmark/)


## Supported Features

* Input text files in `parsed`, `plain`, `subtitles`, or `unparsed` (XML) format
* Multilevel input text files in `mplain` and `munparsed` (XML) format
* Text extraction from XML (e.g., XHTML) files using `id` and `class` attributes
* Arbitrary text fragment granularity (single word, subphrase, phrase, paragraph, etc.)
* Input audio file formats: all those readable by `ffmpeg`
* Output sync map formats: AUD, CSV, EAF, JSON, SMIL, SRT, SSV, SUB, TSV, TTML, TXT, VTT, XML
* Confirmed working on languages: ARA, BUL, CAT, CYM, CES, DAN, DEU, ELL, ENG, EPO, EST, FAS, FIN, FRA, GLE, GRC, HRV, HUN, ISL, ITA, JPN, LAT, LAV, LIT, NLD, NOR, RON, RUS, POL, POR, SLK, SPA, SRP, SWA, SWE, TUR, UKR
* MFCC and DTW computed via Python C extensions to reduce the processing time
* Several built-in TTS engine wrappers: eSpeak (default, FLOSS), Festival (FLOSS), Nuance TTS API (commercial)
* Default TTS (eSpeak) called via a Python C extension for fast audio synthesis
* A custom, user-provided TTS engine Python wrapper can be used instead of the built-in ones (included example for speect)
* Batch processing of multiple audio/text pairs
* Download audio from a YouTube video
* In multilevel mode, recursive alignment from paragraph to sentence to word level
* Robust against misspelled/mispronounced words, local rearrangements of words, background noise/sporadic spikes
* Adjustable splitting times, including a max character/second constraint for CC applications
* Automated detection of audio head/tail
* Output an HTML file for fine tuning the sync map manually (`finetuneas` project)
* Execution parameters tunable at runtime
* Code suitable for Web app deployment (e.g., on-demand cloud computing)
* Extensive test suite including 898 unit/integration/performance tests, that run and must pass before each release


## Limitations and Missing Features 

* Audio should match the text: large portions of spurious text or audio might produce a wrong sync map
* Audio is assumed to be spoken: not suitable for song captioning, YMMV for CC applications
* No protection against memory trashing if you feed extremely long audio files (>1.5h per single audio file)
* [Open issues](https://github.com/readbeyond/aeneas/issues)


## License

**aeneas** is released under the terms of the
GNU Affero General Public License Version 3.
See the
[LICENSE file](https://github.com/readbeyond/aeneas/blob/master/LICENSE) for details.

Licenses for third party code and files included in **aeneas**
can be found in the
[licenses](https://github.com/readbeyond/aeneas/blob/master/licenses/README.md) directory.

No copy rights were harmed in the making of this project.


## Supporting and Contributing

### Sponsors 

* **July 2015**: [Michele Gianella](https://plus.google.com/+michelegianella/about) generously supported the development of the boundary adjustment code (v1.0.4)

* **August 2015**: [Michele Gianella](https://plus.google.com/+michelegianella/about) partially sponsored the port of the MFCC/DTW code to C (v1.1.0)

* **September 2015**: friends in West Africa partially sponsored the development of the head/tail detection code (v1.2.0)

* **October 2015**: an anonymous donation sponsored the development of the "YouTube downloader" option (v1.3.0)

* **April 2016**: the Fruch Foundation kindly sponsored the development and documentation of v1.5.0

### Supporting

Would you like supporting the development of **aeneas**?

I accept sponsorships to

* fix bugs,
* add new features,
* improve the quality and the performance of the code,
* port the code to other languages/platforms,
* support of third party installations, and
* improve the documentation.

Feel free to
[get in touch](mailto:aeneas@readbeyond.it).

### Contributing

If you think you found a bug,
please use the
[GitHub issue tracker](https://github.com/readbeyond/aeneas/issues)
to file a bug report.

If you are able to contribute code directly, that is awesome!
I will be glad to merge it!
Just a few rules, to make life easier for both you and me:

1. Please do not work on the `master` branch.
   Instead, create a new branch on your GitHub repo
   by cheking out the `devel` branch.
   Open a pull request from your branch on your repo
   to the `devel` branch on this GitHub repo.

2. Please make your code consistent with
   the existing code base style
   (see the
   [Google Python Style Guide](https://google-styleguide.googlecode.com/svn/trunk/pyguide.html)
   ), and test your contributed code
   against the unit tests
   before opening the pull request.

3. Ideally, add some unit tests for the code you are submitting,
   either adding them to the existing unit tests or creating a new file
   in `aeneas/tests/`.

4. **Please note that, by opening a pull request,
   you automatically agree to apply
   the AGPL v3 license
   to the code you contribute.**


## Acknowledgments

Many thanks to **Nicola Montecchio**,
who suggested using MFCCs and DTW,
and co-developed the first experimental code
for aligning audio and text.

**Paolo Bertasi**, who developed the
APIs and Web application for ReadBeyond Sync,
helped shaping the structure of this package
for its asynchronous usage.

**Chris Hubbard** prepared the files for
packaging aeneas as a Debian/Ubuntu `.deb`.

**Daniel Bair**, **Chris Hubbard**, and **Richard Margetts**
packaged the installers for Mac OS X and Windows.

**Firat Ozdemir** contributed the `finetuneas`
HTML/JS code for fine tuning sync maps in the browser.

All the mighty
[GitHub contributors](https://github.com/readbeyond/aeneas/graphs/contributors),
and the members of the
[Google Group](https://groups.google.com/d/forum/aeneas-forced-alignment).



