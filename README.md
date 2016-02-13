# aeneas 

**aeneas** is a Python/C library and a set of tools to automagically synchronize audio and text (aka forced alignment).

* Version: 1.4.1
* Date: 2016-02-13
* Developed by: [ReadBeyond](http://www.readbeyond.it/)
* Lead Developer: [Alberto Pettarin](http://www.albertopettarin.it/)
* License: the GNU Affero General Public License Version 3 (AGPL v3)
* Contact: [aeneas@readbeyond.it](mailto:aeneas@readbeyond.it)
* Quick Links: [Home](http://www.readbeyond.it/aeneas/) - [GitHub](https://github.com/readbeyond/aeneas/) - [PyPI](https://pypi.python.org/pypi/aeneas/) - [API Docs](http://www.readbeyond.it/aeneas/docs/) - [Mailing List](https://groups.google.com/d/forum/aeneas-forced-alignment) - [Web App](http://aeneasweb.org)

 
## Goal

**aeneas** automatically generates a **synchronization map**
between a list of text fragments
and an audio file containing the narration of the text.
In computer science this task is known as
(automatically computing a) **forced alignment**.

For example, given [this text file](https://raw.githubusercontent.com/readbeyond/aeneas/master/aeneas/tests/res/container/job/assets/p001.xhtml)
and [this audio file](https://raw.githubusercontent.com/readbeyond/aeneas/master/aeneas/tests/res/container/job/assets/p001.mp3),
**aeneas** determines, for each fragment, the corresponding time interval in the audio file:

```
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
```

This synchronization map can be output to file in several formats:
SMIL for EPUB 3, SBV/SRT/SUB/TTML/VTT for closed captioning,
JSON/RBSE for Web usage,
or raw CSV/SSV/TSV/TXT/XML for further processing.


## System Requirements, Supported Platforms and Installation

### System Requirements

1. a reasonably recent machine (recommended 4 GB RAM, 2 GHz 64bit CPU)
2. [Python](https://python.org/) 2.7 (Linux, OS X, Windows) or 3.4 or later (Linux, OS X)
3. [FFmpeg](https://www.ffmpeg.org/)
4. [eSpeak](http://espeak.sourceforge.net/)
5. Python modules `BeautifulSoup4`, `lxml`, and `numpy`
6. Python C headers to compile the Python C extensions (Optional but strongly recommended)
7. A shell supporting UTF-8 (Optional but strongly recommended)
8. Python module `pafy` (Optional, only required if you want to download audio from YouTube)

### Supported Platforms

**aeneas** has been developed and tested on **Debian 64bit**,
which is the **only supported OS** at the moment.

However, **aeneas** has been confirmed to work on
other Linux distributions, OS X, and Windows.
See the [PLATFORMS file](https://github.com/readbeyond/aeneas/blob/master/wiki/PLATFORMS.md) for the details.

If installing **aeneas** natively on your OS proves difficult,
you are strongly encouraged to use
[aeneas-vagrant](https://github.com/readbeyond/aeneas-vagrant),
which provides **aeneas** inside a virtualized Debian image
running under [VirtualBox](https://www.virtualbox.org/)
and [Vagrant](http://www.vagrantup.com/), which can be installed
on any modern OS (Linux, Mac OS X, Windows).

### Installation

1. Install [Python](https://python.org/) (2.7.x preferred),
   [FFmpeg](https://www.ffmpeg.org/), and
   [eSpeak](http://espeak.sourceforge.net/)

2. Make sure the following executables can be called from your shell:
   `espeak`, `ffmpeg`, `ffprobe`, `pip`, and `python`

3. First install `numpy` with `pip` and then `aeneas`:
    
    ```bash
    pip install numpy
    pip install aeneas
    ```

See the [INSTALL file](https://github.com/readbeyond/aeneas/blob/master/wiki/INSTALL.md)
for detailed, step-by-step procedures for Linux, OS X, and Windows.


## Usage

1. To check that you installed `aeneas` correctly, run:

   ```bash
    python -m aeneas.diagnostics
    ```

2. Run `execute_task` or `execute_job`
   with `-h` (resp., `--help`) to get a short (resp., long) usage message:

    ```bash
    python -m aeneas.tools.execute_task -h
    python -m aeneas.tools.execute_job -h
    ```

    The above commands also print a list of live usage examples
    that you can immediately run on your machine,
    thanks to the included example files.

3. To compute a synchronization map `map.json` for a pair
   (`audio.mp3`, `text.txt` in [`plain`](http://www.readbeyond.it/aeneas/docs/textfile.html#aeneas.textfile.TextFileFormat.PLAIN) text format), you can run:

    ```bash
    python -m aeneas.tools.execute_task \
        audio.mp3 \
        text.txt \
        "task_language=en|os_task_file_format=json|is_text_type=plain" \
        map.json
    ```

   To compute a synchronization map `map.smil` for a pair
   (`audio.mp3`, [`page.xhtml`](http://www.readbeyond.it/aeneas/docs/textfile.html#aeneas.textfile.TextFileFormat.UNPARSED) containing fragments marked by `id` attributes like `f001`),
   you can run:

    ```bash
    python -m aeneas.tools.execute_task \
        audio.mp3 \
        page.xhtml \
        "task_language=en|os_task_file_format=smil|os_task_file_smil_audio_ref=audio.mp3|os_task_file_smil_page_ref=page.xhtml|is_text_type=unparsed|is_text_unparsed_id_regex=f[0-9]+|is_text_unparsed_id_sort=numeric" \
        map.smil
    ```

   The third parameter (the _configuration string_) can specify several other parameters/options.
   See the [documentation](http://www.readbeyond.it/aeneas/docs/) for details.

4. If you have several tasks to process,
   you can create a job container and a configuration file,
   to process them all at once:

    ```bash
    python -m aeneas.tools.execute_job job.zip output_directory
    ```
    
   File `job.zip` should contain a `config.txt` or `config.xml`
   configuration file, providing **aeneas**
   with all the information needed to parse the input assets
   and format the output sync map files.
   See the [documentation](http://www.readbeyond.it/aeneas/docs/) for details.

The [documentation](http://www.readbeyond.it/aeneas/docs/)
provides an introduction to the concepts of
[`task`](http://www.readbeyond.it/aeneas/docs/#tasks) and
[`job`](http://www.readbeyond.it/aeneas/docs/#job),
and it lists of all the options and tools available in the library.


## Documentation and Support

Documentation: [http://www.readbeyond.it/aeneas/docs/](http://www.readbeyond.it/aeneas/docs/)

High level description of how aeneas works: [HOWITWORKS](https://github.com/readbeyond/aeneas/blob/master/wiki/HOWITWORKS.md)

Tutorial: [A Practical Introduction To The aeneas Package](http://www.albertopettarin.it/blog/2015/05/21/a-practical-introduction-to-the-aeneas-package.html)

Mailing list: [https://groups.google.com/d/forum/aeneas-forced-alignment](https://groups.google.com/d/forum/aeneas-forced-alignment)

Changelog: [http://www.readbeyond.it/aeneas/docs/changelog.html](http://www.readbeyond.it/aeneas/docs/changelog.html)

Development history: [HISTORY](https://github.com/readbeyond/aeneas/blob/master/wiki/HISTORY.md)


## Supported Features

* Input text files in plain, parsed, subtitles, or unparsed format
* Text extraction from XML (e.g., XHTML) files using `id` and `class` attributes
* Arbitrary text fragment granularity (single word, subphrase, phrase, paragraph, etc.)
* Input audio file formats: all those supported by `ffmpeg`
* Possibility of downloading the audio file from a YouTube video
* Batch processing
* Output sync map formats: CSV, JSON, RBSE, SMIL, SSV, TSV, TTML, TXT, VTT, XML
* Tested languages: BG, CA, CY, CS, DA, DE, EL, EN, EO, ES, ET, FA, FI, FR, GA, GRC, HR, HU, IS, IT, LA, LT, LV, NL, NO, RO, RU, PL, PT, SK, SR, SV, SW, TR, UK
* Robust against misspelled/mispronounced words, local rearrangements of words, background noise/sporadic spikes
* Code suitable for a Web app deployment (e.g., on-demand AWS instances)
* Adjustable splitting times, including a max character/second constraint for CC applications
* Automated detection of audio head/tail
* MFCC and DTW computed via Python C extensions to reduce the processing time
* On Linux, `espeak` called via a Python C extension for faster audio synthesis
* Output an HTML file (from `finetuneas` project) for fine tuning the sync map manually
* Execution parameters tunable at runtime


## Limitations and Missing Features 

* Audio should match the text: large portions of spurious text or audio might produce a wrong sync map
* Audio is assumed to be spoken: not suitable/YMMV for song captioning
* No protection against memory trashing if you feed extremely long audio files
* On Mac OS X and Windows, audio synthesis might be slow if you have thousands of text fragments
* [Open issues](https://github.com/readbeyond/aeneas/issues)


## License

**aeneas** is released under the terms of the
GNU Affero General Public License Version 3.
See the [LICENSE file](https://github.com/readbeyond/aeneas/blob/master/LICENSE) for details.

Licenses for third party code and files included in **aeneas**
can be found in the [licenses/](https://github.com/readbeyond/aeneas/blob/master/licenses/README.md) directory.

No copy rights were harmed in the making of this project.


## Supporting and Contributing

### Sponsors 

* **July 2015**: [Michele Gianella](https://plus.google.com/+michelegianella/about) generously supported the development of the boundary adjustment code (v1.0.4)

* **August 2015**: [Michele Gianella](https://plus.google.com/+michelegianella/about) partially sponsored the port of the MFCC/DTW code to C (v1.1.0)

* **September 2015**: friends in West Africa partially sponsored the development of the head/tail detection code (v1.2.0)

* **October 2015**: an anonymous donation sponsored the development of the "YouTube downloader" option (v1.3.0)

### Supporting

Would you like supporting the development of **aeneas**?

I accept sponsorships to

* fix bugs,
* add new features,
* improve the quality and the performance of the code,
* port the code to other languages/platforms,
* support of third party installations, and
* improve the documentation.

Feel free to [get in touch](mailto:aeneas@readbeyond.it).

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

All the mighty [GitHub contributors](https://github.com/readbeyond/aeneas/graphs/contributors),
and the members of the [Google Group](https://groups.google.com/d/forum/aeneas-forced-alignment).



