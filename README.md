# aeneas 

**aeneas** is a Python library and a set of tools to automagically synchronize audio and text.

* Version: 1.0.4
* Date: 2015-08-09
* Developed by: [ReadBeyond](http://www.readbeyond.it/)
* Lead Developer: [Alberto Pettarin](http://www.albertopettarin.it/)
* License: the GNU Affero General Public License Version 3 (AGPL v3)
* Contact: [aeneas@readbeyond.it](mailto:aeneas@readbeyond.it)

## Goal

**aeneas** automatically generates a **synchronization map**
between a list of text fragments
and an audio file containing the narration of the (same) text.

For example, given [this text file](aeneas/tests/res/container/job/assets/p001.xhtml)
and [this audio file](aeneas/tests/res/container/job/assets/p001.mp3),
**aeneas** computes the following map:

```
[00:00:00.000, 00:00:02.680] <=> 1                                                      
[00:00:02.680, 00:00:05.480] <=> From fairest creatures we desire increase,            
[00:00:05.480, 00:00:08.640] <=> That thereby beauty's rose might never die,           
[00:00:08.640, 00:00:11.960] <=> But as the riper should by time decease,              
[00:00:11.960, 00:00:15.279] <=> His tender heir might bear his memory:                
[00:00:15.279, 00:00:18.519] <=> But thou contracted to thine own bright eyes,         
[00:00:18.519, 00:00:22.760] <=> Feed'st thy light's flame with self-substantial fuel, 
[00:00:22.760, 00:00:25.719] <=> Making a famine where abundance lies,                 
[00:00:25.719, 00:00:31.239] <=> Thy self thy foe, to thy sweet self too cruel:        
[00:00:31.239, 00:00:34.280] <=> Thou that art now the world's fresh ornament,         
[00:00:34.280, 00:00:36.960] <=> And only herald to the gaudy spring,                  
[00:00:36.960, 00:00:40.640] <=> Within thine own bud buriest thy content,             
[00:00:40.640, 00:00:43.600] <=> And tender churl mak'st waste in niggarding:          
[00:00:43.600, 00:00:48.000] <=> Pity the world, or else this glutton be,              
[00:00:48.000, 00:00:53.280] <=> To eat the world's due, by the grave and thee.  
```

Moreover, the map can be output in several formats: SMIL for EPUB 3,
SRT/TTML/VTT for closed captioning, JS for Web usage,
or raw CSV/SSV/TSV/TXT/XML for further processing.


## System Requirements, Supported Platforms and Installation

### System Requirements

1. 2 GB RAM (4 GB recommended), 2 GHz CPU (3 GHz 64bit recommended)
2. `ffmpeg` and `ffprobe` executables installed (`apt-get install ffmpeg*` from `deb-multimedia`)
3. `espeak` executable installed (`apt-get install espeak*`)
4. Python 2.7.x
5. Python optional modules ``BeautifulSoup``, ``lxml``, ``numpy``, and ``scikits.audiolab`` (``pip install ...``)

Depending on the format(s) of audio files you work with,
you might need to install additional audio codecs for `ffmpeg`.
Similarly, you might need to install additional voices
for `espeak`, depending on the language(s) you work on.
(Installing _all_ the codecs and _all_ the voices available
in the Debian repository might be a good idea.)

If installing the above dependencies prove difficult on your OS,
consider using the [Vagrant box](http://www.vagrantup.com)
created by [aeneas-vagrant](https://github.com/readbeyond/aeneas-vagrant).

### Supported Platforms

**aeneas** has been developed and tested on **Debian 64bit**,
which is the **only supported OS** at the moment.
Other Linux distributions should be good too.

However, it should work on Mac OS X and Windows as well,
once you make sure `ffmpeg`, `ffprobe` and `espeak`
are properly installed and
callable by the `subprocess` Python module.
A way to ensure the latter consists
in adding the three executables to your `$PATH`.
Alternatively, you can use VirtualBox
to run **aeneas** inside a virtualized Debian image,
for example using [aeneas-vagrant](https://github.com/readbeyond/aeneas-vagrant).

### Installation

```bash
$ git clone https://github.com/readbeyond/aeneas.git
$ cd aeneas
$ pip install -r requirements.txt
$ python check_dependencies.py
```

If the last command prints a success message,
you have all the required dependencies installed
and you can confidently run **aeneas** in production.

If you get an error, try running the
[provided `install_dependencies.sh` script](install_dependencies.sh)

```bash
$ sudo bash install_dependencies.sh
```

and then try running `check_dependencies.py` again.

Alternatively, consider using the [Vagrant box](http://www.vagrantup.com)
created by [aeneas-vagrant](https://github.com/readbeyond/aeneas-vagrant).

## Usage

1. Clone this GitHub repo:

    ```bash
    $ git clone https://github.com/readbeyond/aeneas.git
    $ cd aeneas
    ```

2. To compute a SMIL synchronization map `map.smil` for a pair
(`audio.mp3`, `text.txt`), you can run:

    ```bash
    $ python -m aeneas.tools.execute_task audio.mp3 text.txt config_string map.smil 
    ```

    `config_string` is string containing all the
    parameters to parse `text.txt` correctly and to
    format `map.smil` as desired.
    See the [documentation](http://www.readbeyond.it/aeneas/docs/) for details.

3. If you have several tasks to run,
you can create a job container and a configuration file,
and run them all at once:

    ```bash
    $ python -m aeneas.tools.execute_job job.zip /tmp/
    ```
    
    File `job.zip` should contain a `config.txt` or `config.xml`
    configuration file, providing **aeneas**
    with all the information needed to parse the input assets
    and format the output sync map files.
    See the [documentation](http://www.readbeyond.it/aeneas/docs/) for details.

You might want to run the above modules without arguments
to get their manual:

```bash
$ python -m aeneas.tools.execute_task
$ python -m aeneas.tools.execute_job
```

See the [documentation](http://www.readbeyond.it/aeneas/docs/)
for an introduction to the concepts of `task` and  `job`,
and for a list of the available options.


## Documentation

Online: [http://www.readbeyond.it/aeneas/docs/](http://www.readbeyond.it/aeneas/docs/)

Generated from the source (requires `sphinx`):

```bash
$ git clone https://github.com/readbeyond/aeneas.git
$ cd aeneas/docs
$ make html
```

Tutorial: [A Practical Introduction To The aeneas Package](http://www.albertopettarin.it/blog/2015/05/21/a-practical-introduction-to-the-aeneas-package.html)

Mailing list: [https://groups.google.com/d/forum/aeneas-forced-alignment](https://groups.google.com/d/forum/aeneas-forced-alignment)


## Supported Features

* Input text files in plain, parsed, subtitles, or unparsed format
* Text extraction from XML (e.g., XHTML) files using `id` and `class` attributes
* Arbitrary text fragment granularity (single word, subphrase, phrase, paragraph, etc.)
* Input audio file formats: all those supported by `ffmpeg`
* Batch processing
* Output sync map formats: CSV, JS, SMIL, TSV, TTML, TXT, VTT, XML
* Supported (= tested) languages: BG, CA, CY, DA, DE, EL, EN, ES, ET, FI, FR, GA, GRC, HR, HU, IS, IT, LA, LT, LV, NL, NO, RO, RU, PL, PT, SK, SR, SV, TR, UK
* Robust against misspelled/mispronounced words, local rearrangements of words, background noise/sporadic spikes
* Code suitable for a Web app deployment (e.g., on-demand AWS instances)
* Adjustable splitting times, including a max character/second constraint for CC applications


## Limitations and Missing Features 

* Audio should match the text: large portions of spurious text or audio might produce a wrong sync map
* Audio is assumed to be spoken: not suitable/YMMV for song captioning
* Offline (i.e., not real time/near real time) approach
* DTW computation is memory hungry
* No protection against memory trashing


## TODO List

* Improving the speed of the code, especially when Sakoe-Chiba kicks in
* Improving robustness against music in background
* Isolate non-speech intervals (music, prolonged silence)
* Automated text fragmentation based on audio analysis
* Auto-tuning DTW parameters
* Reporting the alignment score
* Improving (removing?) dependency from `espeak`, `ffmpeg`, `ffprobe` executables
* Multilevel sync map granularity (e.g., multilevel SMIL output)
* Supporting input text encodings other than UTF-8
* Adding (testing) more languages
* Better documentation
* Testing other approaches, like HMM
* Publishing the package on PyPI


## How Does This Thing Work?

### One Word Explanation

Math.

### One Sentence Explanation (Layman Edition)

A good deal of math and computer science,
a handful of software engineering and
some optimization tricks.

### One Sentence Explanation (Pro Edition)

Using the Sakoe-Chiba Band Dynamic Time Warping (DTW) algorithm
to align the Mel-frequency cepstral coefficients (MFCCs)
representation of the given (real) audio wave and
the audio wave obtained by synthesizing the text fragments
with a TTS engine, eventually mapping
the computed alignment back onto the (real) time domain.

### Extended Explanation

To be written. Eventually. Some day.


## License

**aeneas** is released under the terms of the
GNU Affero General Public License Version 3.
See the [LICENSE](LICENSE) file for details.

The code for computing the MFCCs
[`aeneas/mfcc.py`](aeneas/mfcc.py)
is a verbatim copy from the
[CMU Sphinx III project](http://cmusphinx.sourceforge.net/).

Audio files contained in the unit tests `aeneas/tests/res/` directory
are adapted from recordings produced by
the [LibriVox Project](http://www.librivox.org)
and they are in the public domain.

Text files contained in the unit tests `aeneas/tests/res/` directory
are adapted from files produced by
the [Project Gutenberg](http://www.gutenberg.org)
and they are in the public domain.

No copy rights were harmed in the making of this project.


## Supporting and Contributing

### Sponsors 

* **July 2015**: [Michele Gianella](https://plus.google.com/+michelegianella/about) generously supported the development of the boundary adjustment code

### Supporting

Would you like supporting the development of **aeneas**?

We are open to accept sponsorships to

* fix bugs,
* add new features,
* improve the quality and the performance of the code,
* port the code to other languages/platforms,
* support of third party installations, and
* improve the documentation.

Feel free to [get in touch](mailto:aeneas@readbeyond.it).

### Contributing

If you are able to contribute code directly,
that's great!
Feel free to open a pull request,
we will be glad to have a look at it.

Please make your code consistent with
the existing code base style
(see the [Google Python Style Guide](https://google-styleguide.googlecode.com/svn/trunk/pyguide.html)),
and test your contributed code
against the unit tests
before opening the pull request.
Ideally, add some unit tests on the code written by you.

**Please note that, by opening a pull request,
you automatically agree to apply
the AGPL v3 license
to the code you contribute.**

If you think you found a bug,
please use the
[GitHub issue tracker](https://github.com/readbeyond/aeneas/issues)
to file a bug report.


## Development History

**Early 2012**: Nicola Montecchio and Alberto Pettarin
co-developed an initial experimental package
to align audio and text, intended to be run locally
to compute Media Overlay (SMIL) files for
EPUB 3 Audio-eBooks

**Late 2012-June 2013**: Alberto Pettarin
continued engineering and tuning the alignment tool,
making it faster and memory efficient,
writing the I/O functions for batch processing
of multiple audio/text pairs,
and started producing the first EPUB 3 Audio-eBooks
with Media Overlays (SMIL files) computed automatically
by this package

**July 2013**: incorporation of ReadBeyond Srl

**July 2013-March 2014**: development of ReadBeyond Sync,
a SaaS version of this package,
exposing the alignment function via APIs
and a Web application

**March 2014**: launch of ReadBeyond Sync beta

**April 2015**: ReadBeyond Sync beta ended

**May 2015**: release of this package on GitHub


## Acknowledgments

Many thanks to **Nicola Montecchio**,
who suggested using MFCCs and DTW,
and co-developed the first experimental code
for aligning audio and text.

**Paolo Bertasi**, who developed the
APIs and Web application for ReadBeyond Sync,
helped shaping the structure of this package
for its asynchronous usage.


