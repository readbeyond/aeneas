Changelog
=========

v1.7.3 (2017-03-15)
-------------------

#. Fixed bug #168 and added a regression test for it
#. Added option ``-f, --full`` to ``aeneas.tools.read_audio`` and tests for it

v1.7.2 (2017-03-03)
-------------------

#. Added MacOS TTS Wrapper (courtesy of Chris Vaughn)
#. Removed dependency from ``pafy``, using ``youtube-dl`` directly (#159)
#. Added retry mechanism to ``Downloader``, including ``DOWNLOADER_SLEEP`` and ``DOWNLOADER_RETRY_ATTEMPTS`` in ``RuntimeConfiguration``
#. Fixed bug #160
#. Fixed a latent bug with arbitrary shifts in aba when using the ``task_adjust_boundary_no_zero`` option
#. Fixed a latent bug in AWS Polly and Nuance wrappers
#. Updated copyright strings with 2017
#. Updated ``INSTALL.md`` to brew install from Daniel Bair's tap instead of official brew repo since they removed the formula for aeneas (#165)

v1.7.1 (2016-12-20)
-------------------

#. Fix bug #151
#. Downgraded dependency on lxml to lxml>=3.6.0 to help packaging the Windows installer
#. Added aeneas version to log
#. Changed default voice for Festival TTS Wrapper to ``eng-USA`` to help people installing from source on Mac OS X

v1.7.0 (2016-12-07)
-------------------

#. Moved syncmap I/O functions in ``aeneas.syncmap`` subpackage
#. Renamed ``aeneas.timevalue`` into ``aeneas.exacttiming``
#. Renamed ``audio_duration`` into ``length`` in ``SyncMapFragment`` for consistency with ``TimeInterval``
#. Renamed ``os_task_file_no_zero`` (``PPN_OS_TASK_FILE_NO_ZERO``) to ``task_adjust_boundary_no_zero`` (``PPN_TASK_ADJUST_BOUNDARY_NO_ZERO``) in ``TaskConfiguration``
#. Renamed ``nuance_tts_api_sleep`` (``NUANCE_TTS_API_SLEEP``) to ``tts_api_sleep`` (``TTS_API_SLEEP``) in ``RuntimeConfiguration``
#. Renamed ``nuance_tts_api_retry_attempts`` (``NUANCE_TTS_API_RETRY_ATTEMPTS``) to ``tts_api_retry_attempts`` (``TTS_API_RETRY_ATTEMPTS``) in ``RuntimeConfiguration``
#. Renamed ``--rates`` to ``--rate`` in ``ExecuteTaskCLI``
#. Renamed ``--example-rates`` to ``--example-rate`` in ``ExecuteTaskCLI``
#. Changed ``config_string()`` to property in ``aeneas.configuration.Configuration``
#. Changed the default value of ``TASK_MAX_AUDIO_LENGTH`` to ``0`` (i.e., process audio file with arbitrary audio length)
#. Added two new output formats: ``TEXTGRID`` (Praat long TextGrid) and ``TEXTGRID_SHORT`` (Praat short TextGrid)
#. More robust and generic reading of SRT-like files, especially WebVTT
#. Fixed typos in ``SyncMapFormat`` docstrings
#. Added ``safety_checks`` parameter to ``RuntimeConfiguration`` that can be disabled to trade safety for speed (issue #117)
#. Added Makefile files to C/C++ extensions, replacing previous Bash scripts
#. Simplified ``ExecuteTask``, offloading some sub-tasks to ``SyncMap``, ``SyncMapFragmentList``, and ``AdjustBoundaryAlgorithm``
#. Simplified ``AdjustBoundaryAlgorithm``
#. Added method ``sync_map_vleaves`` in ``Task`` for quick access to sync map fragments (vleaves)
#. Added ``aeneas.exacttiming.TimeInterval`` class to represent time intervals and act upon them
#. Added tests for ``TimeInterval``
#. Added ``has_zero_length`` in ``SyncMapFragment``
#. Added comparison functions to ``SyncMapFragment``, based on ``TimeInterval``
#. Added ``aeneas.syncmap.fragmentlist.SyncMapFragmentList`` class to represent a list of sync map fragments, sorted and with positional/timing constraints
#. Added tests for ``SyncMapFragmentList``
#. Added ``clone()`` method to ``aeneas.configuration.Configuration``
#. Removed ``clone()`` method to ``aeneas.runtimeconfiguration.RuntimeConfiguration`` (now it inherits from ``Configuration``)
#. Added ``clone()`` method to ``aeneas.tree.Tree``
#. Added more tests for ``Tree``
#. Added ``PPN_TASK_ADJUST_BOUNDARY_NONSPEECH_MIN`` and ``PPN_TASK_ADJUST_BOUNDARY_NONSPEECH_STRING`` to ``TaskConfiguration``
#. Added ``ABA_NONSPEECH_TOLERANCE`` and ``ABA_NO_ZERO_DURATION`` parameters to ``RuntimeConfiguration``
#. Added more tests for ``Configuration`` and ``RuntimeConfiguration``
#. Added map from language code to human-readable name for TTS wrappers and for ``Language``, usable by ``aeneas.tools.execute_task``
#. Marked Afrikaans (``afr``) language as tested
#. Added more examples to ``aeneas.tools.execute_task``
#. Added more tool and long tests
#. Added bench tests
#. Added ``wiki/TESTING.md``
#. Added ``venvs`` directory with scripts to automate testing with virtual environments
#. Added ``RuntimeConfiguration`` parameters to switch MFCC masking on, including per level in multilevel tasks
#. Modified ``DTWAligner``, ``AudioFileMFCC``, ``ExecuteTask``, and ``VAD`` to allow MFCC masking
#. Added field human-readable descriptions in ``Configuration`` and its subclasses
#. Better ``--list-parameters`` in ``ExecuteTask``
#. Added ``--list-parameters`` in ``ExecuteJob``
#. Added ``--help-rconf`` option to all tools
#. Added check in ``ExecuteTask`` on the consistency of the computed sync map
#. Added ``RuntimeConfiguration`` parameters ``DTW_MARGIN_L1``, ``DTW_MARGIN_L2``, ``DTW_MARGIN_L3``, to change DTW margin of each level
#. Added ``FFMPEG_PARAMETERS_SAMPLE_48000`` to ``ffmpegwrapper.py``
#. Fixes issue with ``gf.relative_path()`` in Windows, if executed from a drive different than the install drive
#. Fixed a bug with empty fragments when using subprocess TTS with TTS cache enabled
#. Added ``--presets-word`` switch to ``aeneas.tools.execute_task``
#. Added ``AWSTTSWrapper`` wrapper for AWS Polly TTS API 
#. Revised docs
#. Fixed a bug in reading SMIL files with machine-readable timings
#. Fixed a bug in ``SyncMapFragmentList`` which caused sync map to contain overlapping fragments

v1.6.0.1 (2016-09-30)
---------------------

#. Fixes bug in Nuance TTS wrapper

v1.6.0 (2016-09-26)
-------------------

#. Fixed bug #102 by checking that the audio file produced by the TTS engine is mono WAVE and has correct sample rate: slightly slower but safe
#. Created ``aeneas.ttswrappers`` subpackage
#. Renamed ``aeneas.ttswrapper`` to ``aeneas.ttswrappers.basettswrapper``, and ``TTSWrapper`` to ``BaseTTSWrapper``
#. Renamed ``aeneas.espeakwrapper`` to ``aeneas.ttswrappers.espeakttswrapper``, and ``ESPEAKWrapper`` to ``ESPEAKTTSWrapper``
#. Renamed ``aeneas.festivalwrapper`` to ``aeneas.ttswrappers.festivalttswrapper``, and ``FESTIVALWrapper`` to ``FESTIVALTTSWrapper``
#. Renamed ``aeneas.nuancettsapiwrapper`` to ``aeneas.ttswrappers.nuancettswrapper``, and ``NuanceTTSAPIWrapper`` to ``NuanceTTSWrapper``
#. Modified the value for using the Nuance TTS API from ``nuancettsapi`` to ``nuance`` in ``aeneas.synthesizer.SYNTHESIZER``
#. Now each TTS wrapper must declare the format (codec, channels, sample rate) of its output
#. Now each TTS wrapper can declare the default path for the TTS engine executable, using ``DEFAULT_TTS_PATH``
#. Changed the constructor of ``BaseTTSWrapper`` and derived classes, moving call method flags from constructor parameters to class fields
#. Added check when synthesizing multiple: at least one fragment should be not empty
#. Simplified writing custom TTS wrappers by providing the "multiple generic" method in ``BaseTTSWrapper``
#. Removed ``synthesize_single()`` function in all TTS wrappers and in ``cew``
#. When working on multilevel sync, user can specify a different TTS for each level
#. Added an optional TTS caching mechanism to reduce subprocess/API calls to the TTS engine (closes #87)
#. Added wrapper for eSpeak-ng (subprocess only)
#. Added ``cfw`` Python C++ Extension to call ``Festival`` via its C++ API, disabled by default (closes #106)
#. Unified unit tests for eSpeak, eSpeak-ng, and Festival
#. Python C extension compilation can be disabled/forced in setup.py via env vars
#. Added check on head/process/tail length which should not exceed the audio file length (closes #80)
#. Moved package metadata from ``setup.py`` into ``setupmeta.py``
#. Added AGPL header to all source files
#. Removed metadata (e.g., version) from all source files, except those directly facing the user
#. PEP 8 compliance for all Python files (except for E501 "line too long")
#. Added ``wiki/CONTRIBUTING.md`` explaining the contribution rules (branch policy, code style, etc.)
#. Using Sphinx theme from readthedocs.org if available
#. Updated dependencies: BeautifulSoup4>=4.5.1 and lxml>=3.6.4 (see discussion in #93)
#. Updated documentation
#. Several other minor code improvements

v1.5.1 (2016-07-25)
-------------------

#. Added ``invoke`` parameter to ``AbstractCLIProgram`` constructor and modified tools consequently
#. Added ``aeneas.tools.hydra`` tool to run all other aeneas.tools.* scripts
#. Added ``pyinstaller-*`` configuration files for ``pyinstaller``
#. Added the ``PYINSTALLER.md`` file documenting how to compile an executable with PyInstaller
#. Added rconf ``CDTW``, ``CEW``, and ``CMFCC`` parameters to prevent running a single C extension
#. Added ``PPN_TASK_OS_FILE_EAF_AUDIO_REF`` to specify audio file URI for EAF output sync maps
#. Added function to output current date and time in EAF output sync maps
#. Removed copies of ``cint.[ch]`` and ``cwave.[ch]`` in other C extensions, changed include paths
#. Removed unused keys from ``JobConfiguration`` and related source files
#. Moved import statements for ``lxml`` and ``bs4`` inside the functions actually using them (better isolation for future purposes)
#. Fixed a numerical issue in ``dtw.py`` by explicit stating ``dtype=int`` in ``centers`` initializer, pure Python code only
#. Extension ``cew`` compiled for Mac OS X and Windows
#. Added links to installers for Mac OS X and Windows in the documentation
#. Explicitly requiring lxml v3.6.0 and BeautifulSoup4 v4.4.1 due to a change in BeautifulSoup4 v4.5.0 API (to be investigated later)

v1.5.0.3 (2016-04-23)
---------------------

#. Fix an issue in ``sd`` with ``float`` returned instead of ``TimeValue``

v1.5.0.2 (2016-04-09)
---------------------

#. Fix an issue in ``dtw`` with ``numpy.searchsorted`` returning an invalid index

v1.5.0.1 (2016-04-03)
---------------------

#. Fix an issue with compiling C extensions on Windows

v1.5.0 (2016-04-02)
-------------------

#. Rewritten ``vad.py``
#. Rewritten ``sd.py``, removed ``SDMetric``
#. Rewritten ``adjustboundaryalgorithm.py``
#. Simplified ``executetask.py``
#. Added ``Loggable`` to ``logger.py``, now most classes derive from it
#. Added ``timevalue.py`` containing an arbitrary-precision type to represent time values (instead of ``float``)
#. Added ``ttswrapper.py`` to support generic TTS engine invocation
#. Added ``festivalwrapper.py``
#. Added ``nuancettsapiwrapper.py``
#. Modified ``espeakwrapper.py`` to fit in the new TTS architecture
#. Renamed ``espeak_path`` to ``tts_path`` in ``RuntimeConfiguration``
#. Deleted ``aeneas.tools.espeak_wrapper`` CLI tool, use ``aeneas.tools.synthesize_text`` instead
#. Added ``CEWSubprocess`` to run ``aeneas.cew`` in a separate process to work around a bug in libespeak
#. Added ``aeneas/extra`` directory, containing some custom TTS wrappers
#. Changed meaning of ``language.py`` and added list of supported languages inside TTS wrappers
#. Added ``plotter.py`` to plot waveforms and sets of labels to image file
#. Added ``aeneas.tools.plot_waveform`` CLI tool
#. Added ``tree.py`` to support the new tree-like structure of ``TextFile`` and ``SyncMap``
#. Modified ``textfile.py`` with the new tree-like structure
#. Added ``multilevel`` input text format
#. Added initial support for outputting multilevel JSON, SMIL, TTML, and XML sync maps
#. Added README files and documentation to the C extensions subdirectories
#. Added Bash scripts to compile and run C drivers
#. Added usage messages to C drivers
#. Converted all ``malloc()`` calls to ``calloc()`` calls to avoid dirty allocations, added checks on the returned pointers
#. Introduced fixed-size int types in C extensions, with explicit definitions for the MS C compiler
#. Merged ``AudioFileMonoWAVE`` back into ``AudioFile``
#. More efficient append/prepend operations in ``AudioFile`` thanks to preallocated memory and space doubling
#. Created ``AudioFileMFCC`` to handle the MFCC representation of audio files
#. Added ``run_vad()`` to ``AudioFileMFCC``, ``VAD`` is just an "algorithm-switcher" class now
#. Added ``detect_head_tail()`` to ``AudioFileMFCC``, ``SD`` is just an "algorithm-switcher" class now
#. Listed supported keys in ``RuntimeConfiguration`` documentation
#. Renamed ``ConfigurationObject`` to ``Configuration``
#. Renamed ``append_*`` functions to ``add_*`` in several classes
#. Removed ``computed_path`` property in ``DTWAligner``, ``compute_path()`` now returns it
#. Fixed a bug with logger and rconf initialization in all classes
#. Added ``--cewsubprocess`` option to ``aeneas.tools.execute_job``
#. Fixed a bug in ``aeneas.tools.execute_job`` that prevented processing uncompressed containers
#. Added ``--faster-rate``, ``--rates``, and ``--zero`` options to ``aeneas.tools.execute_task``
#. More ``--example-*`` shortcuts in ``aeneas.tools.execute_task``
#. Added list of supported language codes to ``--list-values`` in ``aeneas.tools.execute_task``
#. All ``aeneas.tools.*`` CLI tools now print messages in color on POSIX OSes
#. Added ``gc.PPN_TASK_OS_FILE_NO_ZERO`` (i.e., ``os_task_file_no_zero``) to avoid fragments with zero duration in sync maps
#. Added ``"TRUE"`` and ``"YES"`` as aliases for ``True`` value in ``Configuration``
#. Added ``AUD``, ``AUDH`` and ``AUDM`` sync map output format for use with ``Audacity``
#. Added ``EAF`` sync map output format for use with ``ELAN``
#. Deprecated ``RBSE`` sync map output format
#. More unit tests
#. More uniform documentation: unless ``byte`` is specified, ``string`` indicates a Unicode string (``unicode`` in Python 2, ``str`` in Python 3)

v1.4.1 (2016-02-13)
-------------------

#. Added ``DFXP`` sync map output format, alias for ``TTML``
#. Added ``SBV`` sync map output format (SubViewer format with newline, used by YouTube)
#. Added ``SUB`` sync map output format (SubViewer format with ``[br]``)
#. Added ``aeneas.diagnostics`` to perform setup check, modified ``aeneas_check_setup.py`` accordingly
#. Marked Czech (``cs``) language as tested
#. Optimizated computation of MFCCs if no head/tail has been cut
#. Fixed the ``numpy deprecated API warning`` for C extensions
#. Fixed a few bugs and cleaned the source code of the ``cmfcc`` C extension, added a C driver program
#. Cleaned the source code of the ``cew`` C extension, added a C driver program
#. Cleaned the source code of the ``cdtw`` C extension, added a C driver program
#. Added ``cwave`` C extension (currently not used), added a C driver program
#. Added ``write`` method to ``Logger`` to dump log to file
#. Added ``ConfigurationObject`` to represent a dictionary with a fixed set of keys, default values, and aliases
#. Now ``JobConfiguration`` and ``TaskConfiguration`` extend ``ConfigurationObject``
#. Added ``RuntimeConfiguration``, extending ``ConfigurationObject``, to keep the runtime settings, tunable by (expert) users
#. Added to ``AbstractCLIProgram`` support for specifying log file path
#. Added to ``AbstractCLIProgram`` support for specifying runtime configuration
#. Changed ``FFMPEG_PARAMETERS_DEFAULT`` in ``ffmpeg.py`` to ``FFMPEG_PARAMETERS_SAMPLE_16000`` (i.e., from 22050 Hz to 16000 Hz)
#. Added support for specifying the temporary directory path in the ``RuntimeConfiguration``
#. Refactored ``mfcc.py`` to better fit into the library structure
#. Moved the original ``mfcc.py`` into the ``thirdparty/`` directory for clarity and attribution
#. Nicer ``aeneas_check_setup.py`` script
#. More unit tests covering runtime configuration options
#. Slimmed the ``README.md`` down

v1.4.0 (2016-01-15)
-------------------

#. Now running on both Python 2.7.x and Python 3.4.x or later, including C extensions
#. For XML-based sync map formats, now using ``UTF-8`` encoding instead of ``ASCII``
#. Unified ``aeneas.tools.*`` structure, with better help messages and exit codes
#. All ``aeneas.tools.*`` can be run interactively or called from Python code by passing a list of arguments
#. ``aeneas.tools.convert_syncmap`` has slightly different option names
#. ``aeneas.tools.read_text`` has a different order of arguments and different option names
#. ``aeneas.tools.synthesize_text`` has a different order of arguments and different option names
#. ``aeneas.tools.run_sd`` has a different order of arguments and different option names
#. Added ``bin/`` scripts
#. Added a flag to disable checking a language code string against listed (supported) ones, allowing for testing with custom espeak voices
#. Ported the unit test launcher ``run_all_unit_tests.py`` in Python, with more options than ``unittest discover``
#. Added unit test ``aeneas.tests.test_idsortingalgorithm``
#. Added unit tests for ``aeneas.tools.*`` (``--tool-tests``)
#. Added unit tests for ``executejob.py`` and ``executetask.py`` (``--long-tests``)
#. Added unit tests for ``downloader.py`` and ``aeneas.tools.download`` (``--net-tests``)
#. Better and more unit tests
#. Changed all ``IOError`` to ``OSError``, per Python 3 recommendation
#. Changed ``parameters=None`` default value in the constructor of ``FFMPEGWrapper``
#. Renamed ``AudioFileMonoWAV`` to ``AudioFileMonoWAVE``
#. Renamed ``best_audio`` parameter to ``largest_audio`` in ``downloader.py`` and in ``aeneas.tools.execute_task`` and ``aeneas.tools.download``
#. Renamed ``get_rel_path`` (resp., ``get_abs_path``) into ``relative_path`` (resp., ``absolute_path``) in ``aeneas.globalfunctions``
#. Fixed a potential bug in ``relative_path``: now getting the cwd path using ``os.getcwd()``
#. Fixed a bug in ``cew.c`` triggered when passing espeak voices with variants (e.g., ``en-gb``)

v1.3.3 (2015-12-20)
-------------------

#. Added all voice variants (e.g., ``en-gb`` to ``language.py``) supported by espeak v1.48.03

v1.3.2 (2015-11-11)
-------------------

#. Added ``is_text_file_ignore_regex`` parameter to ignore text from the input file
#. Added ``is_text_file_transliterate_map`` parameter to read a transliteration map from file and apply it to the input text
#. Added ``thirdparty/transliteration.map`` sample transliteration map (courtesy of Steve Gallagher and Richard Margetts)
#. Edited ``README.md``, stating the optional dependency from ``pafy``
#. Renamed ``check_dependencies.py`` into ``aeneas_check_setup.py``

v1.3.1.1 (2015-11-03)
---------------------

#. Added ``debian/`` directory containing files for creating a Debian/Ubuntu ``.deb`` package (courtesy of Chris Hubbard)
#. Removed ``pafy`` from required dependencies

v1.3.1 (2015-10-28)
-------------------

#. Added ``os_task_file_id_regex`` parameter to add user-defined ``id`` values for ``plain`` and ``subtitles`` input files
#. Added the HTML file ``finetuneas.html`` for manually fine tuning the sync maps (courtesy of Firat Özdemir)
#. Added an option to ``aeneas.tools.convert_syncmap`` and ``aeneas.tools.execute_task`` to output ``finetuneas`` HTML file

v1.3.0 (2015-10-14)
-------------------

#. Added ``cew`` C module for synthesizing text with ``espeak`` much faster than in pure Python (only available on Linux at the moment)
#. Added ``wavfile.py`` from ``scipy.io`` to replace ``scikits.audiolab``
#. Added ``AudioFileMonoWAV``, containing all the mono WAV functions previously in ``AudioFile``
#. Added ``is_audio_file_tail_length`` parameter
#. Added exception handling, especially in ``aeneas.tools.*``
#. Added ``Downloader`` to download files from Web sources (currently, audio from YouTube)
#. Added the corresponding ``aeneas.tools.download`` utility
#. Added ``pafy`` as a Python dependency, and removed ``scikits.audiolab``
#. Added third party licenses
#. Unified the I/O of ``aeneas.tools.*``, creating the ``aeneas/tools/res/`` and ``output/`` directories
#. Better and more unit tests
#. Improved documentation, especially the ``README.md``
#. Added ``licenses/`` directory, containing the licenses of third party code

v1.2.0 (2015-09-27)
-------------------

#. Added ``sd.py`` to automatically detect the head/tail/interval of an audio file
#. Added the corresponding ``aeneas.tools.run_sd`` utility
#. Added the corresponding Task configuration parameters: ``is_audio_file_detect_head_min``, ``is_audio_file_detect_head_max``, ``is_audio_file_detect_tail_min``, ``is_audio_file_detect_tail_max``, and ``os_task_file_head_tail_format``
#. Added ``SMILH`` and ``SMILM`` sync map output formats (``SMIL`` becoming an alias of ``SMILH``)
#. Added ``CSVM``, ``SSVM``, ``TSVM``, and ``TXTM`` formats (``CSV``, ``SSV``, ``TSV``, and ``TXT`` becoming their aliases)
#. Renamed the previous ``JSON`` sync map output format to ``RBSE``
#. Added a new ``JSON`` format
#. Renamed the previous ``XML`` sync map output format to ``XML_LEGACY``
#. Changed ``JSON`` (and ``RBSE``) write function, now using the ``json`` library
#. Added a new ``XML`` format
#. Changed ``SMIL``, ``TTML``, and ``XML`` write functions, now using the ``lxml`` library
#. Added functions to read sync map files
#. Added the ``aeneas.tools.convert_syncmap`` utility to convert sync maps
#. Added ``reverse``, ``trim``, and ``write`` functions to ``AudioFile``
#. Added all the languages that espeak v1.48.03 supports to the ``Language`` enumeration (those not tested yet are marked as such)
#. Marked Persian (``fa``) and Swahili (``sw``) languages as tested
#. Added the ``aeneas.tools.synthesize_text`` utility to synthesize multiple fragments into a single wave file
#. Changed ``FFMPEG_PARAMETERS_DEFAULT`` in ``ffmpeg.py`` to ``FFMPEG_PARAMETERS_SAMPLE_22050`` (i.e., from 44100 Hz to 22050 Hz)
#. Fixed the ``TTML`` output
#. Fixed a ``KeyError`` bug in ``ffprobewrapper.py`` when probing a file not recognized as audio file
#. Fixed a bug in ``cdtw.c``: int overflow when computing the ``centers`` array on long (>30 minutes) audio files
#. Many unit tests have been rewritten, extended, or refactored
#. Other minor fixes and code/documentation improvements

v1.1.2 (2015-09-24)
-------------------

#. Better ``setup.py``, especially for Windows users (courtesy of David Smith)

v1.1.1 (2015-08-23)
-------------------

#. Added ``compile_c_extensions.bat`` and directions for Windows users (courtesy of Richard Margetts)
#. Added warning to ``aeneas.tools.*`` when running without Python C extensions compiled
#. Improved ``README.md``

v1.1.0 (2015-08-21)
-------------------

#. Added ``cdtw`` C module for running the DTW much faster than in pure Python (falling back to Python if ``cdtw`` cannot be load)
#. Added ``cmfcc`` C module for extracting the MFCCs much faster than in pure Python (falling back to Python if ``cmfcc`` cannot be load)
#. Moved code for extracting MFCCs into ``AudioFile``, and rewritten ``dtw.py`` and ``vad.py`` accordingly
#. Added ``aeneas.tools.extract_mfcc`` utility
#. Rewritten the ``STRIPE`` and ``EXACT`` (Python) algorithms to compute the accumulated cost matrix in place
#. Renamed ``ALIGNER_USE_EXACT_ALGO_WHEN_MARGIN_TOO_LARGE`` to ``ALIGNER_USE_EXACT_ALGORITHM_WHEN_MARGIN_TOO_LARGE``
#. Removed ``STRIPE_NOT_OPTIMIZED`` algorithm from ``dtw.py``
#. Added the ``OFFSET`` and ``RATEAGGRESSIVE`` boundary adjustment algorithms
#. Cleaned the code for ``RATE`` boundary adjustment algorithm
#. Other minor fixes and code/docs improvements

v1.0.4 (2015-08-09)
-------------------

#. Added boundary adjustment algorithm
#. Added VAD algorithm and ``aeneas.tools.run_vad`` utility
#. Added ``subtitles`` input text format and the ability of dealing with multiline text fragments
#. Added ``SSV`` output format
#. Added ``CSVH``, ``SSVH``, ``TSVH``, ``TXTH`` output formats (i.e., human-readable variants)
#. Added ``-v`` option to ``aeneas.tools.execute_task`` and ``aeneas.tools.execute_job`` to produce verbose output
#. Added ``install_dependencies.sh``
#. Added this changelog
#. Sanitized log messages, fixing a problem with ``tee=True`` crashing in non UTF-8 shells (tested in a POSIX shell)
#. Improved unit tests
#. Other minor fixes and code/docs improvements

v1.0.3 (2015-06-13)
-------------------

#. Added ``TSV`` output format
#. Added reference to ``aeneas-vagrant``
#. Added ``run_all_unit_tests.sh``

v1.0.2 (2015-05-14)
-------------------

#. Corrected typos
#. Merged ``requirements.txt``

v1.0.1 (2015-05-12)
-------------------

#. Initial version


