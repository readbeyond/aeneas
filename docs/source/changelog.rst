Changelog
=========

v1.1.0 (2015-08-29)
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


