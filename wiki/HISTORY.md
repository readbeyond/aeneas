# Development History

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

**August 2015**: release of v1.1.0, including Python C extensions
to speed the computation of audio/text alignment up

**September 2015**: release of v1.2.0,
including code to automatically detect the audio head/tail

**October 2015**: release of v1.3.0,
including calling espeak via its C API (on Linux)
for faster audio synthesis, and the possibility
of downloading audio from YouTube

**November 2015**: release of v1.3.2,
for the first time available
also on [PyPI](https://pypi.python.org/pypi/aeneas/)

**January 2016**: release of v1.4.0,
supporting both Python 2.7 and 3.4 or later

**April 2016**: release of v1.5.0,
with faster C extension, multilevel alignment,
custom TTS support, and more

**July 2016**: release of v1.5.1,
with cew C extension support for Mac OS X,
and installers for Mac OS X and Windows

**September 2016**: release of v1.6.0,
with refactored TTS engine wrappers,
added TTS cache, and
experimental cfw C++ extension for Festival.

**December 2016**: release of v1.7.0,
with TextGrid output,
MFCC masking for better word-level alignment,
revised code for boundary adjustment,
and removal of long nonspeech intervals.
