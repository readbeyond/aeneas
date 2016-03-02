#!/usr/bin/env python
# coding=utf-8

"""
aeneas.cew is a Python C extension to synthesize text with eSpeak.

The functions provided by this module are:

.. function:: cew.synthesize_single(output_file_path, voice_code, text)

    Synthesize a single text fragment into a single WAVE file.

    The returned tuple ``(sr, begin, end)`` contains
    the sample rate and the begin and end time values
    of the output WAVE file.

    Note that ``begin`` is always ``0.0``, while ``end`` is equal to the
    duration of the synthesized WAVE file, in seconds.

    :param string output_file_path: the path of the WAVE file to be created, UTF-8 encoded
    :param string voice_code: the eSpeak voice code (e.g., ``en``, ``en-gb``, ``it``, etc.)
    :param string text: the text to be synthesized, UTF-8 encoded
    :rtype: tuple


.. function:: cew.synthesize_multiple(output_file_path, quit_after, backwards, text)

    Synthesize several text fragments into a single WAVE file.

    The returned tuple ``(sr, synt, anchors)`` contains
    the sample rate of the output WAVE file,
    the number of fragments actually synthesized,
    and a list of time values, each representing
    the begin time in the output WAVE file
    of the corresponding text fragment.

    Note that if ``quit_after`` is specified,
    the number ``synt`` of fragments actually synthesized
    might be less than the number of fragments in ``text``.

    :param string output_file_path: the path of the WAVE file to be created, UTF-8 encoded
    :param float quit_after: stop synthesizing after reaching the given duration (in seconds)
    :param int backwards: if nonzero, synthesize backwards, that is,
                          starting from the last fragment.
                          In any case, the fragments in the output WAVE file
                          will be in natural order.
                          This option is meaningful only if ``quit_after > 0``.
    :param list text: a list of ``(voice_code, fragment_text)`` tuples
                      with the text to be synthesized.
                      The ``voice_code`` is the the eSpeak voice code
                      (e.g., ``en``, ``en-gb``, ``it``, etc.).
                      The ``fragment_text`` must be UTF-8 encoded.
    :rtype: tuple

"""

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.5.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"



