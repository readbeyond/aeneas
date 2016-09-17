#!/usr/bin/env python
# coding=utf-8

# aeneas is a Python/C library and a set of tools
# to automagically synchronize audio and text (aka forced alignment)
#
# Copyright (C) 2012-2013, Alberto Pettarin (www.albertopettarin.it)
# Copyright (C) 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
# Copyright (C) 2015-2016, Alberto Pettarin (www.albertopettarin.it)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
aeneas.cew is a Python C extension to synthesize text with eSpeak.

The only function provided by this module is:

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
