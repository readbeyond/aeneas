/*

# aeneas is a Python/C library and a set of tools
# to automagically synchronize audio and text (aka forced alignment)
#
# Copyright (C) 2012-2013, Alberto Pettarin (www.albertopettarin.it)
# Copyright (C) 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
# Copyright (C) 2015-2017, Alberto Pettarin (www.albertopettarin.it)
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

Python C Extension for synthesizing text with eSpeak

*/

#define CEW_SUCCESS 0
#define CEW_FAILURE 1

struct FRAGMENT_INFO {
    float begin;
    float end;
    const char *voice_code;
    const char *text;
};

/*
    Synthesize multiple text fragments,
    described by the FRAGMENT_INFO fragments_ret array,
    creating a WAVE file at output_file_path.

    If quit_after > 0, then the synthesis is terminated
    as soon as the total duration reaches >= quit_after seconds.

    If backwards is != 0, then the synthesis is done
    backwards, from the end of the fragments array.
    This option is meaningful only if quit_after is > 0,
    otherwise it has no effect.

    The sample rate of the output WAVE file is stored
    in sample_rate_ret, the number of synthesized fragments
    in synthesized_ret, and the begin and end times
    are stored in the begin and end attributes of
    the elements of fragments_ret.
*/
int _synthesize_multiple(
    const char *output_file_path,
    struct FRAGMENT_INFO **fragments_ret,
    const size_t number_of_fragments,
    const float quit_after,
    const int backwards,
    int *sample_rate_ret, // int because the espeak lib returns it as such
    size_t *synthesized_ret
);



