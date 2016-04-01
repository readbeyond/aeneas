/*

Python C Extension for synthesizing text with eSpeak

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.5.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

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
    Synthesize a single text fragment,
    described by the FRAGMENT_INFO fragment_ret,
    creating a WAVE file at output_file_path.

    The sample rate of the output WAVE file is stored
    in sample_rate_ret, and the begin and end times
    are stored in the begin and end attributes of
    fragment_ret.
*/
int _synthesize_single(
    const char *output_file_path,
    int *sample_rate_ret, // int because the espeak lib returns it as such
    struct FRAGMENT_INFO *fragment_ret
);

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



