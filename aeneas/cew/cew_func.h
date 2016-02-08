/*

Python C Extension for synthesizing text with espeak

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.4.1"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

*/

struct FRAGMENT_INFO {
    float begin;
    float end;
    const char *voice_code;
    const char *text;
};

// synthesize a single text fragment
int _synthesize_single(
    const char *output_file_path,
    int *sample_rate_ret,
    struct FRAGMENT_INFO *ret
);

// synthesize multiple fragments
int _synthesize_multiple(
    const char *output_file_path,
    struct FRAGMENT_INFO **ret,
    const int number_of_fragments,
    const float quit_after,
    const int backwards,
    int *sample_rate_ret,
    unsigned int *synthesized_ret
);



