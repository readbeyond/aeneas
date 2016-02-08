/*

Python C Extension for computing the MFCC

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

// NOTE: using unsigned int as it is 32-bit wide on all modern architectures
//       not using uint32_t because the MS C compiler does not have <stdint.h>
//       or, at least, it is not easy to use it

enum {
    WAVE_FORMAT_PCM        = 0x0001, // PCM
    WAVE_FORMAT_IEEE_FLOAT = 0x0003, // IEEE float
    WAVE_FORMAT_ALAW       = 0x0006, // 8-bit ITU-T G.711 A-law
    WAVE_FORMAT_MULAW      = 0x0007, // 8-bit ITU-T G.711 mu-law
    WAVE_FORMAT_EXTENSIBLE = 0xFFFE, // extensible format

    WAVE_CHANNELS_MONO     = 0x0001, // mono
    WAVE_CHANNELS_STERO    = 0x0002  // stereo
};

struct WAVE_INFO {
    // be = big endian
    // le = little endian

    // read
    unsigned int leChunkSize;          // (size of the whole file in bytes - 8)
    unsigned int leSubchunkFmtSize;    // (size of the subchunk 1 in bytes - 4)
    unsigned int leAudioFormat;        // one of the WAVE_FORMAT_* values
    unsigned int leNumChannels;        // number of channels (1 = mono, 2 = stereo)
    unsigned int leSampleRate;         // samples per second (e.g. 48000, 44100, 22050, 16000, 8000)
    unsigned int leByteRate;           // leSampleRate * leNumChannels * leBitsPerSample/8 => data bytes/s
    unsigned int leBlockAlign;         // beNumChannels * beBitsPerSample/8 => bytes/sample, including all channels
    unsigned int leBitsPerSample;      // number of bits per sample (e.g., 8, 16, 32)
    unsigned int leSubchunkDataSize;   // leNumSamples * leNumChannels * leBitsPerSample/8 => data bytes

    // computed
    unsigned int coNumSamples;         // number of samples
    unsigned int coSubchunkDataStart;  // byte at which the data chunk starts
    unsigned int coBytesPerSample;     // leBitsPerSample / 8 => bytes/sample (single channel)
    unsigned int coMaxDataPosition;    // coSubchunkDataStart + leSubchunkDataSize => max byte position of data
};

FILE *wave_open(const char *path, struct WAVE_INFO *audio_info);
int wave_close(FILE *audio_file_ptr);
int wave_read_double(
    FILE *audio_file_ptr,
    struct WAVE_INFO *audio_info,
    double *dest,
    const unsigned int from_sample,
    const unsigned int number_samples
);



