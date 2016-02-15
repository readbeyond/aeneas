/*

Python C Extension for reading WAVE mono files.

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

#include "cint.h"

#define CWAVE_SUCCESS 0
#define CWAVE_FAILURE 1

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
    // be = big endian in file => converted into cpu endianness
    // le = little endian in file => converted into cpu endianness
    // co = computed, always in cpu endianness

    // first 12 bytes
    //uint32_t beChunkID;          // string 'RIFF'
    uint32_t leChunkSize;          // (size of the whole file in bytes - 8)
    //uint32_t beFormat;           // string 'WAVE'

    // then, we have at least the SubchunkFmt and SubchunkData
    // in any order, and other kinds of Subchunk can be present as well
    uint32_t leSubchunkFmtSize;    // (size of the subchunk 1 in bytes - 4)
    uint16_t leAudioFormat;        // one of the WAVE_FORMAT_* values
    uint16_t leNumChannels;        // number of channels (1 = mono, 2 = stereo)
    uint32_t leSampleRate;         // samples per second (e.g. 48000, 44100, 22050, 16000, 8000)
    uint32_t leByteRate;           // leSampleRate * leNumChannels * leBitsPerSample/8 => data bytes/s
    uint16_t leBlockAlign;         // leNumChannels * leBitsPerSample/8 => bytes/sample, including all channels
    uint16_t leBitsPerSample;      // number of bits per sample (e.g., 8, 16, 32)
    uint32_t leSubchunkDataSize;   // leNumSamples * leNumChannels * leBitsPerSample/8 => data bytes

    // computed
    uint32_t coNumSamples;         // number of samples
    uint32_t coSubchunkDataStart;  // byte at which the data chunk starts
    uint32_t coBytesPerSample;     // leBitsPerSample / 8 => bytes/sample (single channel)
    uint32_t coMaxDataPosition;    // coSubchunkDataStart + leSubchunkDataSize => max byte position of data
};

// open a WAVE mono file and read header info
FILE *wave_open(const char *path, struct WAVE_INFO *audio_info);

// close an open WAVE mono file
int wave_close(FILE *audio_file_ptr);

// read samples from an open WAVE mono file
int wave_read_double(
    FILE *audio_file_ptr,
    struct WAVE_INFO *audio_info,
    double *dest,
    const uint32_t from_sample,
    const uint32_t number_samples
);



