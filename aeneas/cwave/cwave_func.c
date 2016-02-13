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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "cwave_func.h"

static const int CWAVE_BUFFER_SIZE = 4096;

// TODO make me faster and more portable
// convert a little-endian buffer to big-endian as unsigned int
// return the number of bytes read
unsigned int _be_to_le_uint(unsigned char *buffer, const int length) {
    unsigned int ret;

    ret = 0;
    if (length == 1) {
        ret = buffer[0];
    }
    if (length == 2) {
        ret = buffer[0];
        ret |= ((buffer[1]) << 8);
    }
    if (length == 4) {
        ret = buffer[0];
        ret |= ((buffer[1]) << 8);
        ret |= ((buffer[2]) << 16);
        ret |= ((buffer[3]) << 24);
    }
    return ret;
}

// TODO make me faster and more portable
// convert a little-endian buffer to big-endian as unsigned int
// return the number of bytes read
int _be_to_le_int(unsigned char *buffer, const int length) {
    int ret;

    ret = 0;
    if (length == 1) {
        ret = buffer[0];
        ret = (ret << 24) >> 24;
    }
    if (length == 2) {
        ret = buffer[0];
        ret |= ((buffer[1]) << 8);
        ret = (ret << 16) >> 16;
    }
    if (length == 4) {
        ret = buffer[0];
        ret |= ((buffer[1]) << 8);
        ret |= ((buffer[2]) << 16);
        ret |= ((buffer[3]) << 24);
    }
    return ret;
}

// TODO make me faster and more portable
// convert a little-endian buffer to big-endian as signed double
// return the number of bytes read
double _be_to_le_double(unsigned char *buffer, const int length) {
    if (length == 1) {
        return ((double)_be_to_le_int(buffer, length)) / 128;
    }
    if (length == 2) {
        return ((double)_be_to_le_int(buffer, length)) / 32768;
    }
    if (length == 4) {
        return ((double)_be_to_le_int(buffer, length)) / 2147483648;
    }
    return 0;
}

// TODO make me faster and more portable
// read a little-endian field and convert it to big-endian into an int
// return the number of bytes read
int _read_le_field(FILE *ptr, unsigned int *dest, const int length) {
    unsigned char buffer1[1];
    unsigned char buffer2[2];
    unsigned char buffer4[4];
    unsigned char *buffer;
    int read;

    if (length == 1) {
        buffer = buffer1;
    } else if (length == 2) {
        buffer = buffer2;
    } else if (length == 4) {
        buffer = buffer4;
    } else {
        return 0;
    }
    read = fread(buffer, length, 1, ptr);
    *dest = _be_to_le_uint(buffer, length);
    return read;
}

// TODO make me faster and more portable
// read a big-endian field
// return the number of bytes read
int _read_be_field(FILE *ptr, char *dest, const int length) {
    return fread(dest, length, 1, ptr);
}

// find the "match" chunk, and store its size in size
// return 1 on success or 0 on failure
int _seek_to_chunk(FILE *ptr, struct WAVE_INFO *header, const char *match, unsigned int *size) {
    char buffer4[4];
    unsigned int chunk_size;
    const unsigned int max_pos = (*header).leChunkSize + 8; // max pos in file

    rewind(ptr);
    chunk_size = 12; // skip first 12 bytes
    while(ftell(ptr) + chunk_size + 8 < max_pos) {
        if (fseek(ptr, chunk_size, SEEK_CUR) != 0) {
            return 0;
        }
        if (_read_be_field(ptr, buffer4, 4) != 1) {
            return 0;
        }
        if (_read_le_field(ptr, &chunk_size, 4) != 1) {
            return 0;
        }
        if (memcmp(buffer4, match, 4) == 0) {
            *size = chunk_size;
            return 1;
        }
    }
    return 0;
}

// parse the header
// it assumes the given file is a RIFF WAVE file
FILE *wave_open(const char *path, struct WAVE_INFO *header) {
    FILE *ptr;
    char buffer4[4];
    struct WAVE_INFO h;

    // open file
    if (path == NULL) {
        printf("Error: path is NULL\n");
        return NULL;
    }
    ptr = fopen(path, "rb");
    if (ptr == NULL) {
        printf("Error: unable to open input file %s\n", path);
        return NULL;
    }

    // read first 12 bytes: RIFF header.leChunkSize WAVE
    rewind(ptr);
    if (_read_be_field(ptr, buffer4, 4) != 1) {
        printf("Error: cannot read beChunkID\n");
        return NULL;
    }
    if (memcmp(buffer4, "RIFF", 4) != 0) {
        printf("Error: beChunkID is not RIFF\n");
        return NULL;
    }

    if (_read_le_field(ptr, &h.leChunkSize, 4) != 1) {
        printf("Error: cannot read leChunkSize\n");
        return NULL;
    }
    //printf("leChunkSize: %d\n", header.leChunkSize);

    if (_read_be_field(ptr, buffer4, 4) != 1) {
        printf("Error: cannot read beFormat\n");
        return 0;
    }
    if (memcmp(buffer4, "WAVE", 4) != 0) {
        printf("Error: beFormat is not WAVE\n");
        return NULL;
    }

    // locate the fmt chunk
    if (! _seek_to_chunk(ptr, &h, "fmt ", &h.leSubchunkFmtSize)) {
        printf("Error: cannot locate fmt chunk\n");
        return NULL;
    }
    if (h.leSubchunkFmtSize < 16) {
        printf("Error: fmt chunk has length < 16\n");
        return NULL;
    }
    _read_le_field(ptr, &h.leAudioFormat, 2);
    _read_le_field(ptr, &h.leNumChannels, 2);
    _read_le_field(ptr, &h.leSampleRate, 4);
    _read_le_field(ptr, &h.leByteRate, 4);
    _read_le_field(ptr, &h.leBlockAlign, 2);
    _read_le_field(ptr, &h.leBitsPerSample, 2);
    if (h.leAudioFormat != WAVE_FORMAT_PCM) {
        printf("Error: leAudioFormat is not PCM\n");
        return NULL;
    }
    if (h.leNumChannels != WAVE_CHANNELS_MONO) {
        printf("Error: leNumChannels is not 1\n");
        return NULL;
    }

    // locate the data chunk
    if (! _seek_to_chunk(ptr, &h, "data", &h.leSubchunkDataSize)) {
        printf("Error: cannot locate data chunk\n");
        return NULL;
    }
    if (h.leSubchunkDataSize == 0) {
        printf("Error: data chunk has length zero\n");
        return NULL;
    }
    // here ptr is at the beginnig of the data info
    h.coSubchunkDataStart = ftell(ptr);
    // compute number of samples
    h.coNumSamples = (h.leSubchunkDataSize / (h.leNumChannels * h.leBitsPerSample / 8));
    // compute number of bytes/sample (single channel)
    h.coBytesPerSample = h.leBitsPerSample / 8;
    // max byte position
    h.coMaxDataPosition = h.coSubchunkDataStart + h.leSubchunkDataSize;

    // copy h into header and return success
    *header = h;
    return ptr;
}

// close file
int wave_close(FILE *ptr) {
    int ret;

    ret = fclose(ptr);
    ptr = NULL;
    return ret;
}

// read number_samples samples, starting from sample with index from_sample
// and save them as doubles into dest
int wave_read_double(
        FILE *ptr,
        struct WAVE_INFO *header,
        double *dest,
        const unsigned int from_sample,
        const unsigned int number_samples
    ) {
    unsigned char *buffer;
    unsigned int target_pos;
    unsigned int i, j, read, remaining;
    const unsigned int bytes_per_sample = (*header).coBytesPerSample;

    if (from_sample + number_samples > (*header).coNumSamples) {
        printf("Error: attempted reading outside data\n");
        return 0;
    }

    target_pos = (*header).coSubchunkDataStart + bytes_per_sample * from_sample;
    if (ftell(ptr) != target_pos) {
        fseek(ptr, target_pos, SEEK_SET);
    }

    buffer = (unsigned char *)calloc(CWAVE_BUFFER_SIZE, bytes_per_sample);
    remaining = number_samples;
    j = 0;
    while (remaining > 0) {
        if (remaining >= CWAVE_BUFFER_SIZE) {
            read = fread(buffer, bytes_per_sample, CWAVE_BUFFER_SIZE, ptr);
        } else {
            read = fread(buffer, bytes_per_sample, remaining, ptr);
        }
        for (i = 0; i < read; ++i) {
            dest[j++] = _be_to_le_double(buffer + i * bytes_per_sample, bytes_per_sample);
        }
        remaining -= read;
    }
    free((void *)buffer);
    buffer = NULL;

    return 1;
}



