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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "cwave_func.h"

static const int CWAVE_BUFFER_SIZE = 4096;

// convert a little-endian buffer to signed double
static double _le_to_double(unsigned char *buffer, const uint32_t length) {
    if (length == 1) {
        return ((double)le_s8_to_cpu(buffer)) / 128;
    }
    if (length == 2) {
        return ((double)le_s16_to_cpu(buffer)) / 32768;
    }
    if (length == 4) {
        return ((double)le_s32_to_cpu(buffer)) / 2147483648;
    }
    return 0.0;
}

// read a little-endian u16 field
static int _read_le_u16_field(FILE *ptr, uint16_t *dest) {
    unsigned char buffer[2];

    if (fread(buffer, 2, 1, ptr) != 1) {
        return CWAVE_FAILURE;
    }
    *dest = le_u16_to_cpu(buffer);
    return CWAVE_SUCCESS;
}

// read a little-endian u32 field
static int _read_le_u32_field(FILE *ptr, uint32_t *dest) {
    unsigned char buffer[4];

    if (fread(buffer, 4, 1, ptr) != 1) {
        return CWAVE_FAILURE;
    }
    *dest = le_u32_to_cpu(buffer);
    return CWAVE_SUCCESS;
}

// read a big-endian field
static int _read_be_field(FILE *ptr, char *dest, const int length) {
    if (fread(dest, length, 1, ptr) != 1) {
        return CWAVE_FAILURE;
    }
    return CWAVE_SUCCESS;
}

// find the "match" chunk, and store its size in "size"
static int _seek_to_chunk(FILE *ptr, struct WAVE_INFO *header, const char *match, uint32_t *size) {
    char buffer4[4];
    uint32_t chunk_size;
    const uint32_t max_pos = (*header).leChunkSize + 8; // max pos in file

    rewind(ptr);
    chunk_size = 12; // skip first 12 bytes
    while((ftell(ptr) >= 0) && (ftell(ptr) + chunk_size + 8 < max_pos)) {
        // seek to the next chunk
        if (fseek(ptr, chunk_size, SEEK_CUR) != 0) {
            return CWAVE_FAILURE;
        }
        // read the chunk description
        if (_read_be_field(ptr, buffer4, 4) != CWAVE_SUCCESS) {
            return CWAVE_FAILURE;
        }
        // read the chunk size 
        if (_read_le_u32_field(ptr, &chunk_size) != CWAVE_SUCCESS) {
            return CWAVE_FAILURE;
        }
        // compare the chunk description with the desired string
        if (memcmp(buffer4, match, 4) == 0) {
            *size = chunk_size;
            return CWAVE_SUCCESS;
        }
    }
    return CWAVE_FAILURE;
}

// open a WAVE mono file and read header info
// the header is always initialized to zero
FILE *wave_open(const char *path, struct WAVE_INFO *header) {
    FILE *ptr;
    char buffer4[4];
    struct WAVE_INFO h;

    // initialize header
    memset(header, 0, sizeof(*header));

    // open file
    if (path == NULL) {
        //printf("Error: path is NULL\n");
        return NULL;
    }
    ptr = fopen(path, "rb");
    if (ptr == NULL) {
        //printf("Error: unable to open input file %s\n", path);
        return NULL;
    }

    // read first 12 bytes: RIFF header.leChunkSize WAVE
    rewind(ptr);
    if (_read_be_field(ptr, buffer4, 4) != CWAVE_SUCCESS) {
        //printf("Error: cannot read beChunkID\n");
        return NULL;
    }
    if (memcmp(buffer4, "RIFF", 4) != 0) {
        //printf("Error: beChunkID is not RIFF\n");
        return NULL;
    }

    if (_read_le_u32_field(ptr, &h.leChunkSize) != CWAVE_SUCCESS) {
        //printf("Error: cannot read leChunkSize\n");
        return NULL;
    }

    if (_read_be_field(ptr, buffer4, 4) != CWAVE_SUCCESS) {
        //printf("Error: cannot read beFormat\n");
        return NULL;
    }
    if (memcmp(buffer4, "WAVE", 4) != 0) {
        //printf("Error: beFormat is not WAVE\n");
        return NULL;
    }

    // locate the fmt chunk
    if (_seek_to_chunk(ptr, &h, "fmt ", &h.leSubchunkFmtSize) != CWAVE_SUCCESS) {
        //printf("Error: cannot locate fmt chunk\n");
        return NULL;
    }
    if (h.leSubchunkFmtSize < 16) {
        //printf("Error: fmt chunk has length < 16\n");
        return NULL;
    }

    // read fields
    if (_read_le_u16_field(ptr, &h.leAudioFormat) != CWAVE_SUCCESS) {
        //printf("Error: cannot read leAudioFormat\n");
        return NULL;
    }
    // NOTE we fail here because we are only interested in PCM files!
    if (h.leAudioFormat != WAVE_FORMAT_PCM) {
        //printf("Error: leAudioFormat is not PCM\n");
        return NULL;
    }
    if (_read_le_u16_field(ptr, &h.leNumChannels) != CWAVE_SUCCESS) {
        //printf("Error: cannot read leNumChannels\n");
        return NULL;
    }
    // NOTE we fail here because we are only interested in mono files!
    if (h.leNumChannels != WAVE_CHANNELS_MONO) {
        //printf("Error: leNumChannels is not 1\n");
        return NULL;
    }
    if (_read_le_u32_field(ptr, &h.leSampleRate) != CWAVE_SUCCESS) {
        //printf("Error: cannot read leSampleRate\n");
        return NULL;
    }
    if (_read_le_u32_field(ptr, &h.leByteRate) != CWAVE_SUCCESS) {
        //printf("Error: cannot read leByteRate\n");
        return NULL;
    }
    if (_read_le_u16_field(ptr, &h.leBlockAlign) != CWAVE_SUCCESS) {
        //printf("Error: cannot read leBlockAlign\n");
        return NULL;
    }
    if (_read_le_u16_field(ptr, &h.leBitsPerSample) != CWAVE_SUCCESS) {
        //printf("Error: cannot read leBitsPerSample\n");
        return NULL;
    }

    // locate the data chunk
    if (_seek_to_chunk(ptr, &h, "data", &h.leSubchunkDataSize) != CWAVE_SUCCESS) {
        //printf("Error: cannot locate data chunk\n");
        return NULL;
    }
    if (h.leSubchunkDataSize == 0) {
        //printf("Error: data chunk has length zero\n");
        return NULL;
    }
    // here ptr is at the beginnig of the data info
    h.coSubchunkDataStart = (uint32_t)ftell(ptr);
    // compute number of samples
    h.coNumSamples = (h.leSubchunkDataSize / (h.leNumChannels * h.leBitsPerSample / 8));
    // compute number of bytes/sample (single channel)
    h.coBytesPerSample = h.leBitsPerSample / 8;
    // max byte position
    h.coMaxDataPosition = h.coSubchunkDataStart + h.leSubchunkDataSize;

    // copy h into header and return the pointer to the audio file 
    *header = h;
    return ptr;
}

// close a WAVE mono file previously open
int wave_close(FILE *ptr) {
    int ret;

    ret = fclose(ptr);
    ptr = NULL;
    return ret;
}

// read samples from an open WAVE mono file
int wave_read_double(
        FILE *ptr,
        struct WAVE_INFO *header,
        double *dest,
        const uint32_t from_sample,
        const uint32_t number_samples
    ) {
    unsigned char *buffer;
    uint32_t target_pos;
    const uint32_t bytes_per_sample = (*header).coBytesPerSample;
    uint32_t i, j, read, remaining;

    if (from_sample + number_samples > (*header).coNumSamples) {
        //printf("Error: attempted reading outside data\n");
        return CWAVE_FAILURE;
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
            dest[j++] = _le_to_double(buffer + i * bytes_per_sample, bytes_per_sample);
        }
        remaining -= read;
    }
    free((void *)buffer);
    buffer = NULL;

    return CWAVE_SUCCESS;
}



