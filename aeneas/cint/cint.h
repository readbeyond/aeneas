/*

Portable fixed-size int definitions for the other Python C extensions.

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

#ifdef _MSC_VER
typedef          __int8  int8_t;
typedef          __int16 int16_t;
typedef          __int32 int32_t;
typedef          __int64 int64_t;
typedef unsigned __int8  uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
#else
#include <stdint.h>
#endif

uint8_t le_u8_to_cpu(const unsigned char *buf);
uint8_t be_u8_to_cpu(const unsigned char *buf);
uint16_t le_u16_to_cpu(const unsigned char *buf);
uint16_t be_u16_to_cpu(const unsigned char *buf);
uint32_t le_u32_to_cpu(const unsigned char *buf);
uint32_t be_u32_to_cpu(const unsigned char *buf);

int8_t le_s8_to_cpu(const unsigned char *buf);
int8_t be_s8_to_cpu(const unsigned char *buf);
int16_t le_s16_to_cpu(const unsigned char *buf);
int16_t be_s16_to_cpu(const unsigned char *buf);
int32_t le_s32_to_cpu(const unsigned char *buf);
int32_t be_s32_to_cpu(const unsigned char *buf);

void cpu_to_le_u8(unsigned char *buf, uint8_t val);
void cpu_to_be_u8(unsigned char *buf, uint8_t val);
void cpu_to_le_u16(unsigned char *buf, uint16_t val);
void cpu_to_be_u16(unsigned char *buf, uint16_t val);
void cpu_to_le_u32(unsigned char *buf, uint32_t val);
void cpu_to_be_u32(unsigned char *buf, uint32_t val);

void cpu_to_le_s8(unsigned char *buf, int8_t val);
void cpu_to_be_s8(unsigned char *buf, int8_t val);
void cpu_to_le_s16(unsigned char *buf, int16_t val);
void cpu_to_be_s16(unsigned char *buf, int16_t val);
void cpu_to_le_s32(unsigned char *buf, int32_t val);
void cpu_to_be_s32(unsigned char *buf, int32_t val);

