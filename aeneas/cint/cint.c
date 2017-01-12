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

Portable fixed-size int definitions for the other Python C extensions.

*/

#include "cint.h"

uint8_t le_u8_to_cpu(const unsigned char *buf) {
    return (uint8_t)buf[0];
}
uint8_t be_u8_to_cpu(const unsigned char *buf) {
    return (uint8_t)buf[0];
}
uint16_t le_u16_to_cpu(const unsigned char *buf) {
    return ((uint16_t)buf[0]) | (((uint16_t)buf[1]) << 8);
}
uint16_t be_u16_to_cpu(const unsigned char *buf) {
    return ((uint16_t)buf[1]) | (((uint16_t)buf[0]) << 8);
}
uint32_t le_u32_to_cpu(const unsigned char *buf) {
    return ((uint32_t)buf[0]) | (((uint32_t)buf[1]) << 8) | (((uint32_t)buf[2]) << 16) | (((uint32_t)buf[3]) << 24);
}
uint32_t be_u32_to_cpu(const unsigned char *buf) {
    return ((uint32_t)buf[3]) | (((uint32_t)buf[2]) << 8) | (((uint32_t)buf[1]) << 16) | (((uint32_t)buf[0]) << 24);
}

int8_t le_s8_to_cpu(const unsigned char *buf) {
    return (uint8_t)buf[0];
}
int8_t be_s8_to_cpu(const unsigned char *buf) {
    return (uint8_t)buf[0];
}
int16_t le_s16_to_cpu(const unsigned char *buf) {
    return ((uint16_t)buf[0]) | (((uint16_t)buf[1]) << 8);
}
int16_t be_s16_to_cpu(const unsigned char *buf) {
    return ((uint16_t)buf[1]) | (((uint16_t)buf[0]) << 8);
}
int32_t le_s32_to_cpu(const unsigned char *buf) {
    return ((uint32_t)buf[0]) | (((uint32_t)buf[1]) << 8) | (((uint32_t)buf[2]) << 16) | (((uint32_t)buf[3]) << 24);
}
int32_t be_s32_to_cpu(const unsigned char *buf) {
    return ((uint32_t)buf[3]) | (((uint32_t)buf[2]) << 8) | (((uint32_t)buf[1]) << 16) | (((uint32_t)buf[0]) << 24);
}

void cpu_to_le_u8(unsigned char *buf, uint8_t val) {
    buf[0] = (val & 0xFF);
}
void cpu_to_be_u8(uint8_t *buf, uint8_t val) {
    buf[0] = (val & 0xFF);
}
void cpu_to_le_u16(unsigned char *buf, uint16_t val) {
    buf[0] = (val & 0x00FF);
    buf[1] = (val & 0xFF00) >> 8;
}
void cpu_to_be_u16(uint8_t *buf, uint16_t val) {
    buf[0] = (val & 0xFF00) >> 8;
    buf[1] = (val & 0x00FF);
}
void cpu_to_le_u32(unsigned char *buf, uint32_t val) {
    buf[0] = (val & 0x000000FF);
    buf[1] = (val & 0x0000FF00) >> 8;
    buf[2] = (val & 0x00FF0000) >> 16;
    buf[3] = (val & 0xFF000000) >> 24;
}
void cpu_to_be_u32(uint8_t *buf, uint32_t val) {
    buf[0] = (val & 0xFF000000) >> 24;
    buf[1] = (val & 0x00FF0000) >> 16;
    buf[2] = (val & 0x0000FF00) >> 8;
    buf[3] = (val & 0x000000FF);
}

void cpu_to_le_s8(unsigned char *buf, int8_t val) {
    buf[0] = (val & 0xFF);
}
void cpu_to_be_s8(uint8_t *buf, int8_t val) {
    buf[0] = (val & 0xFF);
}
void cpu_to_le_s16(unsigned char *buf, int16_t val) {
    buf[0] = (val & 0x00FF);
    buf[1] = (val & 0xFF00) >> 8;
}
void cpu_to_be_s16(uint8_t *buf, int16_t val) {
    buf[0] = (val & 0xFF00) >> 8;
    buf[1] = (val & 0x00FF);
}
void cpu_to_le_s32(unsigned char *buf, int32_t val) {
    buf[0] = (val & 0x000000FF);
    buf[1] = (val & 0x0000FF00) >> 8;
    buf[2] = (val & 0x00FF0000) >> 16;
    buf[3] = (val & 0xFF000000) >> 24;
}
void cpu_to_be_s32(uint8_t *buf, int32_t val) {
    buf[0] = (val & 0xFF000000) >> 24;
    buf[1] = (val & 0x00FF0000) >> 16;
    buf[2] = (val & 0x0000FF00) >> 8;
    buf[3] = (val & 0x000000FF);
}

