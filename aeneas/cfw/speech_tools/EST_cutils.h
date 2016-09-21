/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                     Copyright (c) 1994,1995,1996                      */
/*                        All Rights Reserved.                           */
/*                                                                       */
/*  Permission is hereby granted, free of charge, to use and distribute  */
/*  this software and its documentation without restriction, including   */
/*  without limitation the rights to use, copy, modify, merge, publish,  */
/*  distribute, sublicense, and/or sell copies of this work, and to      */
/*  permit persons to whom this work is furnished to do so, subject to   */
/*  the following conditions:                                            */
/*   1. The code must retain the above copyright notice, this list of    */
/*      conditions and the following disclaimer.                         */
/*   2. Any modifications must be clearly marked as such.                */
/*   3. Original authors' names are not deleted.                         */
/*   4. The authors' names are not used to endorse or promote products   */
/*      derived from this software without specific prior written        */
/*      permission.                                                      */
/*                                                                       */
/*  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK        */
/*  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      */
/*  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   */
/*  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE     */
/*  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    */
/*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   */
/*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          */
/*  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       */
/*  THIS SOFTWARE.                                                       */
/*                                                                       */
/*************************************************************************/
/*                    Author :  Paul Taylor and Alan W Black             */
/*                    Date   :  July 1996                                */
/*-----------------------------------------------------------------------*/
/*              Various C utility functions                              */
/*                                                                       */
/*=======================================================================*/
#ifndef __EST_CUTILS_H__
#define __EST_CUTILS_H__

#include "EST_common.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const char * const est_tools_version;
extern const char * const est_name;
extern const char * const est_libdir;
extern const char * const est_datadir;
extern const char * const est_ostype;

#include "EST_walloc.h"
#include "EST_system.h"

#ifndef streq
#define streq(X,Y) (strcmp(X,Y)==0)
#endif

char *cmake_tmp_filename();

/* NOTE perqs (from Three Rivers) have the third byte order, are not  */
/* supported, if you find a working one let me know and I'll add      */
/* support -- awb (hoping no one responds :-)                         */
enum EST_bo_t {bo_big, bo_little, bo_perq};

extern int est_endian_loc;
/* Sun, HP, SGI Mips, M68000 */
#define EST_BIG_ENDIAN (((char *)&est_endian_loc)[0] == 0)
/* Intel, Alpha, DEC Mips, Vax */
#define EST_LITTLE_ENDIAN (((char *)&est_endian_loc)[0] != 0)
#define EST_NATIVE_BO (EST_BIG_ENDIAN ? bo_big : bo_little)
#define EST_SWAPPED_BO (EST_BIG_ENDIAN ? bo_little : bo_big)

#define SWAPINT(x) ((((unsigned)x) & 0xff) << 24 | \
                    (((unsigned)x) & 0xff00) << 8 | \
		    (((unsigned)x) & 0xff0000) >> 8 | \
                    (((unsigned)x) & 0xff000000) >> 24)
#define SWAPSHORT(x) ((((unsigned)x) & 0xff) << 8 | \
                      (((unsigned)x) & 0xff00) >> 8)
void swapdouble(double *d);
void swapfloat(float *f);

void swap_bytes_ushort(unsigned short *data, int length);
void swap_bytes_short(short *data, int length);
void swap_bytes_uint(unsigned int *data, int length);
void swap_bytes_int(int *data, int length);
void swap_bytes_float(float *data, int length);
void swap_bytes_double(double *data, int length);

enum EST_bo_t str_to_bo(const char *boname);
const char *bo_to_str(enum EST_bo_t bo);

/* return the greater of the two values */
#define Gof(a, b) (((a) > (b)) ? (a) : (b))
/* return the lesser of the two values  */
#define Lof(a, b) (((a) < (b)) ? (a) : (b))


#ifdef __cplusplus
}
#endif


#endif /*__EST_CUTILS_H__ */
