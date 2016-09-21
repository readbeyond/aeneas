 /************************************************************************/
 /*                                                                      */
 /*                Centre for Speech Technology Research                 */
 /*                     University of Edinburgh, UK                      */
 /*                       Copyright (c) 1996,1997                        */
 /*                        All Rights Reserved.                          */
 /*                                                                      */
 /*  Permission is hereby granted, free of charge, to use and distribute */
 /*  this software and its documentation without restriction, including  */
 /*  without limitation the rights to use, copy, modify, merge, publish, */
 /*  distribute, sublicense, and/or sell copies of this work, and to     */
 /*  permit persons to whom this work is furnished to do so, subject to  */
 /*  the following conditions:                                           */
 /*   1. The code must retain the above copyright notice, this list of   */
 /*      conditions and the following disclaimer.                        */
 /*   2. Any modifications must be clearly marked as such.               */
 /*   3. Original authors' names are not deleted.                        */
 /*   4. The authors' names are not used to endorse or promote products  */
 /*      derived from this software without specific prior written       */
 /*      permission.                                                     */
 /*                                                                      */
 /*  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK       */
 /*  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING     */
 /*  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT  */
 /*  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE    */
 /*  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES   */
 /*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN  */
 /*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,         */
 /*  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF      */
 /*  THIS SOFTWARE.                                                      */
 /*                                                                      */
 /************************************************************************/
 /*                                                                      */
 /*                 Author: Richard Caley (rjc@cstr.ed.ac.uk)            */
 /*                   Date: Wed Jun 25 1997                              */
 /* -------------------------------------------------------------------- */
 /*                                                                      */
 /* Declaration for string comparison operation.                         */
 /*                                                                      */
 /************************************************************************/

#ifndef __EST_STRCASECMP_H__
#define __EST_STRCASECMP_H__

#ifdef __cplusplus
#include <cstdlib>
using namespace std;
#else
#include <stdlib.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

int EST_strcasecmp(const char *s1, const char *s2,
		   const unsigned char *charmap);
int EST_strncasecmp(const char *s1, const char *s2,
		    size_t n, const unsigned char *charmap);

#ifdef __cplusplus
 }

inline int EST_strcasecmp(const char *s1, const char *s2) 
	{ return EST_strcasecmp(s1, s2, NULL); };
inline int EST_strncasecmp(const char *s1, const char *s2, size_t n)
	{ return EST_strncasecmp(s1, s2, n, NULL); };

#endif

#endif
