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
 /*                 Author: Richard Caley (rjc@cstr.ed.ac.uk)            */
 /*                   Date: Tue Apr  1 1997                              */
 /************************************************************************/
 /*                                                                      */
 /* A place for things to be seen by all of the speech tools.            */
 /*                                                                      */
 /************************************************************************/

#ifndef __EST_COMMON_H__
#define __EST_COMMON_H__

/* all this stuff should be common to C and C++ */

#if defined __GNUC__
  #define EST_WARN_UNUSED_RESULT __attribute__((warn_unused_result))
#elif defined __clang__
  #define EST_WARN_UNUSED_RESULT __attribute__((annotate("lo_warn_unused")))
#else
  #define EST_WARN_UNUSED_RESULT
#endif

#ifdef __cplusplus
extern "C" {
#endif

  /* Nasty, horrible, yeuch, like gag me with an algebra, man! */
#include "EST_bool.h"

#ifdef INCLUDE_DMALLOC
#  ifdef __cplusplus
#    include <cstdlib>
#  else
#    include <stdlib.h>  
#  endif
#    include <dmalloc.h>
#endif

#ifdef __cplusplus
 }
#endif

#endif
