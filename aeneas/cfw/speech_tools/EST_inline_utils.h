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

/**@name EST_inline_utils.h
  * Some simple inline functions gathered together in one place.
  * <p>
  * @author Richard Caley <rjc@cstr.ed.ac.uk>
  * @version $Id: EST_inline_utils.h,v 1.3 2004/05/04 00:00:17 awb Exp $
  */

//@{

#ifndef __EST_INLINE_UTILS_H__
#define __EST_INLINE_UTILS_H__

/// Round to nearest integer
static inline int irint(float f) { return (int)(f+0.5); }
/// Round to nearest integer
static inline int irint(double f) { return (int)(f+0.5); }
/// Round to nearest integer
static inline int srint(float f) { return (short)(f+0.5); }
/// Round to nearest integer
static inline int srint(double f) { return (short)(f+0.5); }
/// Round down
static inline int ifloor(float f) { return (int)(f); }
/// Round up
static inline int iceil(float f) { return (int)(f+0.9999999); }

/// Smaller of two ints
static inline int min(int a, int b) { return (a<b)?a:b; }
/// Larger of two ints
static inline int max(int a, int b) { return (a>b)?a:b; }
/// Smaller of two floats
static inline float min(float a, float b) { return (a<b)?a:b; }
/// Larger of two floats
static inline float max(float a, float b) { return (a>b)?a:b; }
/// Smaller of two doubles
static inline double min(double a, double b) { return (a<b)?a:b; }
/// Larger of two doubles
static inline double max(double a, double b) { return (a>b)?a:b; }

/// Absolute value.
static inline short  absval(short n) { return n<0?-n:n; }
/// Absolute value.
static inline int    absval(int n) { return n<0?-n:n; }
/// Absolute value.
static inline float  absval(float n) { return n<0.0?-n:n; }
/// Absolute value.
static inline double absval(double n) { return n<0.0?-n:n; }

#endif
//@}
