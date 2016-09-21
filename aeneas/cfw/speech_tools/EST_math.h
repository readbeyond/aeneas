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
/*                    Author :  Alan W Black                             */
/*                    Date   :  August 1996                              */
/*-----------------------------------------------------------------------*/
/*       OS system dependent math routines                               */
/*   You may use this instead of math.h to get a system independent      */
/*   interface to the math functions (or include in addition, it's up to */
/*   you)                                                                */
/*=======================================================================*/
#ifndef __EST_MATH_H__
#define __EST_MATH_H__

#if defined(__APPLE__)
/* Not sure why I need this here, but I do */
extern "C" int isnan(double);
#endif

/* this isn't included from c, but just to be safe... */
#ifdef __cplusplus
#include <cmath>
#include <climits>
#include <cfloat>
#else
#include <math.h>
#include <limits.h>
#include <float.h>
#endif

using namespace std;

#ifdef __cplusplus
extern "C" {
#endif

/* Although isnan(double) exists on all machine isnanf(float) does not */
/* Automatic conversion between floats to doubles for out of range     */
/* values in ANSI is undefined so we can't depend on that, but I       */

/* Solaris 2.X and SGIs IRIX*/
#if defined(__svr4__) || defined(__SYSTYPE_SVR4__)
#include <ieeefp.h>
#endif

/* SunOS 4.1.X */
/* It doesn't exist on SunOS.  One could use the macro that Solaris uses */
/* but I can't including it here, besides the follow will almost definitely */
/* have the same effect                                                     */
/* The defines are of course heuristics, this fails for NetBSD */
#if defined(__sun__) && defined(__sparc__) && !defined(__svr4__)
#define isnanf(X) isnan(X)
#endif

/* Linux (and presumably Hurd too as Linux is GNU libc based) */
/* Sorry I haven't confirmed this cpp symbol yet              */
#if defined(linux)
#define isnanf(X) __isnanf(X)
#endif

/* OS/2 with gcc EMX */
#if defined(__EMX__)
#define isnanf(X) isnan(X)
#define finite(X) isfinite(X)
#endif

/* AIX */
#if defined(_AIX)
#define isnanf(X) isnan(X)
#endif

/* Apple OSX */
#if defined(__APPLE__)
#define isnanf(X) isnan((double)(X))
/* on some previous versions of OSX we seemed to need the following */
/* but not on 10.4 */
/* #define isnan(X) __isnan(X) */
#endif

/* FreeBSD *and other 4.4 based systems require anything, isnanf is defined */
#if defined(__FreeBSD__)

#endif

/* Cygwin (at least cygwin 1.7 with gcc 4.3.4) */ 
#if defined(__CYGWIN__)
#if __GNUG__ > 3
#define isnanf(X) isnan(X)
#endif
#endif

/* WIN32 has stupid names for things */
#if defined(SYSTEM_IS_WIN32)
#define isfinite(X) _finite(X)
#define finite(X) _finite(X)
#define round(X) win32_round(X)
  inline double win32_round(double d) { return (d>0.0)?floor(d+0.5):ceil(d-0.5);}
#endif

/* These are making assumptions about the under lying architecture  */
/* that could be wrong (though most probably in a conservative way) */
#ifndef MAXFLOAT
#define MAXFLOAT ((float)3.0e+37)
#endif
#ifndef FLT_MAX
#define FLT_MAX ((float)3.0e+37)
#endif
#ifndef MINFLOAT
#define MINFLOAT ((float)1e-37)
#endif
#ifndef FLT_MAX
#define FLT_MIN ((float)1e-37)
#endif

#ifndef PI
#define PI 3.14159265358979323846
#endif
#ifndef M_PI
#define M_PI PI
#endif

#ifndef RAND_MAX
#define RAND_MAX 32767
#endif

#define SAFE_LOG_ZERO -9538

#define EST_NINT(X) ((int)((X)+0.5))

inline double safe_log(const double x)
{
    double l;
    if (x == 0)
	return SAFE_LOG_ZERO;
    l=log(x);
    if (l<SAFE_LOG_ZERO)
	return SAFE_LOG_ZERO;
    else
	return l;
}

inline double safe_exp(const double x)
{
    if(x<=SAFE_LOG_ZERO)
	return 0;
    else
	return exp(x);
}

inline double safe_log10(const double x)
{
    double l;
    if (x == 0)
	return SAFE_LOG_ZERO;
    l=log10(x);
    if(l<SAFE_LOG_ZERO)
	return SAFE_LOG_ZERO;
    else
	return l;
}

inline double safe_exp10(const double x)
{
    if(x<=SAFE_LOG_ZERO)
	return 0;
    else
      return pow(10.0,x);
}


#ifdef __cplusplus
}
#endif


#endif /*__EST_CUTILS_H__ */
