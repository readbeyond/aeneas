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
 /*************************************************************************/
 /*                                                                       */
 /*                 Author: Richard Caley (rjc@cstr.ed.ac.uk)             */
 /*                   Date: Thu Aug 14 1997                               */
 /* --------------------------------------------------------------------  */
 /* Fatal error calls.                                                    */
 /*                                                                       */
 /*************************************************************************/

#ifndef __EST_ERROR_H__
#define __EST_ERROR_H__

/* may get included from C */
#ifdef __cplusplus
#include <cstdarg>
#include <cstdio>
#else
#include <stdarg.h>
#include <stdio.h>
#endif

#include <setjmp.h>
#include "EST_unix.h"


#ifdef __cplusplus
extern "C" {
#endif

#define MAX_ERROR_MESSAGE_LENGTH 1024

typedef void (*EST_error_handler)(const char *format, ...);
extern const char *EST_error_where;
extern char *EST_error_message;

extern EST_error_handler EST_bug_func;
extern EST_error_handler EST_error_func;
extern EST_error_handler EST_sys_error_func;
extern EST_error_handler EST_warning_func;
extern EST_error_handler old_error_function;
extern EST_error_handler old_sys_error_function;

extern FILE *EST_error_stream;
extern FILE *EST_warning_stream;

extern jmp_buf *est_errjmp;
extern long errjmp_ok;

extern void EST_errors_default();
extern void EST_errors_quiet();

void EST_quiet_error_fn(const char *format, ...);
void EST_quiet_sys_error_fn(const char *format, ...);


#define _rxp_S_(X) #X
#define _rxp_s_(X) _rxp_S_(X)

#define EST_bug       (EST_error_where = __FILE__ ", line " _rxp_s_(__LINE__)),\
			(*EST_bug_func)

#if defined(EST_DEBUGGING)
#define EST_exit(N) abort()
#define EST_error     (EST_error_where = __FILE__ ", line " _rxp_s_(__LINE__)),\
			(*EST_error_func)
#define EST_warning   (EST_error_where = __FILE__ ", line " _rxp_s_(__LINE__)),\
			(*EST_warning_func)
#define EST_sys_error (EST_error_where = __FILE__ ", line " _rxp_s_(__LINE__)),\
			  (*EST_sys_error_func)
#else

#define EST_exit(N) exit(N)
#define EST_error     (EST_error_where = NULL),\
			(*EST_error_func)
#define EST_warning   (EST_error_where = NULL),\
			(*EST_warning_func)
#define EST_sys_error (EST_error_where = NULL),\
			  (*EST_sys_error_func)
#endif

#define est_error_throw() (est_errjmp ? longjmp(*est_errjmp,1) : (void)EST_exit(-1))
#define est_error() est_error_throw()

#define CATCH_ERRORS_SKEL( INIT,CLEANUP) \
	{ \
	INIT \
	jmp_buf *old_errjmp = est_errjmp; \
	int old_errjmp_ok = errjmp_ok; \
	errjmp_ok =1; \
        est_errjmp = (jmp_buf *)malloc(sizeof(jmp_buf)); \
	int jmp_val = setjmp(*est_errjmp); \
	if (jmp_val) { free(est_errjmp); est_errjmp = old_errjmp;  errjmp_ok = old_errjmp_ok; CLEANUP} \
        if (jmp_val)

#define CATCH_ERRORS() \
	  CATCH_ERRORS_SKEL(\
	const int est_err_quiet=0; \
	, \
	;)

#define CATCH_ERRORS_QUIET() \
	  CATCH_ERRORS_SKEL(\
	  const int est_err_quiet=1; \
	  EST_error_handler old_error_function=EST_error_func; \
	  EST_error_handler old_sys_error_function=EST_sys_error_func; \
	  EST_error_func = EST_quiet_error_fn; \
	  EST_sys_error_func = EST_quiet_sys_error_fn; \
	  , \
	  EST_error_func=old_error_function; \
	  EST_sys_error_func=old_sys_error_function; \
	  )

#define END_CATCH_ERRORS() \
	   free(est_errjmp); \
           est_errjmp = old_errjmp; \
	   errjmp_ok = old_errjmp_ok; \
           if (est_err_quiet) { \
		EST_error_func=old_error_function; \
		 EST_sys_error_func=old_sys_error_function; \
		} \
	   } while (0)

/** Defines the attitude of a call to possible fatal errors.
  * Passing one of these values to a function tells it how much
  * care it needs to take to avoid calls to EST_error.
  * 
  * These need snappier names.
  * the numbers are their for historical reasons
  */
enum EST_error_behaviour
{
  /** Function will not normally return an error unless something 
    * really bad has gone wrong.  For feature lookup, will return
    * default value if feature doesn't exist
    */
  est_errors_checked = 0,

  /** Function will throw errors when feature doesn't exist.
    */
  est_errors_allowed = 1,

  /** No fatal errors allowed. Function must catch all EST_error calls.
   *  Will *always* return a default value.
    */
  est_errors_never = 2
};

#ifdef __cplusplus
	   }

#include "EST_String.h"

/* These are used to pass values into error functions inside             */
/* templates. For classes we can define a function cast to EST_String,   */
/* but we need the basic versions.                                       */

inline const char *error_name(const EST_String val) {return val;}
inline const char *error_name(const void *val) {return EST_String::cat("<<ptr:", EST_String::Number((long)val, 16), ">>");}
inline const char *error_name(const EST_Regex val) {return val.tostring();}
inline const char *error_name(int val) {return EST_String::Number(val);}
inline const char *error_name(long val) {return EST_String::Number(val);}
inline const char *error_name(float val) {return EST_String::Number(val);}
inline const char *error_name(double val) {return EST_String::Number(val);}
inline const char *error_name(char val) {return EST_String::FromChar(val);}

#endif

#endif
