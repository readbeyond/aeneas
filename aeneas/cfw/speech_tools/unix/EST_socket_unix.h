
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
 /*                   Date: Tue Sep4th 1997                               */
 /* --------------------------------------------------------------------  */
 /* Defines of things which may not be here on all unixes.                */
 /*                                                                       */
 /*************************************************************************/

#if !defined(__EST_SOCKET_UNIX_H__)
#define __EST_SOCKET_UNIX_H__ 1

#include <errno.h>

/* Solaris defines this, linux doesn't */
#if defined(sun) && !defined(SVR4)
typedef int ssize_t;
#endif

#if defined(older_solaris)
/* older versions of Solaris don't have this */
typedef int socklen_t;
#endif

#if defined(__FreeBSD__) &&  __FreeBSD__ < 4
typedef int socklen_t;
#endif

#if defined(__APPLE__) 
#endif

#if defined(__CYGWIN__) &&  __GNUC__ < 3
typedef int socklen_t;
#endif

#if defined(__osf__)
typedef int socklen_t;
#endif

#if defined(_AIX)
#include <sys/select.h>
#endif

#define NOT_A_SOCKET(FD) ((FD) <0)
#define socket_error() errno

#if defined(__cplusplus)
extern "C" {
#endif

int socket_initialise(void);

#if defined(__cplusplus)
}
#endif

#endif

