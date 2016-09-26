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
/*                    Author :  Paul Taylor                              */
/*                    Date   :  April 1994                               */
/*-----------------------------------------------------------------------*/
/*        Utility Function header file  for C and C++                    */
/*                                                                       */
/*=======================================================================*/

#ifndef __RW_STATUS_H__
#define __RW_STATUS_H__

#ifdef __cplusplus
#include <cstdio>
#include <cerrno>
#else
#include <stdio.h>
#include <errno.h>

#endif

#ifdef __cplusplus
extern "C" {
#endif


  /*
   * make_status_int builds the status as an integer
   * make_X_status casts it to the required type.
   */

#define make_status_int(STATE, REASON, ERRNO) \
  (  ((((unsigned int)(STATE))  & 0xff)   <<24) \
   | ((((unsigned int)(REASON)) & 0xff)   <<16) \
   | ((((unsigned int)(ERRNO))  & 0xffff) <<00) \
     )

#define make_read_status(STATE, REASON, ERRNO)\
    ((EST_read_status)(make_status_int(STATE, REASON, ERRNO)))

#define make_write_status(STATE, REASON, ERRNO) \
    ((EST_write_status)(make_status_int(REASON, STATE, ERRNO)))

    /*
     * Extract the bits.
     */

#define get_rw_state(STATUS)	\
    ((EST_rw_state)((((unsigned int)(STATUS)) >> 24) & 0xff))
#define get_rw_reason(STATUS)	\
    ((EST_rw_reason)((((unsigned int)(STATUS)) >> 16) & 0xff))
#define get_rw_errno(STATUS)	\
    ((int)((((unsigned int)(STATUS)) >> 0) & 0xff))


  /*
   * Why we failed.
   */

enum EST_rw_reason {
  rwr_none=0,
  rwr_format=1,
  rwr_existance=2,
  rwr_permission=3,
  rwr_system=4,
  rwr_unknown=0xff
};

/** State of rw */
enum EST_rw_state {
    /// ok
  rws_ok=0,
    /// partial
  rws_partial=1,
    /// failed
  rws_failed=0xff
};


/** Possible outcomes of a file reading operation. More stuff*/
enum EST_read_status {
    /// The file was read in successfully
  read_ok		= make_status_int(rws_ok,	rwr_none,	0),
    /// The file exists but is not in the format specified
  read_format_error	= make_status_int(rws_failed,	rwr_format,	0),
    /// The file does not exist.
  read_not_found_error	= make_status_int(rws_failed,	rwr_existance,	0),
    /// An error occurred while reading
  read_error		= make_status_int(rws_failed,	rwr_unknown,	0)
};


/** Possible outcomes of a file writing operation */
enum EST_write_status {
    /// The file was written successfully
  write_ok		= make_status_int(rws_ok,	rwr_none,	0),
    /// The file was not written successfully
  write_fail		= make_status_int(rws_failed,	rwr_unknown,	0),
    /// The file was not written successfully
  write_error		= make_status_int(rws_failed,	rwr_unknown,	0),
    /// A valid file was created, but only some of the requested data is in there
  write_partial		= make_status_int(rws_partial,	rwr_unknown,	0)
};

/** Possible outcomes of a network connection operation */
enum EST_connect_status {
  /// Connection made.
  connect_ok		= make_status_int(rws_ok,	rwr_none,	0),
  /// Connection failed.
  connect_not_found_error = make_status_int(rws_failed,	rwr_existance,	0),
  /// Connection failed.
  connect_not_allowed_error = make_status_int(rws_failed, rwr_permission, 0),
  /// System failure of some kind
  connect_system_error = make_status_int(rws_failed, rwr_system, 0),
  /// The file was not written successfully
  connect_error		= make_status_int(rws_failed,	rwr_unknown,	0)
};



#ifdef __cplusplus
}
#endif

/*
 * Temporary redefinitions of the old values.
 */

#define format_ok read_ok
#define wrong_format read_format_error
#define misc_read_error read_error
#define misc_write_error write_error

#endif /*__RW_STATUS_H__ */
