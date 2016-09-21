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
/*                    Date   :  February 1997                            */
/*-----------------------------------------------------------------------*/
/*                 Utility IO Function header file                       */
/*                                                                       */
/*=======================================================================*/

#ifndef __EST_IO_AUX_H__
#define __EST_IO_AUX_H__

#include "EST_unix.h"
#include "EST_common.h"
#include "EST_String.h"
#include "EST_types.h"

EST_String make_tmp_filename();
EST_String stdin_to_file();
int writable_file(char *filename);
int readable_file(char *filename);

inline int
delete_file(const EST_String &filename)
{
    // a little unnecessary to wrap this up like this -- except
    // if you want to be portable to weird OSs
    return (unlink(filename) == 0);

    // could do more with return codes from unlink ...
}

EST_String uncompress_file_to_temporary(const EST_String &filename,const EST_String &prog_name);

int compress_file_in_place(const EST_String &filename, const EST_String &prog_name);

int compress_file(const EST_String &filename,
		  const EST_String &new_filename, 
		  const EST_String &prog_name);
		  
#define numeric_char(in) (((in < '9' ) && (in > '0')) ? TRUE : FALSE)

#ifdef WIN32
#include "Winsock2.h"
typedef SOCKET SOCKET_FD;
#else
typedef int SOCKET_FD;
#endif
int socket_receive_file(SOCKET_FD fd, const EST_String &filename);
int socket_send_file(SOCKET_FD fd, const EST_String &filename);

#endif /*__EST_IO_AUX_H__ */
