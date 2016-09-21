/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                    Copyright (c) 1994,1995,1996                       */
/*                         All Rights Reserved.                          */
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
/*                     Author :  Paul Taylor                             */
/*                     Date   :  September 1994                          */
/*-----------------------------------------------------------------------*/
/*                     EST_Option Class header file                      */
/*                                                                       */
/*=======================================================================*/

#ifndef __EST_OPTION_H__
#define __EST_OPTION_H__

#include "EST_String.h"
#include "EST_TKVL.h"
#include "EST_rw_status.h"

/** Provide a high level interface for String String key value lists.
*/

class EST_Option: public EST_TKVL<EST_String, EST_String> {
public:
    /// add prefix to every key 
    void add_prefix(EST_String prefix);
    /// remove prefix from every key 
    void remove_prefix(EST_String prefix);

     /** read keyval list from file. The file type is an ascii file
    with each line representing one key value pair. The first entry in
    the line defines the key, and the rest, which may contain
    whitespaces, defins the value. Lines starting with the comment
    character are ignored. 
    @return returns EST_read_status errors, @see */
    EST_read_status load(const EST_String &filename, const EST_String &comment = ";");

    /// add to end of list or overwrite. If rval is empty, do nothing
    int override_val(const EST_String rkey, const EST_String rval);
    /// add to end of list or overwrite. If rval is empty, do nothing
    int override_fval(const EST_String rkey, const float  rval);
    /// add to end of list or overwrite. If rval is empty, do nothing
    int override_ival(const EST_String rkey, const int rval);
    
    /** return value of type int relating to key. By default,
     an error occurs if the key is not present. Use m=0 if
     to get a dummy value returned if key is not present */
    int ival(const EST_String &rkey, int m=1) const;

    /** return value of type float relating to key. By default,
     an error occurs if the key is not present. Use m=0 if
     to get a dummy value returned if key is not present */
    double dval(const EST_String &rkey, int m=1) const;

    /** return value of type float relating to key. By default,
     an error occurs if the key is not present. Use m=0 if
     to get a dummy value returned if key is not present */
    float fval(const EST_String &rkey, int m=1) const;

    /** return value of type String relating to key. By default,
     an error occurs if the key is not present. Use m=0 if
     to get a dummy value returned if key is not present */
    const EST_String &sval(const EST_String &rkey, int m=1) const;

    /** return value of type String relating to key. By default,
     an error occurs if the key is not present. Use m=0 if
     to get a dummy value returned if key is not present */
//    const EST_String &val(const EST_String &rkey, int m=1) const
//        { return sval(rkey,m); }
    
    int add_iitem(const EST_String &rkey, const int &rval);
    int add_fitem(const EST_String &rkey, const float &rval);

    /// print options
    friend  ostream& operator << (ostream& s, const EST_Option &kv);
};

#endif // __EST_OPTION_H__
