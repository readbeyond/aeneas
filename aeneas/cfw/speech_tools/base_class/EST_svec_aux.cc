/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                    Copyright (c) 1994,1995,1996                       */
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
/*                      Author :  Paul Taylor                            */
/*                      Date   :  May 1994                               */
/*-----------------------------------------------------------------------*/
/*                     StrVector i/o utility functions                   */
/*                                                                       */
/*=======================================================================*/
#include <cstdio>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include "EST_types.h"
#include "EST_String.h"
#include "EST_Pathname.h"
#include "EST_string_aux.h"
#include "EST_cutils.h"
#include "EST_Token.h"

EST_read_status load_TList_of_StrVector(EST_TList<EST_StrVector> &w,
					const EST_String &filename,
					const int vec_len)
{

    EST_TokenStream ts;
    EST_String s;
    EST_StrVector v;
    int c;

    if(ts.open(filename) != 0){
	cerr << "Can't open EST_TList<EST_StrVector> file " << filename << endl;
	return misc_read_error;
    }
    
    v.resize(vec_len);
//    ts.set_SingleCharSymbols("");
//    ts.set_PunctuationSymbols("");

    c=0;
    while (!ts.eof())
    {

	s = ts.get().string();
	if(s != "")
	{
	    if(c == vec_len)
	    {
		cerr << "Too many points in line - expected " << vec_len << endl;
		return wrong_format;
	    }
	    else
		v[c++] = s;
	}

	if(ts.eoln())
	{
	    if(c != vec_len)
	    {
		cerr << "Too few points in line - got "
		    << c << ", expected " << vec_len << endl;
		return wrong_format;
	    }
	    else
	    {
		w.append(v);
		c=0;
	    }
	}
    }

    ts.close();
    return format_ok;

}

