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
/*                 StrList/Vector i/o utility functions                  */
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

int StrListtoFList(EST_StrList &s, EST_FList &f)
{
    EST_Litem *p;

    for (p = s.head(); p; p = p->next())
	if (!s(p).matches(RXdouble))
	{
	    cout << 
		"Expecting a floating point value in StrListtoFlist(): got "
		    << s(p) << endl;
	    return -1;
	}
	else
	    f.append(atof(s(p)));

    return 0;
}

int StrListtoIList(EST_StrList &s, EST_IList &il)
{
    EST_Litem *p;

    for (p = s.head(); p; p = p->next())
	if (!s(p).matches(RXint))
	{
	    cout << 
		"Expecting a integer value in StrListtoIList(): got "
		    << s(p) << endl;
	    return -1;
	}
	else
	    il.append(atoi(s(p)));

    return 0;
}

// read string list eclosed in brackets. Simply a place holder for
// future use with more complicate lists.
void BracketStringtoStrList(EST_String s, EST_StrList &l, EST_String sep)
{
    s.gsub("(", "");
    s.gsub(")", "");
    StringtoStrList(s, l, sep);
}
    
void StringtoStrList(EST_String s, EST_StrList &l, EST_String sep)
{
    EST_TokenStream ts;
    EST_String tmp;

    ts.open_string(s);

    (void)sep;
    if (sep != "")  // default is standard white space
	ts.set_WhiteSpaceChars(sep);
    ts.set_SingleCharSymbols(";");

    // modified by simonk - was appending an empty
    // string at end of list. 
    // unmodified back again by pault
    while (!ts.eof())
	l.append(ts.get().string());
        
    ts.close();
    return;
}
    
void StrListtoString(EST_StrList &l, EST_String &s, EST_String sep)
{
    for (EST_Litem *p = l.head(); p; p = p->next())
	s += l(p) + sep;
}
    
EST_read_status load_StrList(EST_String filename, EST_StrList &l)
{
    EST_TokenStream ts;
    EST_String s;

    if(ts.open(filename) != 0){
	cerr << "Can't open EST_StrList file " << filename << endl;
	return misc_read_error;
    }
    
    ts.set_SingleCharSymbols("");
    ts.set_PunctuationSymbols("");
    
    while (!ts.eof())
	l.append(ts.get().string());
    
    ts.close();
    return format_ok;
}

EST_write_status save_StrList(EST_String filename, EST_StrList &l, 
			      EST_String style)
{
    ostream *outf;
    EST_Litem *p;
    if (filename == "-")
	outf = &cout;
    else
	outf = new ofstream(filename);
    
    if (!(*outf))
	return write_fail;

    if (style == "words")
    {
	for (p = l.head(); p; p = p->next())
	{
	    *outf << l(p);
	    if (p->next() != 0)
		*outf << " ";
	}
	*outf << endl;
    }

    else if (style == "lines")
	for (p = l.head(); p; p = p->next())
	    *outf << l(p) << endl;
    else
    {
	cerr << "Unknown style for writing StrLists: " << style << endl;
	return misc_write_error;
    }

    delete outf;

    return write_ok;
}

int strlist_member(const EST_StrList &l,const EST_String &s)
{
    EST_Litem *p;
    for (p = l.head(); p != 0; p = p->next())
	if (l.item(p) == s)
	    return TRUE;

    return FALSE;
}

int strlist_index(const EST_StrList &l,const EST_String &s)
{
    EST_Litem *p;
    int j=0;
    for (p = l.head(); p != 0; p = p->next())
    {
	if (l.item(p) == s)
	    return j;
	j++;
    }

    return -1;
}

void StrList_to_StrVector(EST_StrList &l, EST_StrVector &v)
{
    int len,i;

    len = l.length();
    v.resize(len);

    //EST_TBI *p;
    EST_Litem *p;
    for (p = l.head(),i=0; p != 0; p = p->next(),i++)
	v[i] = l(p);
}


void StrVector_to_StrList(EST_StrVector &v, EST_StrList &l)
{
    int i;
    l.clear();
    for (i=0;i<v.length();i++)
      l.append(v[i]);
}


int StrVector_index(const EST_StrVector &v,const EST_String &s)
{
    int i;
    for(i=0;i<v.length();i++)
	if(v(i) == s)
	    return i;

    return -1;

}
