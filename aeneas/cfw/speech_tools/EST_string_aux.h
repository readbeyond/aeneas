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
/*                    Author :  Paul Taylor, Simon King                  */
/*                    Date   :  1994-99                                  */
/*-----------------------------------------------------------------------*/
/*            Utility EST_String Functions header file                   */
/*                                                                       */
/*=======================================================================*/

#ifndef __EST_STRING_AUX_H__
#define __EST_STRING_AUX_H__

#include "EST_TList.h"
#include "EST_String.h"
#include "EST_types.h"
#include "EST_rw_status.h"

void StringtoStrList(EST_String s, EST_StrList &l, EST_String sep="");
void BracketStringtoStrList(EST_String s, EST_StrList &l, EST_String sep="");

EST_read_status load_StrList(EST_String filename, EST_StrList &l);
EST_write_status save_StrList(EST_String filename, EST_StrList &l, 
			      EST_String style="words");


void strip_quotes(EST_String &s, const EST_String quote_char="\"");

// makes EST_String from integer.
EST_String itoString(int n); 
// makes EST_String from float, with variable precision
EST_String ftoString(float n, int pres=3, int width=0, int l=0); 
int Stringtoi(EST_String s);

int StrListtoIList(EST_StrList &s, EST_IList &il);
int StrListtoFList(EST_StrList &s, EST_FList &il);

void StrList_to_StrVector(EST_StrList &l, EST_StrVector &v);
void StrVector_to_StrList(EST_StrVector &v,EST_StrList &l);
int  StrVector_index(const EST_StrVector &v,const EST_String &s);

int strlist_member(const EST_StrList &l,const EST_String &s);
int strlist_index(const EST_StrList &l,const EST_String &s);

// strips path off front of filename
EST_String basename(EST_String full, EST_String ext=""); 

// this is not the right place for these
void IList_to_IVector(EST_IList &l, EST_IVector &v);
void IVector_to_IList(EST_IVector &v,EST_IList &l);
int  IVector_index(const EST_IVector &v,const int s);

int ilist_member(const EST_IList &l,int i);
int ilist_index(const EST_IList &l,int i);

#endif // __EST_STRING_AUX_H__
