/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                      Copyright (c) 1995,1996                          */
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
/*                      Author :  Alan W Black                           */
/*                      Date   :  May 1996                               */
/*-----------------------------------------------------------------------*/
/*           Class to represent ints, floats and strings                 */
/*                   and other arbitrary objects                         */
/*=======================================================================*/
#include <cstdlib>
#include "EST_Val.h"
#include "EST_string_aux.h"

val_type val_unset  = "unset";
val_type val_int    = "int";
val_type val_float  = "float";
val_type val_string = "string";

EST_Val::EST_Val(const EST_Val &c)
{
    if (c.t == val_string) 
	sval = c.sval;
    else if (c.t == val_int) 
	v.ival = c.v.ival;
    else if (c.t == val_float) 
	v.fval = c.v.fval;
    else if (c.t != val_unset)
    {    // does references not a real copy
	v.pval = new EST_Contents;
	*v.pval = *c.v.pval;
    }
    t=c.t; 
}

EST_Val::EST_Val(val_type type,void *p, void (*f)(void *))
{
    t=type;
    v.pval = new EST_Contents;
    v.pval->set_contents(p,f);
}

EST_Val::~EST_Val(void)
{
    if ((t != val_int) &&
	(t != val_float) &&
	(t != val_unset) &&
	(t != val_string))
	delete v.pval;
}

EST_Val &EST_Val::operator=(const EST_Val &c)
{
    // Have to be careful with the case where they are different types
    if ((t != val_int) &&
	(t != val_float) &&
	(t != val_unset) &&
	(t != val_string))
	delete v.pval;
	
    if (c.t == val_string) 
	sval = c.sval;
    else if (c.t == val_int) 
	v.ival = c.v.ival;
    else if (c.t == val_float) 
	v.fval = c.v.fval;
    else if (c.t != val_unset)
    {   // does references not a real copy
	v.pval = new EST_Contents;
	*v.pval = *c.v.pval;
    }
    t=c.t; 
    return *this;
}

const int EST_Val::to_int(void) const
{
    // coerce this to an int
    if (t==val_float)
	return (int)v.fval;
    else if (t==val_string)
	return atoi(sval);
    else
	return v.ival;  // just for completeness
}

const float EST_Val::to_flt(void) const
{
    // coerce this to a float
    if (t==val_int)
	return (float)v.ival;
    else if (t==val_string)
	return atof(sval);
    else
	return v.fval;  // just for completeness
}

const EST_String &EST_Val::to_str(void) const
{
    // coerce this to and save it for later
    // This requires the following casting, so we can still tell the
    // compiler this is a const function.  If this was properly declared
    // non-const vast amounts of the rest of this would also have to be
    // non-const.  So we do one nasty bit here for uniformity elsewhere.
    // Not saving the result is also a possibility but probably too
    // inefficient (maybe not with rjc's string class)
    EST_String *n = (EST_String *)((void *)&sval);
    if (t==val_int)
	*n = itoString(v.ival);
    else if (t==val_float)
    {
	if (v.fval == 0)
	    *n = "0";  // to be compatible with other's notion of fstrings
	else
	    *n = ftoString(v.fval);
    }
    else if (t != val_string)
	*n = EST_String("[Val ")+t+"]";

    return sval;
}

