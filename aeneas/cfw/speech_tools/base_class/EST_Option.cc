/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                       Copyright (c) 1995,1996                         */
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
/*                        Author :  Paul Taylor                          */
/*                        Date   :  April 1995                           */
/*-----------------------------------------------------------------------*/
/*                           EST_Option Class                            */
/*                                                                       */
/*=======================================================================*/

#include <cstdlib>
#include "EST_Option.h"
#include "EST_io_aux.h"
#include "EST_Token.h"

static const EST_String Empty_String("");

// Fills in keyval pair. If Key already exists, overwrites value.
int EST_Option::override_val(const EST_String rkey, const EST_String rval)
{
    if (rval == "")
	return 0;

    return add_item(rkey, rval);
}

int EST_Option::override_fval(const EST_String rkey, const float rval)
{
    EST_String tmp;
    char ctmp[100];
    sprintf(ctmp, "%f", rval);
    tmp = ctmp;
    
    return override_val(rkey, tmp);
}

int EST_Option::override_ival(const EST_String rkey, const int rval)
{
    EST_String tmp;
    char ctmp[100];
    sprintf(ctmp, "%d", rval);
    tmp = ctmp;
    
    return override_val(rkey, tmp);
}

int EST_Option::ival(const EST_String &rkey, int must) const
{ 
    const EST_String &tval = val_def(rkey, Empty_String);
    if (tval != "")
	return atoi(tval);

    if (must)
	cerr << "EST_Option: No value set for " << rkey << endl;
    return 0;
}

const EST_String &EST_Option::sval(const EST_String &rkey, int must) const
{ 
    const EST_String &tval = val_def(rkey, Empty_String);
    if (tval != Empty_String)
	return tval;

    if (must)
	cerr << "EST_Option: No value set for " << rkey << endl;
    return Empty_String;
}

float EST_Option::fval(const EST_String &rkey, int must) const
{ 
    const EST_String &tval = val_def(rkey, Empty_String);
    if (tval != Empty_String)
	return atof(tval);

    if (must)
	cerr << "EST_Option: No value set for " << rkey << endl;
    return 0.0;
}

double EST_Option::dval(const EST_String &rkey, int must) const
{ 
    const EST_String &tval = val_def(rkey,Empty_String);
    if (tval != Empty_String)
	return atof(tval);

    if (must)
	cerr << "EST_Option: No value set for " << rkey << endl;
    return 0.0;
}

int EST_Option::add_iitem(const EST_String &rkey, const int &rval)
{
    char tmp[100];
    sprintf(tmp, "%d", rval);
    return add_item(rkey, tmp);
}

int EST_Option::add_fitem(const EST_String &rkey, const float &rval)
{
    char tmp[100];
    sprintf(tmp, "%f", rval);
    return add_item(rkey, tmp);
}

// load in Options from files. This function has a recursive include
// facility fpr reading nested files. Maybe there should be a check on
// the max number of allowable open files.

EST_read_status EST_Option::load(const EST_String &filename, 
			     const EST_String &comment)
{   
    EST_TokenStream ts;
    EST_String k, v;
    
    if (((filename == "-") ? ts.open(cin) : ts.open(filename)) != 0)
    {
	cerr << "can't open EST_Option input file " << filename << endl;
	return misc_read_error;
    }
    // set up the character constant values for this stream
    
    while(!ts.eof())
    {
	k = ts.get().string();
	v = ts.get_upto_eoln().string();
	if (v.contains(RXwhite, 0))
	    v = v.after(RXwhite);
    
	if (k.contains("#include"))	//recursively load additional files
	{
	    cout << "Include directive\n";
	    this->load(v);
	}

	if (!k.contains(comment, 0))
	    add_item(k, v, 0); // e a search is required.
    }
    return format_ok;
}

void EST_Option::add_prefix(EST_String prefix)
{   
    EST_Litem *ptr;
    
    for (ptr = list.head(); ptr; ptr = ptr->next())
	change_key(ptr, prefix + key(ptr));
}

void EST_Option::remove_prefix(EST_String prefix)
{   
    (void)prefix;
}

ostream& operator << (ostream& s, const EST_Option &kv)
{
    EST_Litem *ptr;
    
    for (ptr = kv.list.head(); ptr; ptr = ptr->next())
        s << kv.key(ptr) << "\t" << kv.val((EST_Litem *)ptr) << endl;
    
    return s;
}    
