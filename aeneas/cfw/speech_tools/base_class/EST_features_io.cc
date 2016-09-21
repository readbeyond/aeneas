/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                         Copyright (c) 1998                            */
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
/*                      Date   :  March 1998                             */
/*-----------------------------------------------------------------------*/
/*                  Features i/o                                         */
/*  This is kept separate from EST_Features to help reduce dependencies  */
/*=======================================================================*/
#include <cstdlib>
#include "EST_Features.h"
#include "ling_class/EST_Item.h"
#include "EST_error.h"
#include "EST_String.h"
#include "EST_Token.h"

void EST_Features::set_function(const EST_String &name, 
			        const EST_String &funcname)
{
    EST_Item_featfunc f = get_featfunc(funcname,1);

    set_path(name, est_val(f));
}

void EST_Features::save_fpair(ostream &outf,
			      const EST_String &fname,
			      const EST_Val &fvalue) const
{
    /* Feature valued features themselves (so can't denot empty ones) */
    if (fvalue.type() == val_type_feats)
    {
	EST_Features *f = feats(fvalue);
	if (f->features->list.head() == 0)
	{
	    // An empty feature set
	    outf << fname << " () ; ";
	}
	else
	    for (EST_Litem *q=f->features->list.head(); 
		 q != 0; q=q->next() )
		save_fpair(outf,
			   fname+"."+f->features->list(q).k,
			   f->features->list(q).v);
	return;
    }
    /* a non feature valued one */
    // in case someone has () in their feature names (ought to be shot)
    if (fname.contains("(") ||
	fname.contains(")") ||
	fname.contains(" ") ||    // bang, bang
	fname.contains("\t") ||   // what smoking gun ?
	fname.contains(";") ||
	(fname == ""))           
	outf << quote_string(fname,"\"","\\",1) << " ";
    else 
	outf << fname << " ";
    if (fvalue == ";")
	outf << "\";\"";
    else if ((fvalue.type() == val_string) &&
	     ((fvalue.string().matches(RXint)) ||
	      (fvalue.string().matches(RXdouble)) ||
	      (fvalue.string().contains("(")) ||
	      (fvalue.string().contains(")")) ||
	      (fvalue.string().contains(";")) ))
	// force quoting, cause it looks like a number but isn't
	outf << quote_string(fvalue.string(),"\"","\\",1);
    else if (fvalue.type() == val_float)
    {
	char b[20];
	sprintf(b,"%g",fvalue.Float());
	outf << b;
    }
    else if (fvalue.type() == val_type_featfunc)
    {
	outf << "F:"<<get_featname(featfunc(fvalue));
    }
    else
	outf << quote_string(fvalue.string());
    outf << " ; ";
}

EST_write_status EST_Features::save(ostream &outf) const
{
    // Save features
    if (features->list.head() == 0)
	outf << "()";
    else
	for (EST_Litem *p=features->list.head(); p != 0; p=p->next() )
	    save_fpair(outf,
		       features->list(p).k,
		       features->list(p).v);

    return write_ok;
}

EST_write_status EST_Features::save_sexpr(ostream &outf) const
{
    // Save features as an sexpression
    outf << "(";
    for (EST_Litem *p=features->list.head(); p != 0; p=p->next() )
    {
	const EST_String &fname = features->list(p).k;
	const EST_Val &fvalue = features->list(p).v;
	outf << "(";
	// in case someone has () in their feature names (ought to be shot)
	if (fname.contains("(") ||
	    fname.contains(")") ||
	    fname.contains(" ") ||
	    fname.contains("\t") ||
	    fname.contains(";"))
	    outf << quote_string(fname,"\"","\\",1);
	else
	    outf << fname;
	outf << " ";
	if (fvalue == ";")
	    outf << "\";\"";
	else if ((fvalue.type() == val_string) &&
		 ((fvalue.string().matches(RXint)) ||
		  (fvalue.string().matches(RXdouble)) ||
		  (fvalue.string().contains("(")) ||
		  (fvalue.string().contains(")"))))
	    // force quoting, cause it looks like a number but isn't
	    // or contains a paren
	    outf << quote_string(fvalue.string(),"\"","\\",1);
	else if (fvalue.type() == val_float)
	{
	    char b[20];
	    sprintf(b,"%g",fvalue.Float());
	    outf << b;
	}
	else if (fvalue.type() == val_type_featfunc)
	{
	    outf << "F:"<<get_featname(featfunc(fvalue));
	}
	else if (fvalue.type() == val_type_feats)
	{
	    feats(fvalue)->save_sexpr(outf);
	}
	else
	    outf << quote_string(fvalue.string());
	outf << ")";
	if (p->next())
	    outf << " ";
    }
    outf << ")";

    return write_ok;
}

EST_read_status EST_Features::load_sexpr(EST_TokenStream &ts)
{
    /* Load in feature structure from sexpression */

    if (ts.peek() != "(")
    {
	cerr << "load_features: no sexpression found\n";
	return misc_read_error;
    }
    else
    {
	EST_String f;
	EST_Token v;
	ts.get();  /* skip opening paren */
	for (; ts.peek() != ")"; )
	{
	    if (ts.peek() != "(")
	    {
		cerr << "load_features: no sexpression found\n";
		return misc_read_error;
	    }
	    ts.get();
	    f = ts.get().string();  /* feature name */
	    if ((ts.peek() == "(") && (ts.peek().quoted() == FALSE))
	    {
		EST_Features fv;
		set(f,fv);
		A(f).load_sexpr(ts);
	    }
	    else
	    {
		v = ts.get();
		if (v.quoted())
		    set(f,v.string());
		else if (v.string().matches(RXint))
		    set(f,atoi(v.string()));
		else if (v.string().matches(RXdouble))
		    set(f,atof(v.string()));
		else if (v.string().contains("F:"))
		{
		    EST_Item_featfunc func = 
			get_featfunc(v.string().after("F:"));
		    if (func != NULL)
			set_val(f,est_val(func));
		    else
		    {
			cerr << "load_features: Unknown Function '" << f <<"'\n";
			set_val(f,feature_default_value);
		    }
		}
		else
		    set(f,v.string());
		
	    }
	    if (ts.get() != ")")
	    {
		cerr << "load_features: no sexpression found\n";
		return misc_read_error;
	    }
	}
	if (ts.get() != ")")
	{
	    cerr << "load_features: no sexpression found\n";
	    return misc_read_error;
	}
    }
    return format_ok;
}

EST_read_status EST_Features::load(EST_TokenStream &ts)
{
    // load features from here to end of line separated by semicolons
    EST_String f;
    EST_Token v;
    static EST_Val val0 = EST_Val(0);
    
    while (!ts.eoln())
    {
	if (ts.eof())
	{
	    cerr << "load_features: unexpected end of file\n";
	    return misc_read_error;
	}
	f = ts.get().string();
	v = EST_String::Empty;
	while (((ts.peek() != ";") || (ts.peek().quoted())) &&
	       (!ts.eof()) && (!ts.eoln()))
	    if (v == "")
		v = ts.get();
	    else
		v = v.string()
		    + ts.peek().whitespace() 
			+ ts.get().string();
	if (v.quoted() || (v.string() == ""))
	    set_path(f,EST_Val(v.string()));
	else if (v.string() == "0")  // very common cases for speed
	    set_path(f,val0);
	else if ((strchr("0123456789-.",v.string()(0)) != NULL) &&
		 (v.string().matches(RXdouble)))
	{
	    if (v.string().matches(RXint))
	      set_path(f, EST_Val(atoi(v.string())));
	    else
	      set_path(f, EST_Val(atof(v.string())));
	}
	else if (v.string().contains("F:"))
	{
	    EST_Item_featfunc func = get_featfunc(v.string().after("F:"));
	    if (func != NULL)
		set_path(f, est_val(func));
	    else
	    {
		cerr << "load_features: Unknown Function '" << f <<"'\n";
		set_path(f, feature_default_value);
	    }
	}
	else if (v.string() == "()")
	{   // An empty feature set
	    EST_Features *fs = new EST_Features;
	    set_path(f,est_val(fs));
	}
	else if (v != "<contents>")  // unsupported type
	    set_path(f,EST_Val(v.string()));
	if (ts.peek() == ";")
	    ts.get();
	else if (!ts.eoln())
	{
	    cerr << "load_features: " << ts.pos_description() <<
		" missing semicolon in feature list\n";
	    return misc_read_error;
	}
    }
    return format_ok;
}

