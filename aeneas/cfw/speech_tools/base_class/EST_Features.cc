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
/*                  Feature value pairs in a EST_TKVL                    */
/*=======================================================================*/

#include <cstdlib>
#include "EST_Features.h"
#include "ling_class/EST_Item.h"
#include "EST_error.h"
#include "EST_String.h"
#include "EST_Token.h"

/* Features may used as Vals */
VAL_REGISTER_CLASS(feats,EST_Features)

EST_Val EST_Features::feature_default_value("0");
// Sometimes we need a set of features even though there aren't any
static EST_Features default_features;

EST_Features::EST_Features()
{
    features = new EST_TKVL<EST_String, EST_Val>;
}

EST_Features::EST_Features(const EST_Features &f)
{
    features = new EST_TKVL<EST_String, EST_Val>;
    *features = *f.features;
}

EST_Features::~EST_Features()
{
  if (features != NULL)
    {
      delete features;
      features=NULL;
    }
}

const EST_Val &EST_Features::val(const char *name, const EST_Val &def) const
{
    // Because so many access are from char* literals we all access
    // directly rather than requiring the creation of an EST_String
    EST_Litem *p;

    for (p=features->list.head(); p; p=p->next())
    {
	if (features->list(p).k == name)
	    return features->list(p).v;
    }
    return def;
}

const EST_Val &EST_Features::val(const char *name) const
{
    // Because so many access are from char* literals we all access
    // directly rather than requiring the creation of an EST_String
    EST_Litem *p;

    for (p=features->list.head(); p; p=p->next())
    {
	if (features->list(p).k == name)
	    return features->list(p).v;
    }

    EST_error("{FND} Feature %s not defined\n", name);
    return feature_default_value;
}

const EST_Val &EST_Features::val_path(const EST_String &name, const EST_Val &d) const
{
    // For when name contains references to sub-features
    
    if (strchr(name,'.') == NULL)
	return val(name, d);
    else
    {
	EST_String nname = name;
	EST_String fname = nname.before(".");
	const EST_Val &v = val(fname, d);
	if (v.type() == val_type_feats)
	    return feats(v)->val_path(nname.after("."), d);
	else
	    return d;
    }
}

const EST_Val &EST_Features::val_path(const EST_String &name) const
{
    // For when name contains references to sub-features
    
    if (strchr(name,'.') == NULL)
	return val(name);
    else
    {
	EST_String nname = name;
	EST_String fname = nname.before(".");
	const EST_Val &v = val(fname);
	if (v.type() == val_type_feats)
	    return feats(v)->val_path(nname.after("."));
	else
	    EST_error("Feature %s not feature valued\n", (const char *)fname);
	return feature_default_value; // wont get here 
    }
}

EST_Features &EST_Features::A(const EST_String &path,EST_Features &def) const
{
    EST_Features *ff = new EST_Features(def);

    return *feats(val(path,est_val(ff)));
}

int EST_Features::present(const EST_String &name) const
{
    if (strchr(name,'.') == NULL)
	return features->present(name);
    EST_String nname = name;
    if (features->present(nname.before(".")))
    {
	const EST_Val &v = val(nname.before("."));
	if (v.type() == val_type_feats)
	    return feats(v)->present(nname.after("."));
	else
	    return FALSE;
    }
    else
	return FALSE;
}

void EST_Features::set_path(const EST_String &name, const EST_Val &sval)
{
    // Builds sub features (if necessary)
    
    if (strchr(name,'.') == NULL)
	set_val(name,sval);
    else
    {
	EST_String nname = name;
	EST_String fname = nname.before(".");
	if (present(fname))
	{
	    const EST_Val &v = val(fname);
	    if (v.type() == val_type_feats)
		feats(v)->set_path(nname.after("."),sval);
	    else
		EST_error("Feature %s not feature valued\n", 
			  (const char *)fname);
	}
	else
	{
	    EST_Features f;
	    set(fname,f);
	    A(fname).set_path(nname.after("."),sval);
	}
    }
}

EST_Features &EST_Features::operator=(const EST_Features &x)
{
    *features = *x.features;
    return *this;
}

void merge_features(EST_Features &to,EST_Features &from)
{
  EST_Features::Entries p;

  for(p.begin(from); p; ++p)
    to.set_val(p->k,p->v);
}


EST_String error_name(const EST_Features &a)
{
    (void)a;
    return "<<Features>>";
}

#if defined(INSTANTIATE_TEMPLATES)
typedef EST_TKVI<EST_String, EST_Val> EST_Features_Entry;
Instantiate_TStructIterator_T(EST_Features, EST_Features::IPointer,  EST_Features_Entry, Features_itt)
Instantiate_TIterator_T(EST_Features, EST_Features::IPointer,  EST_Features_Entry, Features_itt)
#endif
