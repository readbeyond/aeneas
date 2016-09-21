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
/*                    Author :  Alan W Black                             */
/*                    Date   :  March 1998                               */
/*-----------------------------------------------------------------------*/
/*   A class for feature value pairs                                     */
/*=======================================================================*/
#ifndef __EST_FEATURES_H__
#define __EST_FEATURES_H__

#include "EST_TKVL.h"
#include "EST_Val.h"
#include "EST_types.h"
#include "EST_TIterator.h"
#include "EST_error.h"

class EST_TokenStream;
class EST_String;

VAL_REGISTER_CLASS_DCLS(feats,EST_Features)

// This shouldn't be here and only is for older code
typedef EST_Val (*EST_Item_featfunc)(class EST_Item *);
EST_Val est_val(const EST_Item_featfunc f);


/** A class for containing feature structures which can hold atomic
values (int, float, string) or other feature structures.
*/


class EST_Features {
 protected:
    EST_TKVL<EST_String, EST_Val> *features;

    void save_fpair(ostream &outf,
		    const EST_String &fname,
		    const EST_Val &fvalue) const;
 public:
    static EST_Val feature_default_value;
    EST_Features();
    EST_Features(const EST_Features &f);
    ~EST_Features();

    /**@name Access functions which return EST_Val. 
       Features can
       either be simple features, in which their name is the name of
       an plain attribute (e.g. "name"), or path features where their
       name is a dot separated path of concatenated attributes
       (e.g. "df.poa.alveolar").  
    */
    //@{
    /** Look up directly without decomposing name as path (just simple feature)
     */
    const EST_Val &val(const char *name) const;

    /** Look up directly without decomposing name as path (just simple feature),
	returning <parameter>def</parameter> if not found
    */
    const EST_Val &val(const char *name, const EST_Val &def) const;

    /** Look up feature name, which may be simple feature or path
     */
    const EST_Val &val_path(const EST_String &path) const;

    /** Look up feature name, which may be simple feature or path,
	returning <parameter>def</parameter> if not found
     */
    const EST_Val &val_path(const EST_String &path, const EST_Val &def) const;

    /** Look up feature name, which may be simple feature or path.
     */
    const EST_Val &operator() (const EST_String &path) const 
       {return val_path(path);}

    /** Look up feature name, which may be simple feature or path,
	returning <parameter>def</parameter> if not found
     */
    const EST_Val &operator() (const EST_String &path, const EST_Val &def) const 
       {return val_path(path, def);}

    /** Look up feature name, which may be simple feature or path.
     */
    const EST_Val &f(const EST_String &path) const
       { return val_path(path); }

    /** Look up feature name, which may be simple feature or path,
	returning <parameter>def</parameter> if not found
     */
    const EST_Val &f(const EST_String &path, const EST_Val &def) const
       { return val_path(path,def); }
    //@}

    /**@name Access functions which return types. 
       These functions cast
       their EST_Val return value to a requested type, either float,
       int, string or features (A). In all cases the name can be a
       simple feature or a path, in which case their name is a dot
       separated string of concatenated attributes
       (e.g. "df.poa.alveolar").  */
    //@{

    /** Look up feature name, which may be simple feature or path, and return
	as a float */
    const float F(const EST_String &path) const
       {return val_path(path).Float(); }

    /** Look up feature name, which may be simple feature or path, and
	return as a float, returning <parameter>def</parameter> if not
	found */
    const float F(const EST_String &path, float def) const
       {return val_path(path, def).Float(); }

    /** Look up feature name, which may be simple feature or path, and return
	as an int */
    const int I(const EST_String &path) const
       {return val_path(path).Int(); }

    /** Look up feature name, which may be simple feature or path, and
	return as an int, returning <parameter>def</parameter> if not
	found */
    const int I(const EST_String &path, int def) const
       {return val_path(path, def).Int(); }

    /** Look up feature name, which may be simple feature or path, and return
	as a EST_String */
    const EST_String S(const EST_String &path) const
       {return val_path(path).string(); }

    /** Look up feature name, which may be simple feature or path, and
	return as a EST_String, returning <parameter>def</parameter> if not
	found */

    const EST_String S(const EST_String &path, const EST_String &def) const
       {return val_path(path, def).string(); }

    /** Look up feature name, which may be simple feature or path, and return
	as a EST_Features */
    EST_Features &A(const EST_String &path) const
       {return *feats(val_path(path));}

    /** Look up feature name, which may be simple feature or path, and
	return as a EST_Features, returning <parameter>def</parameter> if not
	found */
    EST_Features &A(const EST_String &path, EST_Features &def) const;

    //@}

    /**@name Setting features
     */
    //@{
    /** Add a new feature or set an existing feature <parameter>name<parameter>
	to value <parameter>ival</parameter>
    */
    void set(const EST_String &name, int ival)
	{ EST_Val pv(ival); set_path(name, pv);}

    /** Add a new feature or set an existing feature <parameter>name<parameter>
	to value <parameter>fval</parameter>
    */
    void set(const EST_String &name, float fval)
	{ EST_Val pv(fval); set_path(name, pv); }

    /** Add a new feature or set an existing feature <parameter>name<parameter>
	to value <parameter>dval</parameter>
    */
    void set(const EST_String &name, double dval)
	{ EST_Val pv((float)dval); set_path(name, pv); }

    /** Add a new feature or set an existing feature <parameter>name<parameter>
	to value <parameter>sval</parameter>
    */
    void set(const EST_String &name, const EST_String &sval)
	{ EST_Val pv(sval); set_path(name, pv); }

    /** Add a new feature or set an existing feature <parameter>name<parameter>
	to value <parameter>cval</parameter>
    */
    void set(const EST_String &name, const char *cval)
	{ EST_Val pv(cval); set_path(name, pv); }

    /** Add a new feature or set an existing feature <parameter>name<parameter>
	to value <parameter>val<parameter>.  <parameter>Name<parameter> must be
	not be a path.
    */
    void set_val(const EST_String &name, const EST_Val &sval)
	{ features->add_item(name,sval); }

    /** Add a new feature or set an existing feature <parameter>name<parameter>
	to value <parameter>val<parameter>, where <parameter>name<parameter>
	is a path.
    */
    void set_path(const EST_String &name, const EST_Val &sval);

    /** Add a new feature feature or set an existing feature
	<parameter>name<parameter> to value <parameter>f</parameter>, which
	is the named of a registered feature function.
    */
    void set_function(const EST_String &name, const EST_String &f);

    /** Add a new feature or set an existing feature
	<parameter>name<parameter> to value <parameter>f</parameter>,
	which itself is a EST_Features.  The information in
	<parameter>f</parameter> is copied into the features.  */
    void set(const EST_String &name, EST_Features &f)
	{ EST_Features *ff = new EST_Features(f);
	    set_path(name, est_val(ff)); }

    //@}

    /**@name Utility functions
     */

    //@{
    /** remove the named feature */
    void remove(const EST_String &name)
    { features->remove_item(name,1); }

    /** number of features in feature structure */
    int length() const { return features->length(); }

    /** return 1 if the feature is present */
    int present(const EST_String &name) const;

    /** Delete all features from object */
    void clear() { features->clear(); }

    /** Feature assignment */
    EST_Features& operator = (const EST_Features& a);
    /** Print Features */
    friend ostream& operator << (ostream &s, const EST_Features &f)
        { f.save(s); return s; }
    //@}


    // Iteration
#if 0
    EST_Litem *head() const { return features->list.head(); }
    EST_String &fname(EST_Litem *p) const { return features->list(p).k; }
    EST_Val &val(EST_Litem *p) const { return features->list(p).v; }
    float F(EST_Litem *p) const { return features->list(p).v.Float(); }
    EST_String S(EST_Litem *p) const { return features->list(p).v.string(); }
    int I(EST_Litem *p) const { return features->list(p).v.Int(); }
    EST_Features &A(EST_Litem *p) { return *feats(features->list(p).v); }
#endif



  protected:
    struct IPointer { EST_TKVL<EST_String, EST_Val>::RwEntries i;  };

    void point_to_first(IPointer &ip) const 
      { ip.i.begin(*features);}
    void move_pointer_forwards(IPointer &ip) const 
      { ++(ip.i); }
    bool points_to_something(const IPointer &ip) const 
      { return ip.i != 0; }
    EST_TKVI<EST_String, EST_Val> &points_at(const IPointer &ip) 
      { return *(ip.i); }

    friend class EST_TIterator< EST_Features, IPointer, EST_TKVI<EST_String, EST_Val> >;
    friend class EST_TStructIterator< EST_Features, IPointer, EST_TKVI<EST_String, EST_Val> >;
    friend class EST_TRwIterator< EST_Features, IPointer, EST_TKVI<EST_String, EST_Val> >;
    friend class EST_TRwStructIterator< EST_Features, IPointer, EST_TKVI<EST_String, EST_Val> >;

public:

    /**@name Iteration
     */

    //@{
    typedef EST_TKVI<EST_String, EST_Val> Entry;
    typedef EST_TStructIterator< EST_Features, IPointer, Entry> Entries;
    typedef EST_TRwStructIterator< EST_Features, IPointer, Entry> RwEntries;
    //@}

    /**@name File I/O
     */

    //@{
    /// load features from already opened EST_TokenStream
    EST_read_status load(EST_TokenStream &ts);
    /// load features from sexpression, contained in already opened EST_TokenStream 
    EST_read_status load_sexpr(EST_TokenStream &ts);
    /// save features in already opened ostream
    EST_write_status save(ostream &outf) const;
    /// save features as s-expression in already opened ostream
    EST_write_status save_sexpr(ostream &outf) const;

    //@}
};

inline bool operator == (const EST_Features &a,const EST_Features &b) 
{(void)a; (void)b; return false;}

void merge_features(EST_Features &to,EST_Features &from);
EST_String error_name(const EST_Features &a);

#endif

