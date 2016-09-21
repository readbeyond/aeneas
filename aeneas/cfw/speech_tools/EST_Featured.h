 /************************************************************************/
 /*                                                                      */
 /*                Centre for Speech Technology Research                 */
 /*                     University of Edinburgh, UK                      */
 /*                       Copyright (c) 1996,1997                        */
 /*                        All Rights Reserved.                          */
 /*                                                                      */
 /*  Permission is hereby granted, free of charge, to use and distribute */
 /*  this software and its documentation without restriction, including  */
 /*  without limitation the rights to use, copy, modify, merge, publish, */
 /*  distribute, sublicense, and/or sell copies of this work, and to     */
 /*  permit persons to whom this work is furnished to do so, subject to  */
 /*  the following conditions:                                           */
 /*   1. The code must retain the above copyright notice, this list of   */
 /*      conditions and the following disclaimer.                        */
 /*   2. Any modifications must be clearly marked as such.               */
 /*   3. Original authors' names are not deleted.                        */
 /*   4. The authors' names are not used to endorse or promote products  */
 /*      derived from this software without specific prior written       */
 /*      permission.                                                     */
 /*                                                                      */
 /*  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK       */
 /*  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING     */
 /*  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT  */
 /*  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE    */
 /*  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES   */
 /*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN  */
 /*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,         */
 /*  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF      */
 /*  THIS SOFTWARE.                                                      */
 /*                                                                      */
 /*************************************************************************/


#ifndef __EST_FEATURED_H__
#define __EST_FEATURED_H__


#include "EST_Features.h"

/** A class with the mechanisms needed to give an object features and
  * access them nicely. Used as a parent class.
  * 
  * @author Richard Caley <rjc@cstr.ed.ac.uk>
  * @version $Id: EST_Featured.h,v 1.3 2004/05/04 00:00:16 awb Exp $ */

class EST_Featured {
private:
  
  EST_Features *p_features;

protected:

  EST_Featured(void);
  EST_Featured(const EST_Featured &f);
  ~EST_Featured(void);
  
  void init_features();

  void copy_features(const EST_Featured &f);

  void clear_features();

  void ensure_features(void) 
    { if (p_features==NULL) p_features= new EST_Features; }

public:

  int f_Int(const char *name, int def) const
    { return p_features?p_features->I(name, def):def; }
  int f_Int(const char *name) const
    { return p_features?p_features->I(name):0; }
  int f_I(const char *name, int def) const
    {return f_Int(name, def);}
  int f_I(const char *name) const
    {return f_Int(name);}
  void f_set(const EST_String name, int val)
    { ensure_features(); p_features->set(name, val); }
  void f_set_path(const EST_String name, int val)
    { ensure_features(); p_features->set_path(name, val); }
  

  float f_Float(const char *name, float def) const
    { return p_features?p_features->F(name, def):def; }
  float f_Float(const char *name) const
    { return p_features?p_features->F(name):0.0; }
  float f_F(const char *name, float def) const
    {return f_Float(name, def);}
  float f_F(const char *name) const
    {return f_Float(name);}
  void f_set(const EST_String name, float val)
    { ensure_features(); p_features->set(name, val); }
  void f_set_path(const EST_String name, float val)
    { ensure_features(); p_features->set_path(name, val); }
  

  EST_String f_String(const char *name, const EST_String &def) const
    { return p_features?p_features->S(name, def):def; }
  EST_String f_String(const char *name) const
    { return p_features?p_features->S(name):EST_String::Empty; }
  EST_String f_S(const char *name, const EST_String &def) const
    {return f_String(name, def);}
  EST_String f_S(const char *name) const
    {return f_String(name);}
  void f_set(const EST_String name, const char *val)
    { ensure_features(); p_features->set(name, val); }
  void f_set_path(const EST_String name, const char *val)
    { ensure_features(); p_features->set_path(name, val); }
  
  
  const EST_Val &f_Val(const char *name, const EST_Val &def) const;
  const EST_Val &f_Val(const char *name) const;

  const EST_Val &f_V(const char *name, const EST_Val &def) const
    {return f_Val(name, def);}
  const EST_Val &f_V(const char *name) const
    {return f_Val(name);}
  void f_set_val(const EST_String name, EST_Val val)
    { ensure_features(); p_features->set_val(name, val); }
  void f_set_path(const EST_String name, EST_Val val)
    { ensure_features(); p_features->set_path(name, val); }

  void f_set(const EST_Features &f)
    { ensure_features(); *p_features = f; }

  int f_present(const EST_String name) const
    {return p_features && p_features->present(name); }

  void f_remove(const EST_String name)
    { if (p_features) p_features->remove(name); }

  // iteration

  protected:
      struct IPointer_feat { EST_Features::RwEntries i;  };
//    struct IPointer_feat { EST_TRwStructIterator< EST_Features, EST_Features::IPointer, EST_Features::Entry> i;  };

    void point_to_first(IPointer_feat &ip) const 
      { if (p_features) ip.i.begin(*p_features);}
    void move_pointer_forwards(IPointer_feat &ip) const 
      { ++(ip.i); }
    bool points_to_something(const IPointer_feat &ip) const 
      { return ip.i != 0; }
    EST_TKVI<EST_String, EST_Val> &points_at(const IPointer_feat &ip) 
      { return *(ip.i); }

    friend class EST_TIterator< EST_Featured, IPointer_feat, EST_TKVI<EST_String, EST_Val> >;
    friend class EST_TStructIterator< EST_Featured, IPointer_feat, EST_TKVI<EST_String, EST_Val> >;
    friend class EST_TRwIterator< EST_Featured, IPointer_feat, EST_TKVI<EST_String, EST_Val> >;
    friend class EST_TRwStructIterator< EST_Featured, IPointer_feat, EST_TKVI<EST_String, EST_Val> >;

public:
  typedef EST_TKVI<EST_String, EST_Val> FeatEntry;
  typedef EST_TStructIterator< EST_Featured, IPointer_feat, FeatEntry> FeatEntries;
  typedef EST_TRwStructIterator< EST_Featured, IPointer_feat, FeatEntry> RwFeatEntries;

  

};

#endif

