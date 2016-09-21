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
/*                    Author :  Alan W Black                             */
/*                    Date   :  May 1996                                 */
/*-----------------------------------------------------------------------*/
/*                                                                       */
/* A generic container class, originally for ints floats and string now  */
/* extended for some others, eventually allow run addition of new types  */
/* "built-in" types (i.e. ones explicitly mentioned in this file) may    */
/* be accessed by member functions, objects added at run time may only   */
/* be accessed by functions                                              */
/*                                                                       */
/* This is so similar to the LISP class in SIOD it could be viewed as a  */
/* little embarrassing, but this is done without a cons cell heap or gc  */
/* which may or may not be a good thing.                                 */
/*                                                                       */
/*=======================================================================*/
#ifndef __EST_VAL_H__
#define __EST_VAL_H__

#include "EST_String.h"
#include "EST_error.h"
#include "EST_Contents.h"
#include "EST_Val_defs.h"

typedef const char *val_type;

extern val_type val_unset;
extern val_type val_int;
extern val_type val_float;
extern val_type val_string;

/** The EST_Val class is a container class, used to store a single
    item which can be an int, float, string or other user-defined
    class. It is often used as the base item in the <link
    linkend="est-features">EST_Features</link> class, to enable features
    to take on values of different types.
*/

class EST_Val {
  private:
    val_type t;
    union 
    { int ival;
      float fval; 
      EST_Contents *pval;} v;
    // * may have a string name as well as a value
    EST_String sval;
    const int to_int() const;
    const float to_flt() const;
    const EST_String &to_str() const;
  public:
    /**@name Constructor and Destructor functions
     */

    //@{
    /** Default constructor */
    EST_Val() 
	{t=val_unset;}

    /** Copy constructor for another EST_Val*/
    EST_Val(const EST_Val &val);

    /** Copy constructor for an int*/
    EST_Val(const int i) 
	{t=val_int; v.ival=i;}

    /** Copy constructor for a float*/
    EST_Val(const float f) 
	{t=val_float; v.fval=f;}

    /** Copy constructor for a double*/
    EST_Val(const double d) {t=val_float; v.fval=d;}

    /** Copy constructor for a string*/
    //    EST_Val(const EST_String &s) {t=val_string; sval = s;}
    EST_Val(const EST_String &s) : t(val_string), sval(s) {};

    /** Copy constructor for a string literal*/
    //    EST_Val(const char *s) {t=val_string; sval = s;}
    EST_Val(const char *s) : t(val_string), sval(s) {};

    EST_Val(val_type type,void *p, void (*f)(void *));

    /** Destructor */
    ~EST_Val(void);

    //@}

    /**@name Getting cast values
     */

    //@{

    /** returns the type that the val is currently holding */
    const val_type type(void) const 
	{return t;}
    
    /** returns the value, cast as an int */
    const int Int(void) const 
	{if (t==val_int) return v.ival; return to_int();}

    /** returns the value, cast as an int */
    const int I(void) const 
	{ return Int(); }

    /** returns the value, cast as a float */
    const float Float(void) const 
	{if (t==val_float) return v.fval; return to_flt();}

    /** returns the value, cast as a float */
    const float F(void) const 
	{ return Float(); }

    /** returns the value, cast as a string */
    const EST_String &String(void) const
       {if (t!=val_string) to_str(); return sval;}

    /** returns the value, cast as a string */
    const EST_String &string(void) const
       {return String();}

    /** returns the value, cast as a string */
    const EST_String &S(void) const
       {return String();}

    /** returns the value, cast as a string */
    const EST_String &string_only(void) const {return sval;}

    //@}

    // Humans should never call this only automatic functions
    const void *internal_ptr(void) const
	{ return v.pval->get_contents(); }

    /**@name Setting values
     */

    //@{

    /** Assignment of val to an int */
    EST_Val &operator=(const int i) { t=val_int; v.ival=i; return *this;}

    /** Assignment of val to a float */
    EST_Val &operator=(const float f) { t=val_float; v.fval=f; return *this;}

    /** Assignment of val to a double */
    EST_Val &operator=(const double d) { t=val_float; v.fval=d; return *this;}

    /** Assignment of val to a string */
    EST_Val &operator=(const EST_String &s) { t=val_string; sval = s; return *this;}

    /** Assignment of val to a string literal*/
    EST_Val &operator=(const char *s) { t=val_string; sval = s; return *this;}

    /** Assignment of val to another val*/
    EST_Val &operator=(const EST_Val &c);

    //@}

    /**@name Equivalence test
     */

    //@{


    /** Test whether val is equal to a*/
    int operator ==(const EST_Val &a) const
    { if (t != a.t) return (1==0);
      else if (t == val_string) return (sval == a.sval);
      else if (t == val_int) return (v.ival == a.v.ival);
      else if (t == val_float) return (v.fval == a.v.fval);
      else return (internal_ptr() == a.internal_ptr()); }

    /** Test whether val is equal to the string a*/
    int operator ==(const EST_String &a) const { return (string() == a); }
    /** Test whether val is equal to the char * a*/
    int operator ==(const char *a) const { return (string() == a); }
    /** Test whether val is equal to the int a*/
    int operator ==(const int &i) const { return (Int() == i); }
    /** Test whether val is equal to the float a*/
    int operator ==(const float &f) const { return (Float() == f); }
    /** Test whether val is equal to the double a*/
    int operator ==(const double &d) const { return (Float() == d); }


    /** Test whether val is not equal to the val a*/
    int operator !=(const EST_Val &a) const { return (!(*this == a)); }
    /** Test whether val is not equal to the string a*/
    int operator !=(const EST_String &a) const { return (string() != a); }
    /** Test whether val is not equal to the char * a*/
    int operator !=(const char *a) const { return (string() != a); }
    /** Test whether val is not equal to the int a*/
    int operator !=(const int &i) const { return (Int() != i);}
    /** Test whether val is not equal to the float a*/
    int operator !=(const float &f) const { return (Float() != f); }
    /** Test whether val is not equal to the double float a*/
    int operator !=(const double &d) const { return (Float() != d); }

    //@{

    /**@name Automatic casting 
     */
    //@{

    /** Automatically cast val as an int*/
    operator int() const { return Int(); }
    /** Automatically cast val as an float*/
    operator float() const { return Float(); }
    /** Automatically cast val as an string*/
    operator EST_String() const { return string(); }
    //@}
    /** print val*/
    friend ostream& operator << (ostream &s, const EST_Val &a)
    { if (a.type() == val_unset) s << "[VAL unset]" ;
      else if (a.type() == val_int)	  s << a.v.ival;
      else if (a.type() == val_float)  s << a.v.fval;
      else if (a.type() == val_string) s << a.sval;
      else  s << "[PVAL " << a.type() << "]";
      return s;
    }
};

inline const char *error_name(const EST_Val val) { return (EST_String)val;}

// For consistency with other (user-defined) types in val
inline EST_Val est_val(const EST_String s) { return EST_Val(s); }
inline EST_Val est_val(const char *s) { return EST_Val(s); }
inline int Int(const EST_Val &v) { return v.Int(); }
inline EST_Val est_val(const int i) { return EST_Val(i); }
inline float Float(const EST_Val &v) { return v.Float(); }
inline EST_Val est_val(const float f) { return EST_Val(f); }

#endif 
