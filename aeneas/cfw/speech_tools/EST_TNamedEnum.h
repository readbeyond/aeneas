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
 /************************************************************************/
 /*                 Author: Richard Caley (rjc@cstr.ed.ac.uk)            */
 /*                   Date: Fri Feb 28 1997                              */
 /************************************************************************/
 /*                                                                      */
 /* A template class which allows names (const char *s) to be            */
 /* associated with enums, providing conversion.                         */
 /*                                                                      */
 /* EST_TValuesEnum is the obvious generalisation to associating         */
 /* things other than const char * with each value.                      */
 /*                                                                      */
 /* EST_T{Named/Valued}EnumI can include extra information for each      */
 /* enum element.                                                        */
 /*                                                                      */
 /* This should be rewritten as something other than linear search. At   */
 /* least sort them.                                                     */
 /*                                                                      */
 /************************************************************************/

#ifndef __EST_TNAMEDENUM_H__
#define __EST_TNAMEDENUM_H__

#include <cstring>

using namespace std;

#include "EST_String.h"
#include "EST_rw_status.h"

#define NAMED_ENUM_MAX_SYNONYMS (10)

// Used in the type of tables with no info field.

typedef char NO_INFO;

// struct used to define the mapping.

template<class ENUM, class VAL, class INFO> 
struct EST_TValuedEnumDefinition {
public:
    ENUM token; 
    VAL values[NAMED_ENUM_MAX_SYNONYMS];
    INFO info;
} ;

// This is the most general case, a mapping from enum to some other type
// with extra info.

template<class ENUM, class VAL, class INFO> class EST_TValuedEnumI  {

protected:
  int ndefinitions;
  ENUM p_unknown_enum;
  VAL p_unknown_value;
  EST_TValuedEnumDefinition<ENUM,VAL,INFO> *definitions;

  virtual int eq_vals(VAL v1, VAL v2) const {return v1 == v2; };
  // This is only a void * because INFO can`t manage to get the
  // parameter declaration in the definition past gcc with the actual type.
  void initialise(const void *defs);
  void initialise(const void *defs, ENUM (*conv)(const char *));
  void initialise(void) {ndefinitions=0; definitions=NULL;};
  void initialise(ENUM unknown_e, VAL unknown_v) {initialise(); p_unknown_enum=unknown_e; p_unknown_value = unknown_v;};

protected:
  EST_TValuedEnumI(void) {initialise();};

public:
  EST_TValuedEnumI(EST_TValuedEnumDefinition<ENUM,VAL,INFO> defs[]) 
	{initialise((const void *)defs); };
  EST_TValuedEnumI(EST_TValuedEnumDefinition<const char *,VAL,INFO> defs[], ENUM (*conv)(const char *)) 
	{initialise((const void *)defs, conv); };
  virtual ~EST_TValuedEnumI(void);

  int n(void) const;

  ENUM token(VAL value) const;
  ENUM token(int n) const { return nth_token(n); }
  ENUM nth_token(int n) const;
  VAL value(ENUM token, int n=0) const;
  INFO &info(ENUM token) const;

  ENUM unknown_enum(void) const {return p_unknown_enum;};
  VAL  unknown_value(void) const {return p_unknown_value;};
  int  valid(ENUM token) const { return !eq_vals(value(token),p_unknown_value); };
};

// This is a special case for names. This saves typing and also
// takes care of the fact that strings need their own compare function.

template<class ENUM, class INFO> class EST_TNamedEnumI  : public EST_TValuedEnumI<ENUM, const char *, INFO> {

protected:
  EST_TNamedEnumI(void) : EST_TValuedEnumI<ENUM, const char *, INFO>() {};
  int eq_vals(const char *v1, const char *v2) const {return strcmp(v1,v2) ==0; };
public:

  EST_TNamedEnumI(EST_TValuedEnumDefinition<ENUM,const char *,INFO> defs[])
	{this->initialise((const void *)defs); };
  EST_TNamedEnumI(EST_TValuedEnumDefinition<const char *,const char *,INFO> defs[], ENUM (*conv)(const char *))
	{this->initialise((const void *)defs, conv); };
  const char *name(ENUM tok, int n=0) const {return this->value(tok,n); };

};

// Now the simple cases with no extra information

template<class ENUM, class VAL> class EST_TValuedEnum : public EST_TValuedEnumI<ENUM,VAL,NO_INFO> { 
public:
  EST_TValuedEnum(EST_TValuedEnumDefinition<ENUM,VAL,NO_INFO> defs[]) 
	{this->initialise((const void *)defs);};
  EST_TValuedEnum(EST_TValuedEnumDefinition<const char *,VAL,NO_INFO> defs[], ENUM (*conv)(const char *)) 
	{this->initialise((const void *)defs, conv);};
};


template<class ENUM> class EST_TNamedEnum : public EST_TNamedEnumI<ENUM,NO_INFO> { 
private:
  EST_read_status priv_load(EST_String name, EST_TNamedEnum *definitive);
  EST_write_status priv_save(EST_String name, EST_TNamedEnum *definitive, char quote) const;
public:
  EST_TNamedEnum(ENUM undef_e, const char *undef_n = NULL) 
	{this->initialise(undef_e, undef_n);};
  EST_TNamedEnum(EST_TValuedEnumDefinition<ENUM,const char *,NO_INFO> defs[]) 
	{this->initialise((const void *)defs);};
  EST_TNamedEnum(EST_TValuedEnumDefinition<const char *,const char *,NO_INFO> defs[], ENUM (*conv)(const char *)) 
	{this->initialise((const void *)defs, conv);};

  EST_read_status load(EST_String name) { return priv_load(name, NULL); };
  EST_read_status load(EST_String name, EST_TNamedEnum &definitive) { return priv_load(name, &definitive); };
  EST_write_status save(EST_String name, char quote='"') const { return priv_save(name, NULL, quote); };
  EST_write_status save(EST_String name, EST_TNamedEnum &definitive, char quote='"') const { return priv_save(name, &definitive, quote); };

};

#include "instantiate/EST_TNamedEnumI.h"

#endif
