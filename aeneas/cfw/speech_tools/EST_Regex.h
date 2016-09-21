 /************************************************************************/
 /*                                                                      */
 /*                Centre for Speech Technology Research                 */
 /*                     University of Edinburgh, UK                      */
 /*                        Copyright (c) 1997                            */
 /*                        All Rights Reserved.                          */
 /*                                                                      */
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

#ifndef __EST_REGEX_H__
#define __EST_REGEX_H__

class EST_Regex;

#include "EST_String.h"

/** A Regular expression class to go with the CSTR EST_String class. 
  *
  * The regular expression syntax is the FSF syntax used in emacs and
  * in the FSF String library. This is translated into the syntax supported
  * by Henry Spensor's regular expression library, this translation is a place
  * to look if you find regular expressions not matching where expected.
  *
  * @see EST_String
  * @see string_example
  * @author Richard Caley <rjc@cstr.ed.ac.uk>
  * @author (regular expression library by Henry Spencer, University of Toronto)
  * @version $Id: EST_Regex.h,v 1.3 2004/05/04 00:00:16 awb Exp $
  */

class EST_Regex : protected EST_String {

private:
    /// The compiled form.
    void *compiled;	
    /// Compiled form for whole string match.
    void *compiled_match;

protected:
    /// Compile expression.
    void compile();
    /// Compile expression in a form which only matches whole string.
    void compile_match();
    /// Translate the expression into the internally used syntax.
    char *regularize(int match) const;

public:
    /// Empty constructor, just for form.
    EST_Regex(void);

    /// Construct from EST_String.
    EST_Regex(EST_String s);

    /// Construct from C string.
    EST_Regex(const char *ex);

    /// Copy constructor.
    EST_Regex(const EST_Regex &ex);

    /// Destructor.
    ~EST_Regex();

    /// Size of the expression.
    int  size() const { return EST_String::size; };

    /// Run to find a matching substring
    int  run(const char *on, int from, int &start, int &end, int *starts=NULL, int *ends=NULL);
    /// Run to see if it matches the entire string.
    int  run_match(const char *on, int from=0, int *starts=NULL, int *ends=NULL);

    /// Get the expression as a string.
    EST_String tostring(void) const {return (*this);};

    /// Cast operator, disambiguates it for some compilers
    operator const char *() const { return (const char *)tostring(); }

    int operator == (const EST_Regex ex) const
    { return (const EST_String)*this == (const EST_String)ex; }

    int operator != (const EST_Regex ex) const
    { return (const EST_String)*this != (const EST_String)ex; }

    /**@name Assignment */
    //@{
    ///
    EST_Regex &operator = (const EST_Regex ex);
    ///
    EST_Regex &operator = (const EST_String s);
    ///
    EST_Regex &operator = (const char *s);
    //@}

    /// Stream output of regular expression.
    friend ostream &operator << (ostream &s, const EST_Regex &str);
};

ostream &operator << (ostream &s, const EST_Regex &str);

/**@name Predefined_regular_expressions
  * Some regular expressions matching common things are predefined
  */
//@{
/// White space
extern EST_Regex RXwhite;	// "[ \n\t\r]+"
/// Sequence of alphabetic characters.
extern EST_Regex RXalpha;	// "[A-Za-z]+"
/// Sequence of lower case alphabetic characters.
extern EST_Regex RXlowercase;	// "[a-z]+"
/// Sequence of upper case alphabetic characters.
extern EST_Regex RXuppercase;	// "[A-Z]+"
/// Sequence of letters and/or digits.
extern EST_Regex RXalphanum;	// "[0-9A-Za-z]+"
/// Initial letter or underscore followed by letters underscores or digits.
extern EST_Regex RXidentifier;	// "[A-Za-z_][0-9A-Za-z_]+"
/// Integer.
extern EST_Regex RXint;		// "-?[0-9]+"
/// Floating point number.
extern EST_Regex RXdouble;	// "-?\\(\\([0-9]+\\.[0-9]*\\)\\|\\([0-9]+\\)\\|\\(\\.[0-9]+\\)\\)\\([eE][---+]?[0-9]+\\)?"
//@}

// GCC lets us use the static constant to declare arrays, Sun CC
// doesn't, so for a quiet, if ugly, life we declare it here with a suitable
// value and check in EST_Regex.cc to make sure it`s OK

#define  EST_Regex_max_subexpressions 10

#endif	
