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

#ifndef __EST_PATHNAME_H__
#define __EST_PATHNAME_H__

#include "EST_String.h"
#include "EST_TList.h"

/** Class representing pathnames. Makes common filename manipulations 
  * available as methods. Different implementations are provided for 
  * different systems.
  */
class EST_Pathname : public EST_String {

private:
  void setup(void);

public:

  EST_Pathname(void) : EST_String("") { };
  EST_Pathname(EST_String s) : EST_String(s) { this->setup(); };
  EST_Pathname(const char *s) : EST_String(s) { this->setup(); };

  static EST_Pathname construct(EST_Pathname dir, EST_String basename, EST_String extension);
  static EST_Pathname construct(EST_Pathname dir, EST_String filename);

  // component parts of a filename
  EST_Pathname directory(void) const;
  EST_Pathname filename(void) const;
  EST_String basename(int remove_all=0) const;
  EST_String extension(void) const;

  EST_Pathname as_file(void) const;
  EST_Pathname as_directory(void) const;

  int is_absolute(void) const;
  inline int is_relative(void) const {return !is_absolute();};
  int is_dirname(void) const;
  inline int is_filename(void) const {return !is_dirname(); };

  EST_TList<EST_String> entries(int check_for_directories = 1) const;

  static EST_Pathname append(EST_Pathname directory, EST_Pathname addition);

  static void divide(EST_Pathname path, int at, EST_Pathname &start, EST_Pathname &end);

  friend EST_Pathname operator + (const EST_Pathname p, const EST_Pathname addition);
  friend EST_Pathname operator + (const char *p, const EST_Pathname addition);

  // solve an ambiguity
  EST_Pathname &operator += (const char * addition) 
    { return (*this) = append(*this,  addition); }
  EST_Pathname &operator += (const EST_String addition) 
    { return (*this) = append(*this,  addition); }
  EST_Pathname &operator += (const EST_Pathname addition) 
    { return (*this) = append(*this,  addition); }

  EST_Pathname operator + (const EST_String addition) 
    { return append(*this, EST_Pathname(addition)); }
  EST_Pathname operator + (const char *addition) 
    { return append(*this,  EST_Pathname(addition)); }
  
  int operator == (const EST_String thing) 
    { return EST_String(*this) == thing; }
  int operator == (const char * thing) 
    { return EST_String(*this) == EST_String(thing); }
  int operator != (const EST_String thing) 
    { return EST_String(*this) != thing; }
  int operator != (const char * thing) 
    { return EST_String(*this) != EST_String(thing); }
  
};

EST_Pathname operator + (const EST_Pathname p, const EST_Pathname addition);
EST_Pathname operator + (const char *p, const EST_Pathname addition);

#endif
