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
 /*                   Date: Tue Apr  1 1997                              */
 /************************************************************************/
 /*                                                                      */
 /* Temporary bool type definition.                                      */
 /*                                                                      */
 /************************************************************************/

#ifndef __EST_BOOL_H__
#define __EST_BOOL_H__

#if defined(__GNUC__) ||  defined(SYSTEM_IS_WIN32)

  /* GCC seems to be so very fond of bool -- it's built into
   * the compiler and it chokes on my definition.
   */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef TRUE
#define TRUE (1==1)
#endif
#ifndef FALSE
#define FALSE (1==0)
#endif

#ifdef __cplusplus
}
#endif

#else /* __GNUC__ */

  /* For a boring type we still #define everything for code
   * which uses ifdef to see if bool is defined.
   */

#undef true
#undef false
#undef TRUE
#undef FALSE

#ifdef __cplusplus
#if 0

  class BoolType {
    
  private:
    int p_val;

  public:
    BoolType(int i) { p_val = i!=0;};
    BoolType() { p_val = 1==0;};

    operator int () const { return p_val; };

    BoolType operator == (BoolType b) const { return p_val == b.p_val;};
    BoolType operator != (BoolType b) const { return p_val != b.p_val;};

  };

#define true BoolType(1)
#define false BoolType(0)
#define TRUE BoolType(1)
#define FALSE BoolType(0)
#define bool BoolType

#else /* 0 */

/* Because SunCC is stupid we pretend we can't do better than we */
/* could with C.                                                 */
#if __SUNPRO_CC_COMPAT != 5
#define bool int
#endif
#define TRUE (1==1)
#define FALSE (1==0)
#define true TRUE
#define false FALSE

#endif

#else /* __cplusplus */

#define bool int
#define TRUE (1==1)
#define FALSE (1==0)

#endif /* __cplusplus */
#endif /* not __GNUC__ */

#endif
