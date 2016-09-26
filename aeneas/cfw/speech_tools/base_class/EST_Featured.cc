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
 /*                                                                       */
 /*                 Author: Richard Caley (rjc@cstr.ed.ac.uk)             */
 /*                                                                       */
 /* --------------------------------------------------------------------  */
 /* A class encapsulating the mechanics of objects which have features.   */
 /*                                                                       */
 /*************************************************************************/

#include "EST_Featured.h"


EST_Featured::EST_Featured(void)
{
  init_features();
}

EST_Featured::EST_Featured(const EST_Featured &f)
{
  init_features();
  copy_features(f);
}


EST_Featured::~EST_Featured(void)
{
  clear_features();
}

void EST_Featured::init_features()
{
  p_features=NULL;
}

void EST_Featured::clear_features()
{
  if (p_features)
    {
      delete p_features;
      p_features=NULL;
    }
  init_features();
}

const EST_Val &EST_Featured::f_Val(const char *name) const
{ 
    if (p_features)
	return p_features->val(name);
    else
	return EST_Features::feature_default_value;
}

const EST_Val &EST_Featured::f_Val(const char *name, const EST_Val &def) const
{ 
    if (p_features)
	return p_features->val(name);
    else
	return def;
}

void EST_Featured::copy_features(const EST_Featured &f)
{
  clear_features();

  if (f.p_features)
    p_features = new EST_Features(*(f.p_features));
}

#if defined(INSTANTIATE_TEMPLATES)

typedef EST_TKVI<EST_String, EST_Val> EST_Featured_Entry;
Instantiate_TStructIterator_T(EST_Featured, EST_Featured::IPointer_feat,EST_Featured_Entry, Featured_itt)
Instantiate_TIterator_T(EST_Featured, EST_Featured::IPointer_feat,EST_Featured_Entry, Featured_itt)
#endif
