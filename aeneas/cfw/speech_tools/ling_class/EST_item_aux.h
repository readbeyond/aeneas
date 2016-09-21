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


#ifndef __EST_ITEM_AUX_H__
#define __EST_ITEM_AUX_H__

/** Non core feature funtionality.
  * 
  * @author Richard Caley <rjc@cstr.ed.ac.uk>
  * @version $Id: EST_item_aux.h,v 1.3 2004/05/24 10:32:30 korin Exp $
  */

//@{

#include "EST_features_aux.h"

class EST_Item;
class EST_String;

/** Safe feature access functions.
  * 
  * These functions are guaranteed to return a value even if
  * there is an otherwise fatal error.
  */
//@{

/// Return the value as an EST_Val.
EST_Val getVal(const EST_Item &f,
	       const EST_String name,
	       const EST_Val &def,
	       EST_feat_status &s);

/// Return the value as a string.
EST_String getString(const EST_Item &f,
		     const EST_String name,
		     const EST_String &def,
		     EST_feat_status &s);

/// Return the values as a float.
float getFloat(const EST_Item &f,
	       const EST_String name,
	       const float &def,
	       EST_feat_status &s);

/// Return the values as an integer.
int getInteger(const EST_Item &f,
	       const EST_String name,
	       const int &def,
	       EST_feat_status &s);
//@}
		     
float start(const EST_Item &item);
float mid(const EST_Item &item);
float time(const EST_Item &item);
float end(const EST_Item &item);


//@}

//Rob's function for jumping around HRG structures more easily
EST_Item *item_jump(EST_Item *from, const EST_String &to);


#endif

