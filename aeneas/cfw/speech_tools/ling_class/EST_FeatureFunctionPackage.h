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


#ifndef __EST_FEATUREFUNCTIONPACKAGE_H__
#define __EST_FEATUREFUNCTIONPACKAGE_H__

/** Represents a named collection of named functions.
  * 
  * @author Richard Caley <rjc@cstr.ed.ac.uk>
  * @version $Id: EST_FeatureFunctionPackage.h,v 1.2 2001/04/04 13:11:27 awb Exp $
  */

#include "ling_class/EST_Item.h"
#include "EST_THash.h"


class EST_FeatureFunctionPackage {
public:
	struct Entry
	{
	EST_Item_featfunc func;
	};	
private:
	const EST_String p_name;
	EST_TStringHash<Entry> p_entries;


public:
	EST_FeatureFunctionPackage(const EST_String name, int n);
	EST_FeatureFunctionPackage(const char *name, int n);

	const EST_String name(void) { return p_name; }

	void register_func(const EST_String &name, const EST_Item_featfunc func);

	const Entry &lookup(const EST_String &name, int &found) const;

	const EST_String lookup(const EST_Item_featfunc func, int &found) const;
};

int operator == (const EST_FeatureFunctionPackage::Entry &a, const EST_FeatureFunctionPackage::Entry &b);
#endif

