
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


#ifndef __EST_TLIST_I_H__
#define __EST_TLIST_I_H__

/** Instantiate rules for list template.
  * 
  * @author Richard Caley <rjc@cstr.ed.ac.uk>
  * @version $Id: EST_TListI.h,v 1.5 2006/07/19 21:52:12 awb Exp $
  */


// Instantiation Macros

#include <iostream>

using namespace std;

#include "instantiate/EST_TIteratorI.h"

#define Instantiate_TList_T_MIN(TYPE, TAG) \
	template class EST_TList< TLIST_ ## TAG ## _VAL >; \
	template class EST_TItem< TLIST_ ## TAG ## _VAL >; \
	template const char *error_name(EST_TList< TYPE > val); \
        Instantiate_TIterator_T( EST_TList<TYPE>, EST_TList<TYPE>::IPointer, TYPE, TList_ ## TAG ## _itt);

#define Instantiate_TList_T(TYPE, TAG) \
	Instantiate_TList_T_MIN(TYPE, TAG)

#define Instantiate_TList(TYPE) Instantiate_TList_T(TYPE, TYPE)

#define Declare_TList_TN(TYPE,MaxFree,TAG) \
	typedef TYPE TLIST_ ## TAG ## _VAL; \
	template <> EST_TItem< TYPE > * EST_TItem< TYPE >::s_free=NULL; \
	template <> unsigned int EST_TItem< TYPE >::s_maxFree=MaxFree; \
	template <> unsigned int EST_TItem< TYPE >::s_nfree=0;
#define Declare_TList_T(TYPE,TAG) \
	Declare_TList_TN(TYPE,0,TAG)

#define Declare_TList_Base_TN(TYPE,MaxFree,TAG) \
	Declare_TList_TN(TYPE,MaxFree,TAG) 
#define Declare_TList_Base_T(TYPE,TAG) \
	Declare_TList_Base_TN(TYPE,0,TAG) \

#define Declare_TList_Class_TN(TYPE,MaxFree,TAG) \
	Declare_TList_TN(TYPE,MaxFree,TAG) 
#define Declare_TList_Class_T(TYPE,TAG) \
	Declare_TList_Class_TN(TYPE,0,TAG) \

#define Declare_TList_N(TYPE,MaxFree) Declare_TList_TN(TYPE,MaxFree,TYPE)
#define Declare_TList_Base_N(TYPE,MaxFree)  Declare_TList_Base_TN(TYPE,MaxFree,TYPE)
#define Declare_TList_Class_N(TYPE,MaxFree) Declare_TList_Class_TN(TYPE,MaxFree,TYPE)

#define Declare_TList(TYPE) Declare_TList_N(TYPE,0)
#define Declare_TList_Base(TYPE)  Declare_TList_Base_N(TYPE,0)
#define Declare_TList_Class(TYPE) Declare_TList_Class_N(TYPE,0)

#endif

