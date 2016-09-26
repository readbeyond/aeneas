
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


#ifndef __EST_KVL_I_H__
#define __EST_KVL_I_H__

/** Instantiate rules for list template.
  * 
  * @author Richard Caley <rjc@cstr.ed.ac.uk>
  * @version $Id: EST_TKVLI.h,v 1.4 2006/07/19 21:52:12 awb Exp $
  */

#include "instantiate/EST_TListI.h"
#include "instantiate/EST_TIteratorI.h"

// Instantiation Macros

// the typedef is purely to get the type name through the following macro.

#define Instantiate_KVL_T(KEY, VAL, TAG) \
        template class EST_TKVL<KEY, VAL>; \
        template class EST_TKVI<KEY, VAL>; \
        Instantiate_TIterator_T(KVL_ ## TAG ## _t, KVL_ ## TAG ## _t::IPointer_k, KEY, KVL_ ## TAG ##_kitt) \
        Instantiate_TStructIterator_T(KVL_ ## TAG ## _t, KVL_ ## TAG ## _t::IPointer, KVI_ ## TAG ## _t, KVL_ ## TAG ##_itt) \
        Instantiate_TIterator_T(KVL_ ## TAG ## _t, KVL_ ## TAG ## _t::IPointer, KVI_ ## TAG ## _t, KVL_ ## TAG ##_itt) \
        Instantiate_TList(KVI_ ## TAG ## _t)

// template ostream & operator<<(ostream &s, EST_TKVI<KEY, VAL> const &i); 

#define Instantiate_KVL(KEY, VAL) \
		Instantiate_KVL_T(KEY, VAL, KEY ## VAL) 

#define Declare_KVL_TN(KEY, VAL, MaxFree, TAG) \
	typedef EST_TKVI<KEY, VAL> KVI_ ## TAG ## _t; \
	typedef EST_TKVL<KEY, VAL> KVL_ ## TAG ## _t; \
	\
	static VAL TAG##_kv_def_val_s; \
	static KEY TAG##_kv_def_key_s; \
	\
	template <> VAL *EST_TKVL< KEY, VAL >::default_val=&TAG##_kv_def_val_s; \
	template <> KEY *EST_TKVL< KEY, VAL >::default_key=&TAG##_kv_def_key_s; \
	\
	Declare_TList_N(KVI_ ## TAG ## _t, MaxFree)
#define Declare_KVL_T(KEY, VAL, TAG) \
	Declare_KVL_TN(KEY, VAL, 0, TAG)

#define Declare_KVL_Base_TN(KEY, VAL, DEFV, DEFK, MaxFree, TAG) \
	typedef EST_TKVI<KEY, VAL> KVI_ ## TAG ## _t; \
	typedef EST_TKVL<KEY, VAL> KVL_ ## TAG ## _t; \
	\
	static VAL TAG##_kv_def_val_s=DEFV; \
	static KEY TAG##_kv_def_key_s=DEFK; \
	\
	template <> VAL *EST_TKVL< KEY, VAL >::default_val=&TAG##_kv_def_val_s; \
	template <> KEY *EST_TKVL< KEY, VAL >::default_key=&TAG##_kv_def_key_s; \
	\
	Declare_TList_N(KVI_ ## TAG ## _t, MaxFree)
#define Declare_KVL_Base_T(KEY, VAL, DEFV, DEFK, TAG) \
	Declare_KVL_Base_TN(KEY, VAL, DEFV, DEFK, 0, TAG)

#define Declare_KVL_Class_TN(KEY, VAL, DEFV, DEFK, MaxFree, TAG) \
	typedef EST_TKVI<KEY, VAL> KVI_ ## TAG ## _t; \
	typedef EST_TKVL<KEY, VAL> KVL_ ## TAG ## _t; \
	\
	static VAL TAG##_kv_def_val_s(DEFV); \
	static KEY TAG##_kv_def_key_s(DEFK); \
	\
	template <> VAL *EST_TKVL< KEY, VAL >::default_val=&TAG##_kv_def_val_s; \
	template <> KEY *EST_TKVL< KEY, VAL >::default_key=&TAG##_kv_def_key_s; \
	\
	Declare_TList_N(KVI_ ## TAG ## _t, MaxFree)
#define Declare_KVL_Class_T(KEY, VAL, DEFV, DEFK,TAG) \
	Declare_KVL_Class_TN(KEY, VAL, DEFV, DEFK, 0, TAG)

#define Declare_KVL_N(KEY, VAL, MaxFree) \
		Declare_KVL_TN(KEY, VAL, MaxFree, KEY ## VAL)
#define Declare_KVL(KEY, VAL) \
		Declare_KVL_N(KEY, VAL, 0)

#define Declare_KVL_Base_N(KEY, VAL, DEFV, DEFK, MaxFree)  \
		Declare_KVL_Base_TN(KEY, VAL, DEFV, DEFK, , MaxFree, KEY ## VAL)
#define Declare_KVL_Base(KEY, VAL, DEFV, DEFK)  \
		Declare_KVL_Base_N(KEY, VAL, DEFV, DEFK, 0)

#define Declare_KVL_Class_N(KEY, VAL, DEFV, DEFK, MaxFree) \
		Declare_KVL_Class_TN(KEY, VAL, DEFV, DEFK, MaxFree, KEY ## VAL)
#define Declare_KVL_Class(KEY, VAL, DEFV, DEFK) \
		Declare_KVL_Class_N(KEY, VAL, DEFV, DEFK, 0)

#endif

