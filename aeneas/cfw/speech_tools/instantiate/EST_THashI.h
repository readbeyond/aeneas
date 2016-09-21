
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


#ifndef __EST_THASH_I_H__
#define __EST_THASH_I_H__

#include "EST_system.h"
#include "instantiate/EST_TIteratorI.h"

/** Instantiate rules for hash template.
  * 
  * @author Richard Caley <rjc@cstr.ed.ac.uk>
  * @version $Id: EST_THashI.h,v 1.3 2004/05/04 00:00:17 awb Exp $
  */

// Instantiation Macros

#define Instantiate_THash_T_IT_IP(KEY, VAL, TAG, IP, IPK) \
	typedef EST_THash< KEY, VAL > HASH_ ## TAG ## _t; \
	typedef EST_Hash_Pair< KEY, VAL > HASHPAIR_ ## TAG ## _t; \
	Instantiate_TStructIterator_T(HASH_ ## TAG ## _t, HASH_ ## TAG ## _t:: IP, HASHPAIR_ ## TAG ## _t, HASH_ ## TAG ## _itt) \
	Instantiate_TIterator_T(HASH_ ## TAG ## _t, HASH_ ## TAG ## _t:: IP, HASHPAIR_ ## TAG ## _t, HASH_ ## TAG ## _itt) \
	Instantiate_TIterator_T(HASH_ ## TAG ## _t, HASH_ ## TAG ## _t:: IPK, KEY, HASH_ ## TAG ## _itt)

#if defined(VISUAL_CPP)
#    define Instantiate_THash_T_IT(KEY, VAL, TAG) \
	Instantiate_THash_T_IT_IP(KEY, VAL, TAG, IPointer_s, IPointer_k_s)
#else
#    define Instantiate_THash_T_IT(KEY, VAL, TAG) \
	Instantiate_THash_T_IT_IP(KEY, VAL, TAG, IPointer, IPointer_k)
#endif

#define Instantiate_THash_T_MIN(KEY, VAL, TAG) \
        template class EST_Hash_Pair< KEY, VAL >; \
        template class EST_THash< KEY, VAL >;

#define Instantiate_THash_T(KEY, VAL, TAG) \
	Instantiate_THash_T_MIN(KEY, VAL, TAG) \
	Instantiate_THash_T_IT(KEY, VAL, TAG)

#define Instantiate_THash(KEY, VAL) Instantiate_THash_T(KEY, VAL, KEY ## VAL) 
#define Instantiate_THash_MIN(KEY, VAL) Instantiate_THash_T_MIN(KEY, VAL, KEY ## VAL) 
#define Instantiate_THash_IT(KEY, VAL, IP) \
	  Instantiate_THash_T_IT(KEY, VAL, KEY ## VAL, IP)


/* disabled. it's INVALID !!!
#define Declare_THash_T(KEY, VAL, TAG) \
	VAL EST_THash< KEY, VAL >::Dummy_Value; \
	KEY EST_THash< KEY, VAL >::Dummy_Key; \
	EST_THash< KEY, VAL > TAG ## _hash_dummy(0);
*/

#define Declare_THash_Base_T(KEY, VAL, DEFAULTK, DEFAULT, ERROR,TAG) \
	template <> KEY EST_THash< KEY, VAL >::Dummy_Key=DEFAULTK; \
	template <> VAL EST_THash<KEY, VAL>::Dummy_Value=DEFAULT;

#define Declare_THash_Class_T(KEY, VAL, DEFAULTK, DEFAULT, ERROR,TAG) \
	template <> KEY EST_THash< KEY, VAL >::Dummy_Key(DEFAULTK); \
	template <> VAL EST_THash<KEY, VAL>::Dummy_Value(DEFAULT);

#define Declare_THash(KEY, VAL) Declare_THash_T(KEY, VAL, KEY ## VAL)

#define Declare_THash_Base(KEY, VAL, DEFAULTK, DEFAULT, ERROR)  \
	Declare_THash_Base_T(KEY, VAL, DEFAULTK, DEFAULT, ERROR, KEY ## VAL)
#define Declare_THash_Class(KEY, VAL, DEFAULTK, DEFAULT, ERROR) \
	Declare_THash_Class_T(KEY, VAL, DEFAULTK, DEFAULT, ERROR, KEY ## VAL)

#endif

