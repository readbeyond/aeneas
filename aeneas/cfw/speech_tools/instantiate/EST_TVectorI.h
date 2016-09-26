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


#ifndef __EST_TVECTOR_I_H__
#define __EST_TVECTOR_I_H__

/** Instantiate rules for vector template.
  * 
  * @author Richard Caley <rjc@cstr.ed.ac.uk>
  * @version $Id: EST_TVectorI.h,v 1.4 2006/07/19 21:52:12 awb Exp $
  */




// Instantiation Macros

#define Instantiate_TVector_T_MIN(TYPE,TAG) \
	template class EST_TVector< TYPE >;

#define Instantiate_TVector_T(TYPE,TAG) \
        Instantiate_TVector_T_MIN(TYPE,TAG)

#define Instantiate_TVector(TYPE) Instantiate_TVector_T(TYPE,TYPE)

#define Declare_TVector_T(TYPE,TAG)  \
	static TYPE const TAG##_vec_def_val_s; \
	static TYPE TAG##_vec_error_return_s; \
	\
	template <> TYPE const *EST_TVector< TYPE >::def_val=&TAG##_vec_def_val_s; \
	template <> TYPE *EST_TVector< TYPE >::error_return=&TAG##_vec_error_return_s;

#define Declare_TVector_Base_T(TYPE,DEFAULT,ERROR,TAG)  \
	static TYPE const TAG##_vec_def_val_s=DEFAULT; \
	static TYPE TAG##_vec_error_return_s=ERROR; \
	\
	template <> TYPE const *EST_TVector<TYPE>::def_val=&TAG##_vec_def_val_s; \
	template <> TYPE *EST_TVector<TYPE>::error_return=&TAG##_vec_error_return_s;

#define Declare_TVector_Class_T(TYPE,DEFAULT,ERROR,TAG)  \
	static TYPE const TAG##_vec_def_val_s(DEFAULT); \
	static TYPE TAG##_vec_error_return_s(ERROR); \
	\
	template <> TYPE const *EST_TVector<TYPE>::def_val=&TAG##_vec_def_val_s; \
	template <> TYPE *EST_TVector<TYPE>::error_return=&TAG##_vec_error_return_s;

#define Declare_TVector(TYPE) Declare_TVector_T(TYPE,TYPE)
#define Declare_TVector_Base(TYPE,DEFAULT,ERROR)  Declare_TVector_Base_T(TYPE,DEFAULT,ERROR,TYPE)
#define Declare_TVector_Class(TYPE,DEFAULT,ERROR) Declare_TVector_Class_T(TYPE,DEFAULT,ERROR,TYPE)

#endif

