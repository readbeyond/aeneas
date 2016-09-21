
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


#ifndef __EST_TSTRINGHASH_I_H__
#define __EST_TSTRINGHASH_I_H__

/** Instantiate rules for hash template.
  * 
  * @author Richard Caley <rjc@cstr.ed.ac.uk>
  * @version $Id: EST_TStringHashI.h,v 1.2 2001/04/04 13:11:27 awb Exp $
  */

// Instantiation Macros

#define Instantiate_TStringHash_T_IT(VAL, TAG, IP) \
	Instantiate_THash_T_IT(EST_String, VAL, TAG, IP)

#define Instantiate_TStringHash_T_MIN(VAL, TAG) \
        template class EST_TStringHash< VAL >; \
	Instantiate_THash_T_MIN(EST_String,VAL,TAG)

#define Instantiate_TStringHash_T(VAL, TAG) \
        template class EST_TStringHash< VAL >; \
	Instantiate_THash_T(EST_String,VAL,TAG)

#define Instantiate_TStringHash(VAL) Instantiate_TStringHash_T(VAL, KEY ## VAL) 
#define Instantiate_TStringHash_MIN(VAL) Instantiate_TStringHash_T_MIN(VAL, KEY ## VAL) 
#define Instantiate_TStringHash_IT(VAL, IP) Instantiate_TStringHash_T_IT(VAL, KEY ## VAL, IP) 

#define Declare_TStringHash_T(VAL, TAG) \
	Declare_THash_T(EST_String, VAL, TAG)

#define Declare_TStringHash_Base_T(VAL, DEFAULT, ERROR,TAG) \
	Declare_THash_Base_T(EST_String, VAL, "DUMMY", DEFAULT, ERROR,TAG)


#define Declare_TStringHash_Class_T(VAL, DEFAULT, ERROR,TAG) \
	Declare_THash_Class_T(EST_String, VAL, "DUMMY", DEFAULT, ERROR,TAG)

#define Declare_TStringHash(VAL) Declare_TStringHash_T(VAL, VAL)

#define Declare_TStringHash_Base(VAL, DEFAULT, ERROR)  \
	Declare_TStringHash_Base_T(VAL, DEFAULT, ERROR, VAL)
#define Declare_TStringHash_Class(VAL, DEFAULT, ERROR) \
	Declare_TStringHash_Class_T(VAL, DEFAULT, ERROR, VAL)

#endif

