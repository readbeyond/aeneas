/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                      Copyright (c) 1995,1996                          */
/*                        All Rights Reserved.                           */
/*                                                                       */
/*  Permission is hereby granted, free of charge, to use and distribute  */
/*  this software and its documentation without restriction, including   */
/*  without limitation the rights to use, copy, modify, merge, publish,  */
/*  distribute, sublicense, and/or sell copies of this work, and to      */
/*  permit persons to whom this work is furnished to do so, subject to   */
/*  the following conditions:                                            */
/*   1. The code must retain the above copyright notice, this list of    */
/*      conditions and the following disclaimer.                         */
/*   2. Any modifications must be clearly marked as such.                */
/*   3. Original authors' names are not deleted.                         */
/*   4. The authors' names are not used to endorse or promote products   */
/*      derived from this software without specific prior written        */
/*      permission.                                                      */
/*                                                                       */
/*  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK        */
/*  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      */
/*  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   */
/*  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE     */
/*  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    */
/*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   */
/*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          */
/*  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       */
/*  THIS SOFTWARE.                                                       */
/*                                                                       */
/*************************************************************************/
/*                       Author :  Paul Taylor                           */
/*                       Date   :  July 1996                             */
/*-----------------------------------------------------------------------*/
/*                   Type defines for Common Types                       */
/*                                                                       */
/*=======================================================================*/

#ifndef __EST_TYPES_H__
#define __EST_TYPES_H__

#include "EST_TList.h"
#include "EST_TVector.h"
#include "EST_String.h"
#include "EST_TKVL.h"
#include "EST_FMatrix.h"
#include "EST_DMatrix.h"
#include "EST_IMatrix.h"
#include "EST_SMatrix.h"

typedef EST_TVector<EST_String> EST_StrVector;

typedef EST_TSimpleVector<int> EST_IVector;
typedef EST_TSimpleVector<short> EST_SVector;
typedef EST_TSimpleVector<char> EST_CVector;

// DVector is an inherited TSimpleVector in EST_DMatrix.h
// FVector is an inherited TSimpleVector in EST_FMatrix.h

typedef EST_TList<int> EST_IList;
typedef EST_TList<float> EST_FList;
typedef EST_TList<double> EST_DList;

typedef EST_TKVL<int, int> EST_II_KVL;

typedef EST_TList<EST_TList<int> > EST_IListList;

typedef EST_TList<EST_String> EST_StrList;
typedef EST_TList<int> EST_IList;
typedef EST_TList<float> EST_FList;

typedef EST_TList<EST_TList<EST_String> > EST_StrListList;
typedef EST_TVector<EST_StrList> EST_StrListVector;
typedef EST_TKVL<EST_String, EST_String> EST_StrStr_KVL;
typedef EST_TKVL<EST_String, int> EST_StrI_KVL;
typedef EST_TKVL<EST_String, float> EST_StrF_KVL;
typedef EST_TKVL<EST_String, double> EST_StrD_KVL;
//typedef EST_TKVL<EST_String, EST_Val> EST_StrVal_KVL;

#endif // __EST_TYPES_H__
