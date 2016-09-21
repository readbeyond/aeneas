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
/*                   Author :  Simon King                                */
/*                   Date   :  June 1998                                 */
/*-----------------------------------------------------------------------*/
/*                  Dynamical models for ASR                             */
/*                                                                       */
/*=======================================================================*/

#include <cstdlib>
#include <stdio.h>
#include <fstream.h>
#include "EST.h"
#include "EST_Handleable.h"
#include "EST_THandle.h"
#include "EST_TBox.h"
#include "EST_String.h"


struct stats_accumulator_HV
{
    EST_DMatrix M1;
    EST_DMatrix M2;
    EST_DMatrix covar_y;
    int points_processed;
};

struct stats_accumulator_FW
{
    EST_DMatrix M3;
    EST_DMatrix M4;
    EST_DMatrix covar_x_t_plus_1_except_last;
    int points_processed_except_last;
};

typedef EST_TBox<stats_accumulator_FW> Boxed_stats_accumulator_FW;
typedef EST_THandle<Boxed_stats_accumulator_FW,stats_accumulator_FW> refcounted_stats_accumulator_FW_P;

ostream& operator <<(ostream &s, const refcounted_stats_accumulator_FW_P &);

typedef EST_TBox<stats_accumulator_HV> Boxed_stats_accumulator_HV;
typedef EST_THandle<Boxed_stats_accumulator_HV,stats_accumulator_HV> refcounted_stats_accumulator_HV_P;
ostream& operator <<(ostream &s, const refcounted_stats_accumulator_HV_P &);

/// an accumulator to be owned by every model
/// - the underlying HV or FW accumulators are shared via
///   pointers to reference counted objects
struct stats_accumulator
{
    refcounted_stats_accumulator_HV_P hv;
    refcounted_stats_accumulator_FW_P fw;
};

ostream& operator <<(ostream &s, const stats_accumulator_FW &);
ostream& operator <<(ostream &s, const stats_accumulator_HV &);



typedef EST_TBox<EST_DMatrix> Boxed_EST_DMatrix;
typedef EST_THandle<Boxed_EST_DMatrix,EST_DMatrix> refcounted_EST_DMatrixP;

ostream& operator <<(ostream &s, const Boxed_EST_DMatrix &);
ostream& operator <<(ostream &s, const refcounted_EST_DMatrixP &);

//int operator ==(const Boxed_EST_DMatrix &a, 
//		const Boxed_EST_DMatrix &b);
int operator ==(const refcounted_EST_DMatrixP &a, 
		const refcounted_EST_DMatrixP &b);




typedef EST_TBox<EST_DVector> Boxed_EST_DVector;
typedef EST_THandle<Boxed_EST_DVector,EST_DVector> refcounted_EST_DVectorP;
int operator ==(const refcounted_EST_DVectorP &, 
		const refcounted_EST_DVectorP &);

ostream& operator <<(ostream &s, const refcounted_EST_DVectorP &);




