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
/*                     Author :  Paul Taylor                             */
/*                     Date   :  July 1995                               */
/*-----------------------------------------------------------------------*/
/*          Basic Multivariate Statistical Function Prototypes           */
/*                                                                       */
/*=======================================================================*/
#ifndef __Multistats_H__
#define __Multistats_H__

#include "EST_FMatrix.h"
#include "EST_types.h"

float mean(EST_FVector &m);

EST_FVector mean(EST_FMatrix &m);
EST_FVector sample_variance(EST_FMatrix &m);
EST_FVector sample_stdev(EST_FMatrix &m);

EST_FMatrix sample_covariance(EST_FMatrix &m);
EST_FMatrix sample_correlation(EST_FMatrix &m);

EST_FMatrix euclidean_distance(EST_FMatrix &m);
EST_FMatrix penrose_distance(EST_FMatrix &m);


EST_FMatrix normalise(EST_FMatrix &m, EST_FVector &sub, EST_FVector &div);

EST_FMatrix penrose_distance(EST_FMatrix &gu, EST_FVector &gv);
EST_FMatrix mahalanobis_distance(EST_FMatrix &gu, EST_FMatrix &v);
EST_FMatrix population_mean(EST_FMatrix *in, int num_pop);
EST_FMatrix add_populations(EST_FMatrix *in, int num_pop);

EST_FMatrix confusion(EST_StrStr_KVL &list, EST_StrList &lex);
void print_confusion(const EST_FMatrix &a, EST_StrStr_KVL &list, 
		     EST_StrList &lex);

#define OLS_IGNORE 100
int ols(const EST_FMatrix &X,const EST_FMatrix &Y, EST_FMatrix &coeffs);
int robust_ols(const EST_FMatrix &X,
	       const EST_FMatrix &Y, 
	       EST_FMatrix &coeffs);
int robust_ols(const EST_FMatrix &X,
	       const EST_FMatrix &Y, 
	       EST_IVector &included,
	       EST_FMatrix &coeffs);
int stepwise_ols(const EST_FMatrix &X,
		 const EST_FMatrix &Y,
		 const EST_StrList &feat_names,
		 float limit,
		 EST_FMatrix &coeffs,
		 const EST_FMatrix &Xtest,
		 const EST_FMatrix &Ytest,
                 EST_IVector &included);
int ols_apply(const EST_FMatrix &samples,
	      const EST_FMatrix &coeffs,
	      EST_FMatrix &res);
int ols_test(const EST_FMatrix &real,
	     const EST_FMatrix &predicted,
	     float &correlation,
	     float &rmse);

#endif // __Multistats_H__
