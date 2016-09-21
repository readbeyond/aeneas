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
#include <cstdio>
#include <fstream>
#include "EST.h"
#include "EST_model_types.h"

// for now, CSMM means continuous-state Markov model
class CSMM
{
public:
    refcounted_EST_DMatrixP F;
    refcounted_EST_DMatrixP H;
    refcounted_EST_DVectorP u_v;
    refcounted_EST_DMatrixP Q_v;
    refcounted_EST_DVectorP u_w; 
    refcounted_EST_DMatrixP Q_w;
    refcounted_EST_DVectorP mean_initial_x;
    stats_accumulator stats;
};

bool operator ==(const CSMM &, const CSMM &);
ostream& operator <<(ostream &s, const CSMM &);

/// somewhere to keep the model parameters
struct CSMM_parameter_set
{
    EST_TKVL<EST_String,refcounted_EST_DMatrixP> matrix_set;
    EST_TKVL<EST_String,refcounted_EST_DVectorP> vector_set;
};

/// somewhere to keep the models
struct CSMM_set
{
    EST_TKVL<EST_String,CSMM> model_set;
    CSMM_parameter_set parameter_set;
};



ostream& operator <<(ostream &s, const CSMM_set &ms);

bool
create_empty_CSMM(EST_String model_name,
		  CSMM_set &model_set,
		  const int model_order,
		  const int observation_order);

/// at last, some interesting functions
bool
E_step(CSMM &model,
       EST_DMatrix &mean_initial_covar_x,
       const EST_TrackList &train_data,
       const bool trace);
bool
M_step(CSMM &model,
       const bool trace);
bool
EM_shared_hidden_space(CSMM_set &models,
		       const int model_order,
		       EST_DMatrix &mean_initial_covar_x,
		       const EST_TrackList &train_data,
		       const int iterations,
		       const bool trace);
EST_write_status
save_CSMM_set(CSMM_set &model_set, const EST_String &dirname, const EST_String &type);

EST_read_status
load_CSMM_set(CSMM_set &model_set, const EST_String &dirname);

double
CSMM_Mahalanobis_distance(const CSMM &model, const EST_Track &t);
