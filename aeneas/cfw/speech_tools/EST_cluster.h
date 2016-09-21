/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                         Copyright (c) 1996                            */
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
/*                    Author :  Paul Taylor                              */
/*                    Date   :  July 1995                                */
/*-----------------------------------------------------------------------*/
/*                 Clustering routines header file                       */
/*                                                                       */
/*=======================================================================*/
#ifndef __Cluster_H__
#define __Cluster_H__

#include "EST_util_class.h"

int load_names(EST_String file, EST_TList<EST_String> &names);

float lowestval(EST_FMatrix &m, EST_TList<int> &a, EST_TList<int> &b);
float highestval(EST_FMatrix &m, EST_TList<int> &a, EST_TList<int> &b);

int fn_cluster(EST_FMatrix &m, float d);
int nn_cluster(EST_FMatrix &m, float d);
float nearest(EST_TList<int> &cbk);
void merge(EST_TList<int> cbk[], int i, int j);

typedef EST_TList<EST_TList<int> > EST_CBK;

EST_String print_codebook(EST_CBK &cbk, float d, EST_TList<EST_String> &names);
EST_String print_codebook(EST_CBK &cbk, float d);
int cluster(EST_FMatrix &m, EST_CBK &cbk, float d);
void init_cluster(EST_CBK &cbk, int n);
//EST_FVector sortvals(EST_FMatrix &m);

int cluster(EST_FMatrix &m, EST_CBK &cbk, EST_TList<EST_String> &ans, EST_String method,
		    EST_TList<EST_String> &x);

EST_FVector sort_matrix(EST_FMatrix &m);

#endif /* Cluster_h */
