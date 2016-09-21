/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                         Copyright (c) 1994,1995,1996                  */
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
/*                    Date   :  May 1994                                 */
/*-----------------------------------------------------------------------*/
/*                 Label Comparison Routines                             */
/*                                                                       */
/*=======================================================================*/

#ifndef __EST_RELATION_COMPARE_H__
#define __EST_RELATION_COMPARE_H__

float label_distance1(EST_Item &ref, EST_Item &test);
EST_Item *nthpos(EST_Relation &a, int n);
void compare_labels(EST_Relation &reflab, EST_Relation &testlab);
void relation_match(EST_Relation &a, EST_Relation &b);
void function_match(EST_II_KVL &u, EST_Relation &a, EST_Relation &b);
void monotonic_match(EST_II_KVL &a, EST_II_KVL &b);
void show_links(EST_Relation &a, EST_Relation &b);
int close_enough(EST_Item &a, EST_Item &b);
int matrix_deletions(EST_FMatrix &m);
int matrix_insertions(EST_FMatrix &m);
void matrix_ceiling(EST_FMatrix &m, float max);
void minimise_matrix_by_row(EST_FMatrix &m);
void minimise_matrix_by_column(EST_FMatrix &m);
int lowest_pos(EST_FMatrix &m, int j);
float label_distance2(EST_Item &ref, EST_Item &test);


void print_results(EST_Relation &ref, EST_Relation &test, EST_FMatrix &m, int tot,
		   int del, int ins, int v);
void print_aligned_trans(EST_Relation &ref, EST_Relation &test, EST_FMatrix &m);
void pos_only(EST_Relation &a);
void print_s_trans(EST_Relation &a, int width=3);
int num_b_deletions(EST_FMatrix &m, int last, int current);
int num_b_insertions(EST_FMatrix &m, int last, int current);
int column_hit(EST_FMatrix &m, int c);
int row_hit(EST_FMatrix &m, int r);
void print_matrix_scores(EST_Relation &ref, EST_Relation &test, EST_FMatrix &a);
void print_i_d_scores(EST_FMatrix &m);
void test_labels(EST_Utterance &ref, EST_Utterance &test, EST_Option &op);
int commutate(EST_Item *a_ptr, EST_II_KVL &f1, EST_II_KVL &f2,
	      EST_II_KVL &lref, EST_II_KVL &ltest); 

void reassign_links(EST_Relation &a, EST_II_KVL &u, EST_String stream_type);
void reassign_links(EST_Relation &a, EST_Relation &b, EST_II_KVL &ua, EST_II_KVL &ub);
int compare_labels(EST_Utterance &ref, EST_Utterance &test, EST_String name,
		   EST_II_KVL &uref, EST_II_KVL &utest);

int insdel(EST_II_KVL &a);
void error_location(EST_Relation &e, EST_FMatrix &m, int ref);
void multiple_matrix_compare(EST_TList<EST_Relation> &rmlf, EST_TList<EST_Relation>
			     &tmlf, EST_FMatrix &m, EST_String rpos, EST_String tpos, int
			     method, float t, int v);

EST_FMatrix matrix_compare(EST_Relation &reflab, EST_Relation &testlab, int method,
		       float t, int v);

void multiple_labels(EST_Relation &reflab);
void threshold_labels(EST_Relation &reflab, float t);

#endif //__EST_RELATION_COMPARE_H__
