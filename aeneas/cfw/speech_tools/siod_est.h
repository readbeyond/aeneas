/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                      Copyright (c) 1996-1998                          */
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
/*                     Author :  Alan W Black                            */
/*                     Date   :  February 1999                           */
/*-----------------------------------------------------------------------*/
/*                                                                       */
/* Siod additions specific to Speech Tools                               */
/*                                                                       */
/*=======================================================================*/
#ifndef __SIOD_EST_H__
#define __SIOD_EST_H__

void siod_est_init();
void siod_fringe_init();

class EST_Val &val(LISP x);
int val_p(LISP x);
LISP siod(const class EST_Val v);

SIOD_REGISTER_CLASS_DCLS(wave,EST_Wave)
SIOD_REGISTER_CLASS_DCLS(track,EST_Track)
SIOD_REGISTER_CLASS_DCLS(feats,EST_Features)
SIOD_REGISTER_CLASS_DCLS(utterance,EST_Utterance)
SIOD_REGISTER_CLASS_DCLS(item,EST_Item)
/* SIOD_REGISTER_CLASS_DCLS(scheme,obj) */ /* removed for clang -- 14/10/13 */

#define get_c_utt(x) (utterance(x))
#define get_c_item(x) (item(x))

LISP lisp_val(const EST_Val &pv);
EST_Val val_lisp(LISP v);

LISP features_to_lisp(EST_Features &f);
void lisp_to_features(LISP lf,EST_Features &f);

LISP kvlss_to_lisp(const EST_TKVL<EST_String, EST_String> &kvl);
void lisp_to_kvlss(LISP l, EST_TKVL<EST_String, EST_String> &kvl);

EST_Features &Param();

#endif
