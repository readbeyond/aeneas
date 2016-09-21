/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                    Copyright (c) 1994,1995,1996                       */
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
/*                      Author :  Paul Taylor                            */
/*                      Date   :  March 1998                             */
/*-----------------------------------------------------------------------*/
/*                File functions for EST type files                      */
/*                                                                       */
/*=======================================================================*/

// This is the .h file for EST generic header type parsing. 
// Only include this file in file i/o functions or else ott dependencies
// will crop in.

#ifndef __EST_FILETYPE_H__
#define __EST_FILETYPE_H__

#include "EST_Token.h"
#include "EST_Option.h"

typedef enum EST_EstFileType {
    est_file_none=0,
    est_file_track,
    est_file_wave,
    est_file_label,
    est_file_utterance,
    est_file_fmatrix,
    est_file_fvector,
    est_file_dmatrix,
    est_file_dvector,
    est_file_feature_data,
    est_file_fst,
    est_file_ngram,
    est_file_f_catalogue,
    est_file_index,
    est_file_unknown
} EST_EstFileType;

EST_read_status read_est_header(EST_TokenStream &ts, EST_Option &hinfo, 
				bool &ascii, EST_EstFileType &t);

#endif


