/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                       Copyright (c) 1996,1997                         */
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
/*                     Date   :  April 1996                              */
/*-----------------------------------------------------------------------*/
/*                                                                       */
/*  Shared text utilities                                                */
/*                                                                       */
/*=======================================================================*/
#ifndef __TEXT_H__
#define __TEXT_H__

EST_Item *add_token(EST_Utterance *u,EST_Token &t);
void festival_token_init(void);
LISP extract_tokens(LISP file, LISP tokens,LISP ofile);
LISP new_token_utt(void);
void tts_file_xxml(LISP filename);
void tts_file_raw(LISP filename);

LISP xxml_call_element_function(const EST_String &element,
				LISP atts, LISP elements, LISP utt);
LISP xxml_get_tokens(const EST_String &line,LISP feats,LISP utt);

typedef void (*TTS_app_tok)(EST_Item *token);
typedef void (*TTS_app_utt)(LISP utt);

LISP tts_chunk_stream(EST_TokenStream &ts,
		      TTS_app_tok app_tok, 
		      TTS_app_utt app_utt,
		      LISP eou_tree,    
		      LISP utt);

void tts_file_user_mode(LISP filename, LISP params);

#endif /* __TEXT_H__ */



