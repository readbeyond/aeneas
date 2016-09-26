/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                     Copyright (c) 1994,1995,1996                      */
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
/*                    Author :  Paul Taylor and Alan W Black             */
/*                    Date   :  July 1996                                */
/*-----------------------------------------------------------------------*/
/*                   Safe(-er) allocation functions.                     */
/*                                                                       */
/*=======================================================================*/

#if !defined(__EST_WALLOC_H__)

#if defined(__cplusplus)
extern "C" {
#endif

void *safe_walloc(int size);
void *safe_wcalloc(int size);
void *safe_wrealloc(void *ptr, int size);
#define walloc(TYPE,SIZE) ((TYPE *)safe_walloc(sizeof(TYPE)*(SIZE)))
#define wcalloc(TYPE,SIZE) ((TYPE *)safe_wcalloc(sizeof(TYPE)*(SIZE)))
#define wrealloc(PTR,TYPE,SIZE) ((TYPE *)safe_wrealloc((void *)(PTR), sizeof(TYPE)*(SIZE)))
char *wstrdup(const char *s);
void wfree(void *p);

void debug_memory_summary();

#if defined(__cplusplus)
}
#endif

#endif
