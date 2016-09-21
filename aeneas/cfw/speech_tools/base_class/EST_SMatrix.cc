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
/*                         Author :  Paul Taylor                         */
/*                         Date   :  April 1995                          */
/*-----------------------------------------------------------------------*/
/*                        Matrix Class for shorts                        */
/*                                                                       */
/*=======================================================================*/

#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <cmath>
#include <climits>
#include "EST_String.h"
#include "EST_types.h"
#include "EST_FileType.h"
#include "EST_Option.h"
#include "EST_SMatrix.h"
#include "EST_cutils.h"  // for swap functions 
#include "EST_Token.h"
#include "rateconv.h"

EST_SMatrix::EST_SMatrix(EST_SMatrix &a, int b) 
:EST_TSimpleMatrix<short>(a.num_rows(), a.num_columns())
{
    short vv = 0;
    if (b < 0)
	return;
    if (b == 0)
	fill(vv);
}

int EST_SMatrix::rateconv(int in_samp_freq, int out_samp_freq)
{
  short *in_buf = new short[num_rows()];
  short ** results = new short *[num_columns()];
  int *len = new int[num_columns()];
  int max_len=0;

  for(int c=0; c<num_columns(); c++)
    {
      short *out_buf;
      int osize;

      copy_column(c, in_buf);

      if (::rateconv(in_buf,
		   num_rows(), &out_buf, &osize,
		   in_samp_freq, out_samp_freq) == 0)
	{
	  results[c]=out_buf;
	  len[c]=osize;
	  if (osize > max_len)
	    max_len = osize;
	}
      else
	return -1;
    }
  delete [] in_buf;

  resize(max_len, EST_CURRENT, 0);
  fill(0);

  for(int c1=0; c1<num_columns(); c1++)
    {
      set_column(c1, results[c1], 0, len[c1]);
      delete [] results[c1];
    }

  delete [] results;
  delete [] len;
  return 0;
}

