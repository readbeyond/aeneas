 /************************************************************************/
 /*                                                                      */
 /*                Centre for Speech Technology Research                 */
 /*                     University of Edinburgh, UK                      */
 /*                       Copyright (c) 1996,1997                        */
 /*                        All Rights Reserved.                          */
 /*                                                                      */
 /*  Permission is hereby granted, free of charge, to use and distribute */
 /*  this software and its documentation without restriction, including  */
 /*  without limitation the rights to use, copy, modify, merge, publish, */
 /*  distribute, sublicense, and/or sell copies of this work, and to     */
 /*  permit persons to whom this work is furnished to do so, subject to  */
 /*  the following conditions:                                           */
 /*   1. The code must retain the above copyright notice, this list of   */
 /*      conditions and the following disclaimer.                        */
 /*   2. Any modifications must be clearly marked as such.               */
 /*   3. Original authors' names are not deleted.                        */
 /*   4. The authors' names are not used to endorse or promote products  */
 /*      derived from this software without specific prior written       */
 /*      permission.                                                     */
 /*                                                                      */
 /*  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK       */
 /*  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING     */
 /*  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT  */
 /*  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE    */
 /*  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES   */
 /*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN  */
 /*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,         */
 /*  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF      */
 /*  THIS SOFTWARE.                                                      */
 /*                                                                      */
 /*************************************************************************/
 /*                                                                       */
 /*                 Author: Richard Caley (rjc@cstr.ed.ac.uk)             */
 /*                   Date: Tue Aug 26 1997                               */
 /* --------------------------------------------------------------------  */
 /* Extending buffers, i.e. arrays which grow as needed. I got fed up     */
 /* of writing this code all over the place.                              */
 /*                                                                       */
 /*************************************************************************/

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include "EST_unix.h"
#include "EST_TBuffer.h"

template<class T>
EST_TBuffer<T>::EST_TBuffer(unsigned int size, int step)
{
  p_buffer = NULL;
  init(size, step);
}

template<class T>
EST_TBuffer<T>::~EST_TBuffer(void)
{
  // save the buffer if we have a slot
  for(int i=0; i<TBUFFER_N_OLD; i++)
    if (EST_old_buffers[i].mem == NULL)
      {
	EST_old_buffers[i].mem = p_buffer;
	EST_old_buffers[i].size = p_size*sizeof(T);
	p_buffer = NULL;
	p_size =0;
	break;
      }

  if (p_buffer)
    {
      delete[] p_buffer;
      p_buffer = NULL;
      p_size = 0;
    }
}

template<class T>
void EST_TBuffer<T>::init(unsigned int size, int step)
{
  for(int i=0; i<TBUFFER_N_OLD; i++)
    if (EST_old_buffers[i].size/sizeof(T) >= size)
      {
	p_buffer = (T *)EST_old_buffers[i].mem;
	p_size = EST_old_buffers[i].size/sizeof(T);
	EST_old_buffers[i].mem = NULL;
	EST_old_buffers[i].size = 0;
	break;
      }

  if (p_buffer == NULL)
    {
    p_buffer = new T[size];
    p_size = size;
    }
  p_step = step;
}

template<class T>
void EST_TBuffer<T>::expand_to(unsigned int req_size, bool copy)
{
  if (req_size > p_size)
    {
      unsigned int new_size = p_size;

      while(new_size < req_size)
	if (p_step >0)
	  new_size += p_step;
	else
	  new_size = (int)(new_size*(float)(-p_step)/100.0);

      T * new_buffer = new T[new_size];

      if (copy)
	memcpy(new_buffer, p_buffer, p_size*sizeof(T));

      delete[] p_buffer;
      p_buffer = new_buffer;
      p_size = new_size;
    }
}

template<class T>
void EST_TBuffer<T>::expand_to(unsigned int req_size, const T &set_to, int howmany)
{
  if (req_size > p_size)
    {
      unsigned int new_size = p_size;

      while(new_size < req_size)
	if (p_step >0)
	  new_size += p_step;
	else
	  new_size = (int)(new_size*(float)(-p_step)/100.0);

      T * new_buffer = new T[new_size];

      if (howmany<0)
	howmany=new_size;
      for(int i=0; i<howmany; i++)
	new_buffer[i] = set_to;

      delete[] p_buffer;
      p_buffer = new_buffer;
      p_size = new_size;
    }
}

template<class T>
void EST_TBuffer<T>::set(const T &set_to, int howmany)
{
  if (howmany < 0)
    howmany = p_size;

  for(int i=0; i<howmany; i++)
    p_buffer[i] = set_to;
}

