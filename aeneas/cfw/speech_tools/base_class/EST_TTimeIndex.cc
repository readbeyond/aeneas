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
 /*                   Date: Wed Mar 25 1998                               */
 /* --------------------------------------------------------------------  */
 /* Indexing a container by time.                                         */
 /*                                                                       */
 /*************************************************************************/

#include "EST_TTimeIndex.h"

template<class CONTAINER>
void EST_TTimeIndex<CONTAINER>::initialise()
{
  p_time_step=0;
  p_entries.resize(0);
}

template<class CONTAINER>
EST_TTimeIndex<CONTAINER>::EST_TTimeIndex()
{
  initialise();
}


template<class CONTAINER>
EST_TTimeIndex<CONTAINER>::EST_TTimeIndex(CONTAINER &c, int bunch)
{
  initialise();
  index(c, bunch);
}

template<class CONTAINER>
void EST_TTimeIndex<CONTAINER>::index(CONTAINER &c, int bunch)
{
  int n_objects = c.length();
  float total_time = c.end();

  int n_buckets = n_objects/bunch +1;

  p_time_step = total_time / n_buckets;
  p_entries.resize(n_buckets);
  p_container = &c;

  Index i;
  
  i=c.first_index();
  p_entries[0].t = 0.0;
  p_entries[0].i = i;

  for(; c.valid_index(i); i=c.next_index(i))
    {
      float t = c.time_of(i);
      int b = (int)(t/p_time_step);
      if (b>=p_entries.num_columns())
	b = p_entries.num_columns()-1;
      for (int bb=b+1; bb < n_buckets ; bb++)
	if ( t > p_entries(bb).t )
	  {
	    p_entries[bb].t = t;
	    p_entries[bb].i = i;
	  }
	else
	  break;
    }
}

template<class CONTAINER>
void EST_TTimeIndex<CONTAINER>::just_before(float t, 
					    void *inp) const
{
  CONTAINER::Index &in(*(Index *)inp);
  in= CONTAINER::bad_index();

  if (p_container==NULL)
    return;

  int b = (int)(t/p_time_step);

  if (b>=p_entries.num_columns())
    b = p_entries.num_columns()-1;

  Index i = p_entries(b).i;
  
  for(Index j=i; p_container->valid_index(j); j = p_container->next_index(j))
    {
      if (p_container->time_of(j) > t)
	{
	  in=i;
	  return;
	}
      i=j;
    }
  in =i;
  return;
}

template<class CONTAINER>
int operator !=(const EST_TTI_Entry<CONTAINER> &e1, 
		       const EST_TTI_Entry<CONTAINER> &e2)
{ (void)e1; (void)e2; return 1; }

template<class CONTAINER>
ostream& operator <<(ostream &s, 
			    const EST_TTI_Entry<CONTAINER> &e)
{ (void)e; return s << "entry"; }
