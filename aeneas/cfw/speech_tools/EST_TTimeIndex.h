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


#ifndef __EST_TTIMEINDEX_H__
#define __EST_TTIMEINDEX_H__

#include <iostream>

/** A time index for a container. The container defines how to get an
  * object and so on, this lets you find a point in the container not
  * far before the entry you want.
  * 
  * @author Richard Caley <rjc@cstr.ed.ac.uk>
  * @version $Id: EST_TTimeIndex.h,v 1.3 2004/09/29 08:24:17 robert Exp $
  */

template<class  CONTAINER>
struct EST_TTI_Entry
{
  float t;
  CONTAINER::Index i;
};

template<class CONTAINER>
class EST_TTimeIndex 
{
public:
  typedef CONTAINER::Index Index;
  typedef EST_TTI_Entry<CONTAINER> Entry;

protected:
  CONTAINER *p_container;
  EST_TVector<Entry> p_entries;
  float p_time_step;

  void initialise();

private:
  void just_before(float time, void *in) const;

public:
  EST_TTimeIndex();
  EST_TTimeIndex(CONTAINER &c, int bunch=10);

  void index(CONTAINER &c, int bunch=10);
  void just_before(float time, Index &in) const
    { just_before(time, &in); }
};

template<class CONTAINER>
extern int operator !=(const EST_TTI_Entry<CONTAINER> &e1, 
		       const EST_TTI_Entry<CONTAINER> &e2);

template<class CONTAINER>
extern ostream& operator <<(
			    ostream &s, 
			    const EST_TTI_Entry<CONTAINER> &e);
#endif

