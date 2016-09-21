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


#ifndef __EST_HANDLABLE_H__
#define __EST_HANDLABLE_H__

/** Reference Counting Interface.
  *
  * This simple class does most of the things an object which is to be
  * manipulated by EST_THandle style smart pointers needs to provide.
  * 
  * @see EST_THandle
  * @see EST_TBox
  * @see EST_TrackMap
  * @author Richard Caley <rjc@cstr.ed.ac.uk>
  * @version $Id: EST_Handleable.h,v 1.4 2004/09/29 08:24:17 robert Exp $
  */

#include <climits>

using namespace std;

class EST_Handleable
{
private:
  int p_refcount;

public:
#   define NOT_REFCOUNTED (INT_MAX) 

  EST_Handleable(void) { p_refcount=NOT_REFCOUNTED; }

  int refcount(void) const { return p_refcount;}

  void start_refcounting(int initial=0) {p_refcount=initial;}
  void inc_refcount(void) {if (p_refcount!=NOT_REFCOUNTED) p_refcount++;}
  void dec_refcount(void) {if (p_refcount!=NOT_REFCOUNTED) p_refcount--;}

  int is_unreferenced(void) const {return p_refcount == 0;}
  int is_refcounted(void) const {return p_refcount!= NOT_REFCOUNTED;}
  //@}
};

#endif

