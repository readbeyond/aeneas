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


#ifndef __EST_SINGECHANNELWAVE_H__
#define __EST_SINGECHANNELWAVE_H__

/** An alias for wave which can be used to tell the compiler you know it
  * is a simple one channel wave. This can speed up access.
  * 
  * @author Richard Caley <rjc@cstr.ed.ac.uk>
  * @version $Id: EST_SingleChannelWave.h,v 1.2 2001/04/04 13:11:27 awb Exp $
  */

#include "EST_Wave.h"

class EST_SingleChannelWave : public EST_Wave {

public:
  /// default constructor
  EST_SingleChannelWave() : EST_Wave() { };

  EST_SingleChannelWave(const EST_SingleChannelWave &w) : EST_Wave(w) { };

  EST_SingleChannelWave(int n, int sr) : EST_Wave(n, 1, sr) { };

  INLINE short &a_no_check(int i, int channel = 0)
        { return p_values.a_no_check_1(i,channel); }
  INLINE short a_no_check(int i, int channel = 0) const
        { return p_values.a_no_check_1(i,channel); }
};

#endif

