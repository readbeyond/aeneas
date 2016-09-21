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


#ifndef __EST_TDEQUE_H__
#define __EST_TDEQUE_H__

#include "EST_TVector.h"
#include "instantiate/EST_TDequeI.h"

/** Double ended queue.
  * 
  * @author Richard Caley <rjc@cstr.ed.ac.uk>
  * @version $Id: EST_TDeque.h,v 1.3 2006/07/19 21:52:12 awb Exp $
  */

template <class T>
class EST_TDeque {
private:
  EST_TVector<T> p_vector;
  int p_increment;
  int p_back;
  int p_front;

  // Make the structure bigger.
  void expand(void);

public:
  EST_TDeque(unsigned int capacity, unsigned int increment);
  EST_TDeque(unsigned int capacity);
  EST_TDeque(void);

  /// Used to fill empty spaces when possible.
  static const T *Filler;

  /// Empty it out.
  void clear(void);

  /// Is there anything to get?
  bool is_empty(void) const;

  /// print picture of state. Mostly useful for debugging.
  ostream &print(ostream &s) const;

  /**@name stack
    * 
    * An interface looking like a stack.
    */
  //@{
  void push(T &item);
  T &pop(void);
  T &nth(int i);
  //@}

  /**@name inverse stack
    * 
    * The other end as a stack.
    */
  //@{
  void back_push(T& item);
  T &back_pop(void);
  //@}

  /**@name queue
    * 
    * An interface looking like a queue.
    */
  //@{
  void add(T& item) { push(item); }
  T &next() { return back_pop(); }
  T &last() { return pop(); }
  //@}

  /**@name perl
    * 
    * For people who think in perl
    */
  //@{
  void unshift(T& item) { back_push(item); }
  T &shift() { return back_pop(); }
  //@}

  friend ostream& operator << (ostream &st, const EST_TDeque< T > &deq)
    {
        return deq.print(st);
    }
};

#endif

