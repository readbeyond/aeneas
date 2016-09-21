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

#ifndef __EST_THANDLE_H__
#define __EST_THANDLE_H__

#include <iostream>

using namespace std;

#include "EST_bool.h"

/** A `smart' pointer which does reference counting.
  *
  * Behaves almost like a pointer as far as naive code is concerned, but
  * keeps count of how many handles are holding on to the contents
  * and deletes it when there are none.
  *
  * You need to be careful there are no dumb C++ pointers to things which
  * are being handled this way.
  *
  * Things to be handled should implement the same interface as EST_Handleable
  * (either by taking that as a superclass or by reimplementing it) and in
  * addition define {\tt object_ptr()}. See EST_TBox for an example.
  *
  * There are two parameter types. In most cases the thing which contains the
  * reference count and the data it represents will be the same object, but
  * in the case of boxed values it may not be, so you can specify the type
  * of both independently.
  * 
  * @see EST_Handleable
  * @see EST_TBox
  * @see EST_THandle:example
  * 
  * @author Richard Caley <rjc@cstr.ed.ac.uk>
  * @version $Id: EST_THandle.h,v 1.5 2006/07/19 21:52:12 awb Exp $
  */


template<class BoxT, class ObjectT>
class EST_THandle {

private:
  BoxT *ptr;

public:

  EST_THandle(void) { ptr = (BoxT *)NULL; }
    
  EST_THandle(BoxT *p) { if ((ptr=p)) p->inc_refcount(); }

  EST_THandle(const EST_THandle &cp) {
    ptr=cp.ptr;
    if (ptr)
      ptr->inc_refcount();
  }
  
  ~EST_THandle(void) {
    if (ptr)
      ptr->dec_refcount();
    if (ptr && ptr->is_unreferenced())
      delete ptr;
  }
  
  bool null() const { return ptr == NULL; }

  int shareing(void) const { return ptr?(ptr->refcount() > 1):0; }

  EST_THandle &operator = (EST_THandle h) {
    // doing it in this order means self assignment is safe.
    if (h.ptr)
      (h.ptr)->inc_refcount();
    if (ptr)
      ptr->dec_refcount();
    if (ptr && ptr->is_unreferenced())
      delete ptr;
    ptr=h.ptr;
    return *this;
  }

  // If they manage to get hold of one...
  // Actually usually used to assign NULL and so (possibly) deallocate
  // the object currently pointed to.
  EST_THandle &operator = (BoxT *t_ptr) {
    // doing it in this order means self assignment is safe.
    if (t_ptr)
      t_ptr->inc_refcount();
    if (ptr)
      ptr->dec_refcount();
    if (ptr && ptr->is_unreferenced())
      delete ptr;
    ptr=t_ptr;
    return *this;
  }
 
    operator ObjectT *() {
      return ptr?(ptr->object_ptr()):(ObjectT *)NULL;
    }

    operator const ObjectT *() const {
      return ptr?(ptr->object_ptr()):(const ObjectT *)NULL;
    }

  
  int operator == (const BoxT *p) const { return ptr == p; }
  int operator != (const BoxT *p) const { return !(*this == p); }

  const ObjectT& operator *() const { return *(ptr->object_ptr()); }
  ObjectT& operator *() { return *(ptr->object_ptr()); }
  const ObjectT* operator ->() const { return (ptr->object_ptr()); }
  ObjectT* operator ->() { return (ptr->object_ptr()); }

  friend int operator == (const EST_THandle< BoxT, ObjectT > &a, const EST_THandle< BoxT, ObjectT > & b)
    {return a.ptr==b.ptr; }
  friend int operator != (const EST_THandle< BoxT, ObjectT > &a, const EST_THandle< BoxT, ObjectT > & b)
    { return !( a==b ); }

  friend ostream & operator << (ostream &s, const EST_THandle< BoxT, ObjectT > &a)
    { return s << "HANDLE"; }
};

#endif
