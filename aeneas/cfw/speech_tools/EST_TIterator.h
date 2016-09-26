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
 /************************************************************************/

#ifndef __EST_TITERATOR_H__
#define __EST_TITERATOR_H__

/** Template class defining interface to an iterator, i.e an object
  * which returns elements from a structure one at a time.
  *
  * This is template is usually hidden in the declaration of the
  * container classes with a typedef for Entries providing a more
  * convenient name for the iterator. However the interface is that
  * defined here.
  *
  * We support two interfaces, a pointer like interface similar to
  * specialised iteration code elsewhere in the speech tools library
  * and to the iterators in the C++ standard template library and an
  * interface similar to that of Enumerations in Java.
  *
  * <programlisting arch='c++'>
  * MyContainer::Entries them;
  *
  * for(them.begin(container); them; them++)
  *     {
  *     MyContainer::Entry &it = *them;
  *     // Do Something With it
  *     }</programlisting>
  * 
  * <programlisting arch='c++'>
  * MyContainer::Entries them;
  *
  * them.begin(container);
  * while (them.has_more_entries())
  *     {
  *     MyContainer::Entry &it = them.next_entry();
  *     // Do Something With it
  *     }</programlisting>
  * 
  * @author Richard Caley <rjc@cstr.ed.ac.uk>
  * @version $Id: EST_TIterator.h,v 1.7 2013/04/13 14:17:11 awb Exp $ 
  */

template <class Container, class IPointer, class Entry> 
    class EST_TStructIterator;
template <class Container, class IPointer, class Entry> 
    class EST_TRwIterator;
template <class Container, class IPointer, class Entry> 
    class EST_TRwStructIterator;

template <class Container, class IPointer, class Entry>
class EST_TIterator
{
protected:
  /// The container we are looking at.
  Container *cont;

  /// Position in the structure. May or may not be useful.
  unsigned int pos;

  /** Structure defined by the container class which contains the
    * current state of the iteration.
    */
  IPointer pointer;

public:
  /// Name for an iterator like this
  typedef EST_TIterator<Container, IPointer, Entry> Iter;

  /// Create an iterator not associated with any specific container.
  EST_TIterator() {cont=NULL;}

  /// Create an iterator ready to run over the given container.
  EST_TIterator(const Container &over)
    { begin(over); }

  /// Copy an iterator by assignment
  Iter &operator = (const Iter &orig)
    { cont=orig.cont; pos=orig.pos; pointer=orig.pointer; return *this;}

  /// Assigning a container to an iterator sets it ready to start.
  Iter &operator = (const Container &over)
    { begin(over); return *this;}

  /// Set the iterator ready to run over this container.
  void begin(const Container &over)
    {cont=((Container *)(void *)&over); beginning();}

  /// Reset to the start of the container.
  void beginning() 
    {if (cont) cont->point_to_first(pointer); pos=0;}

  /**@name End Tests
    */
  //@{
  /// True if there are more elements to look at.
  bool has_more_elements() const
    {return cont && cont->points_to_something(pointer);}

  /// True when there are no more.
  bool at_end() const
    {return !has_more_elements();}

  /** Viewing the iterator as an integer (for instance in a test)
    * sees a non-zero value iff there are elements still to look at.
    */
  operator int() const
    {return has_more_elements();}
  //@}

  /**@name Moving Forward
    */
  //@{
  /// Next moves to the next entry.
  void next()
    {cont->move_pointer_forwards(pointer); pos++;}

  /// The increment operator does the same as next.
  Iter &operator ++()
    {next(); return *this;}
  Iter operator ++(int dummy)
    {
      (void)dummy; 
      Iter old =*this;
      next(); 
      return old;
    }
  //@}

  /**@name Access
    */
  //@{
  /// Return the element currently pointed to.
  const Entry& current() const
    {return cont->points_at(pointer);}

  /// The * operator returns the current element. 
  const Entry &operator *() const
    {return current();}

#if 0
  // This only works for some Entry types.
  const Entry *operator ->() const
    {return &current();}
#endif

  /// Return the current element and move the pointer forwards.
  const Entry& next_element() 
	{ 
	  const Entry &it = cont->points_at(pointer); 
	  cont->move_pointer_forwards(pointer); 
	  return it; 
	}

  /// Return the current position

  unsigned int n() const { return pos; }
  //@}

  friend class EST_TStructIterator <Container, IPointer, Entry>;
  friend class EST_TRwIterator <Container, IPointer, Entry>;
  friend class EST_TRwStructIterator <Container, IPointer, Entry>;

};

template <class Container, class IPointer, class Entry>
class EST_TStructIterator 
  : public EST_TIterator<Container, IPointer, Entry>
{
public:  

  typedef EST_TIterator<Container, IPointer, Entry> Iter;

  /// Create an iterator not associated with any specific container.
  EST_TStructIterator() {this->cont=NULL;}

  /// Copy an iterator by assignment
  Iter &operator = (const Iter &orig)
    { this->cont=orig.cont; this->pos=orig.pos; this->pointer=orig.pointer; return *this;}

  /// Create an iterator ready to run over the given container.
  EST_TStructIterator(const Container &over)
    { this->begin(over); }

  const Entry *operator ->() const
    {return &this->current();}
};

template <class Container, class IPointer, class Entry>
class EST_TRwIterator 
  : public EST_TIterator<Container, IPointer, Entry>
{
private:
  /// Can't access constant containers this way.
  //  EST_TRwIterator(const Container &over) { (void) over; }

  /// Can't access constant containers this way.
  // void begin(const Container &over) { (void) over; }

public:

  typedef EST_TRwIterator<Container, IPointer, Entry> Iter;

  /// Create an iterator not associated with any specific container.
  EST_TRwIterator() {this->cont=NULL;}

  /// Copy an iterator by assignment
  Iter &operator = (const Iter &orig)
    { this->cont=orig.cont; this->pos=orig.pos; this->pointer=orig.pointer; return *this;}

  /// Create an iterator ready to run over the given container.
  EST_TRwIterator(Container &over)
    { begin(over); }

  /// Set the iterator ready to run over this container.
  void begin(Container &over)
    {this->cont=&over; this->beginning();}

  /**@name Access
    */
  //@{
  /// Return the element currently pointed to.
  Entry& current() const
    {return this->cont->points_at(this->pointer);}

  /// The * operator returns the current element. 
  Entry &operator *() const
    {return current();}

#if 0 
  Entry *operator ->() const
    {return &current();}
#endif

  /// Return the current element and move the pointer forwards.
  Entry& next_element() 
	{ 
	  Entry &it = this->cont->points_at(this->pointer); 
	  this->cont->move_pointer_forwards(this->pointer); 
	  return it; 
	}

  //@}
};

template <class Container, class IPointer, class Entry>
class EST_TRwStructIterator 
  : public EST_TRwIterator<Container, IPointer, Entry>
{
public:

  typedef EST_TRwStructIterator<Container, IPointer, Entry> Iter;

  /// Create an iterator not associated with any specific container.
  EST_TRwStructIterator() {this->cont=NULL;}

  /// Copy an iterator by assignment
  Iter &operator = (const Iter &orig)
    { this->cont=orig.cont; this->pos=orig.pos; this->pointer=orig.pointer; return *this;}

  /// Create an iterator ready to run over the given container.
  EST_TRwStructIterator(Container &over)
    { this->begin(over); }

  Entry *operator ->() const
    {return &this->current();}
};

#endif
