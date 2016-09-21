 /*************************************************************************/
 /*                                                                       */
 /*                Centre for Speech Technology Research                  */
 /*                     University of Edinburgh, UK                       */
 /*                      Copyright (c) 1995,1996                          */
 /*                        All Rights Reserved.                           */
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
 /*                                                                       */
 /*                       Author :  Paul Taylor                           */
 /*                       Date   :  April 1995                            */
 /* --------------------------------------------------------------------- */
 /*                     Double linked list class                          */
 /*                                                                       */
 /* Modified by RJC, 21/7/97. Now much of the working code is in the      */
 /* UList class, this template class provides a type safe front end to    */
 /* the untyped list.                                                     */
 /*                                                                       */
 /*************************************************************************/

#ifndef __Tlist_H__
#define __Tlist_H__

#include <iostream>

using namespace std;

#include "EST_common.h"
#include "EST_UList.h"
#include "EST_TSortable.h"
#include "EST_TIterator.h"

#include "instantiate/EST_TListI.h"

class EST_String;

template<class T> class EST_TList;

template<class T> class EST_TItem : public EST_UItem {
private:
  static void *operator new(size_t not_used, void *place)
    {(void)not_used; return place;}
  static void *operator new(size_t size)
    {void *p;
    p = (void *)walloc(char,size);
     return p;} 
  static void operator delete(void *p)
    { wfree(p);}

  static EST_TItem *s_free;
  static unsigned int s_nfree;
  static unsigned int s_maxFree;

protected:
  static EST_TItem *make(const T &val);
  static void release(EST_TItem<T> *it);

  friend class EST_TList<T>;
  
public:
    T val;

    EST_TItem(const T &v) : val(v)
	    { init(); };
    EST_TItem() 
	    { init();};
};

// pretty name

typedef EST_UItem EST_Litem;

/** 

A Template doubly linked list class. This class contains doubly linked
lists of a type denoted by {\tt T}. A pointer of type \Ref{EST_Litem}
is used to access items in the list. The class supports a variety of
ways of adding, removing and accessing items in the list. For examples
of how to operate lists, see \Ref{list_example}.

Iteration through the list is performed using a pointer of type
\Ref{EST_Litem}. See \Ref{Iteration} for example code.

*/

template <class T> class EST_TList : public EST_UList 
{
  private:
    void copy_items(const EST_TList<T> &l);
  public:
    void init() { EST_UList::init(); };
    static void free_item(EST_UItem *item);

  /**@name Constructor functions */
  //@{
  /// default constructor
    EST_TList() { };
  /// copy constructor
    EST_TList(const EST_TList<T> &l);
    ~ EST_TList() { clear_and_free(free_item); }
  //@}

  /**@name Access functions for reading and writing items.
    See \Ref{EST_TList_Accessing} for examples.*/

  //@{

  /** return the value associated with the EST_Litem pointer. This
    has the same functionality as the overloaded () operator.
    */
    T &item(const EST_Litem *p)     
	    { return ((EST_TItem<T> *)p) -> val; };
  /** return a const value associated with the EST_Litem pointer.*/
    const T &item(const EST_Litem *p) const	
	    { return ((EST_TItem<T> *)p) -> val; };
    /// return the Nth value
    T &nth(int n)	   
	    { return item(nth_pointer(n)); };
    /// return a const Nth value
    const T &nth(int n) const
	    { return item(nth_pointer(n)); };

  /// return const reference to first item in list
    const T &first() const			
            { return item(head()); };
  /// return const reference to last item in list
    const T &last() const			
	    { return item(tail()); };
  /** return reference to first item in list
    * @see last 
    */
    T &first()					
            { return item(head()); };
  /// return reference to last item in list
    T &last()					
	    { return item(tail()); };

  /// return const reference to item in list pointed to by {\tt ptr}
    const T  &operator () (const EST_Litem *ptr) const
	    { return item(ptr); };
  /// return non-const reference to item in list pointed to by {\tt ptr}
    T  &operator () (const EST_Litem *ptr)
 	    { return item(ptr); };

  //@}

  /**@name Removing items in a list. 
    more.
   */
  //@{
  /** remove item pointed to by {\tt ptr}, return pointer to previous item.
 See \Ref{Removing} for example code.*/
    EST_Litem *remove(EST_Litem *ptr)
	    { return EST_UList::remove(ptr, free_item); };

    /// remove nth item, return pointer to previous item
    EST_Litem *remove_nth(int n)
	    { return EST_UList::remove(n, free_item); };

  //@}


  /**@name Adding items to a list. 
    In all cases, a complete copy of
    the item is made and added to the list. See \Ref{Addition} for examples.
   */
  //@{
  /// add item onto end of list
    void append(const T &item)
	    { EST_UList::append(EST_TItem<T>::make(item)); };
  /// add item onto start of list
    void prepend(const T &item)
	    { EST_UList::prepend(EST_TItem<T>::make(item)); };

  /** add {\tt item} after position given by {\tt ptr}, return pointer
    to added item. */

    EST_Litem *insert_after(EST_Litem *ptr, const T &item)
	    { return EST_UList::insert_after(ptr, EST_TItem<T>::make(item)); };

  /** add {\tt item} before position given by {\tt ptr}, return
      pointer to added item. */

    EST_Litem *insert_before(EST_Litem *ptr, const T &item)
	    { return EST_UList::insert_before(ptr, EST_TItem<T>::make(item)); };

  //@}

  /**@name Exchange */
  //@{
  /// exchange 1
    void exchange(EST_Litem *a, EST_Litem *b)
	    { EST_UList::exchange(a, b); };
  /// exchange 2
    void exchange(int i, int j)
	    { EST_UList::exchange(i,j); };
  /// exchange 3    
   static void exchange_contents(EST_Litem *a, EST_Litem *b);
  //@}

  /**@name General functions */
  //@{
  /// make full copy of list
      EST_TList<T> &operator=(const EST_TList<T> &a); 
  /// Add list onto end of existing list
    EST_TList<T> &operator +=(const EST_TList<T> &a);

  /// print list
    friend ostream& operator << (ostream &st, EST_TList<T> const &list) {
        EST_Litem *ptr; 
        for (ptr = list.head(); ptr != 0; ptr = ptr->next()) 
            st << list.item(ptr) << " "; 
        return st;
    }

  /// remove all items in list
    void clear(void) 
	    { clear_and_free(free_item); };
  //@}

  // Iteration support

protected:
  struct IPointer {  EST_Litem *p; };

  void point_to_first(IPointer &ip) const { ip.p = head(); }
  void move_pointer_forwards(IPointer &ip) const { ip.p = ip.p->next(); }
  bool points_to_something(const IPointer &ip) const { return ip.p != NULL; }
  T &points_at(const IPointer &ip) { return item(ip.p); }

  friend class EST_TIterator< EST_TList<T>, IPointer, T >;
  friend class EST_TRwIterator< EST_TList<T>, IPointer, T >;

public:
  typedef T Entry;
  typedef EST_TIterator< EST_TList<T>, IPointer, T > Entries;
  typedef EST_TRwIterator< EST_TList<T>, IPointer, T > RwEntries;

};


template<class T> 
bool operator==(const EST_TList<T> &a, const EST_TList<T> &b)
{ 
    return EST_UList::operator_eq(a, b, EST_TSortable<T>::items_eq); 
}

template<class T> 
bool operator!=(const EST_TList<T> &a, const EST_TList<T> &b)
{ 
    return !(a==b); 
}

template<class T> 
int index(EST_TList<T> &l, T& val, bool (*eq)(const EST_UItem *, const EST_UItem *) = NULL)
{ 
  EST_TItem<T> item(val);
  return EST_UList::index(l, item, eq?eq:EST_TSortable<T>::items_eq); 
}

template<class T> 
void sort(EST_TList<T> &a, bool (*gt)(const EST_UItem *, const EST_UItem *) = NULL)
{ 
    EST_UList::sort(a, gt?gt:EST_TSortable<T>::items_gt); 
}

template<class T> 
void ptr_sort(EST_TList<T> &a)
{ 
    EST_UList::sort(a, EST_TSortable<T *>::items_gt); 
}

template<class T> 
void qsort(EST_TList<T> &a, bool (*gt)(const EST_UItem *, const EST_UItem *) = NULL)
{ 
    EST_UList::qsort(a, gt?gt:EST_TSortable<T>::items_gt, EST_TList<T>::exchange_contents); 
}

template<class T> 
void ptr_qsort(EST_TList<T> &a)
{ 
    EST_UList::qsort(a, EST_TSortable<T *>::items_gt, EST_TList<T>::exchange_contents); 
}

template<class T> 
void sort_unique(EST_TList<T> &l)
{ 
    EST_UList::sort_unique(l, 
			   EST_TSortable<T>::items_eq, 
			   EST_TSortable<T>::items_gt,
			   EST_TList<T>::free_item); 
}

template<class T> 
void merge_sort_unique(EST_TList<T> &l, EST_TList<T> &m)
{  
    EST_UList::merge_sort_unique(l, m,
				 EST_TSortable<T>::items_eq, 
				 EST_TSortable<T>::items_gt,
				 EST_TList<T>::free_item); 
}

template<class T>
const char *error_name(EST_TList<T> val) { (void)val; return "<<TLIST>>"; }

#endif
