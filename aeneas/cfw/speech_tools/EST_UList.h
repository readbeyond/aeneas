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
 /*                   Date: Mon Jul 21 1997                               */
 /* --------------------------------------------------------------------  */
 /* Untyped list used as the basis of the TList class                     */
 /*                                                                       */
 /*************************************************************************/

#ifndef __EST_ULIST_H__
#define __EST_ULIST_H__

#include <iostream>

using namespace std;

#include "EST_common.h"
#include "EST_String.h"

class EST_UItem {
public:
    void init() { n = NULL; p = NULL;}
    EST_UItem *n;
    EST_UItem *p;
    EST_UItem *next() { return n; }
    EST_UItem *prev() { return p; }
};

class EST_UList {
protected: 
    EST_UItem *h;
    EST_UItem *t;

protected:
    void init() { h = NULL; t = NULL; };
    void clear_and_free(void (*item_free)(EST_UItem *item));

public:
    EST_UList() { init(); };
    ~ EST_UList() { clear_and_free(NULL); }

    EST_UItem *nth_pointer(int n) const;

    EST_UItem *insert_after(EST_UItem *ptr, EST_UItem *new_item); // returns pointer to inserted item
    EST_UItem *insert_before(EST_UItem *ptr, EST_UItem *new_item); // returns pointer to item after inserted item

    // remove single item, return pointer to previous
    EST_UItem *remove(EST_UItem *ptr, void (*item_free)(EST_UItem *item));
    EST_UItem *remove(int n,	      void (*item_free)(EST_UItem *item));

    void exchange(EST_UItem *a, EST_UItem *b);
    void exchange(int i, int j);
    
    void reverse();			// in place

    int length() const;			// number of items in list
    int index(EST_UItem *item) const;	// position from start of list (head = 0)
	
    int empty()	const			// returns true if no items in list
	    {return (h == NULL) ? 1 : 0;};
    void clear(void) 
	    { clear_and_free(NULL); };
    void append(EST_UItem *item);	// add item onto end of list

    void prepend(EST_UItem *item);	// add item onto start of list

    EST_UItem *head() const		// return pointer to head of list
	    { return h; };
    EST_UItem *tail() const		// return pointer to tail of list
	    { return t; };


  static bool operator_eq(const EST_UList &a, 
			  const EST_UList &b, 
			  bool (*eq)(const EST_UItem *item1, const EST_UItem *item2));

  static int index(const EST_UList &l, 
			  const EST_UItem &b, 
			  bool (*eq)(const EST_UItem *item1, const EST_UItem *item2));

  static void sort(EST_UList &a,
		   bool (*gt)(const EST_UItem *item1, const EST_UItem *item2));
  static void qsort(EST_UList &a,
		    bool (*gt)(const EST_UItem *item1, const EST_UItem *item2),
		   void (*exchange)(EST_UItem *item1, EST_UItem *item2));
  static void sort_unique(EST_UList &l,
		   bool (*eq)(const EST_UItem *item1, const EST_UItem *item2),
		   bool (*gt)(const EST_UItem *item1, const EST_UItem *item2),
		   void (*item_free)(EST_UItem *item));
  static void merge_sort_unique(EST_UList &l, EST_UList &m,
		   bool (*eq)(const EST_UItem *item1, const EST_UItem *item2),
		   bool (*gt)(const EST_UItem *item1, const EST_UItem *item2),
		   void (*item_free)(EST_UItem *item));
};


// inline functions in header file
// everything else in EST_UList.C


/* Please don't use these - use the member functions instead!
inline EST_UItem *next(EST_UItem *ptr)
{
    if (ptr != 0)
	return ptr->n;
    return 0;
}

inline EST_UItem *prev(EST_UItem *ptr)
{
    if (ptr != 0)
	return ptr->p;
    return 0;
}
*/


#endif
