/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                      Copyright (c) 1997,1998                          */
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
/*                 Author: Richard Caley (rjc@cstr.ed.ac.uk)             */
/*                   Date: Mon Jul 21 1997                               */
/* --------------------------------------------------------------------  */
/* Untyped list used as the basis of the TList class                     */
/*                                                                       */
/*************************************************************************/
#include <EST_UList.h>

void EST_UList::clear_and_free(void (*item_free)(EST_UItem *p))
{
    EST_UItem *q, *np;

    for (q=head(); q != 0; q = np)
    {
	np=q->next();
	if (item_free)
	    item_free(q);
	else
	    delete q;
    }
    h = t = 0;
}

int EST_UList::length() const
{
    EST_UItem *ptr;
    int n = 0;

    for (ptr = head(); ptr != 0; ptr = ptr->next())
	++n;
    return n;
}

int EST_UList::index(EST_UItem *item) const
{
    EST_UItem *ptr;
    int n = 0;
    
    for (ptr = head(); ptr != 0; ptr = ptr->next(), ++n)
	if (item == ptr)
	    return n;

    return -1;
}

EST_UItem *EST_UList::nth_pointer(int n) const
{
    EST_UItem *ptr;
    int i;
    
    for (i = 0, ptr = head(); ptr != 0; ptr = ptr->next(), ++i)
      if (i == n)
	return ptr;

    cerr << "Requested item #" << n << " off end of list" << endl;
    return head();
}

EST_UItem * EST_UList::remove(EST_UItem *item,
			      void (*free_item)(EST_UItem *item))
{
    if (item == 0)
	return 0;

    EST_UItem *prev = item->p;
    if (item->p == 0)  // at start
	h = item->n;
    else
	item->p->n = item->n;
    if (item->n == 0)  // at end
	t = item->p;
    else
	item->n->p = item->p;
	
    if (free_item)
	free_item(item);
    else
	delete item;
    
    return prev;
}

EST_UItem * EST_UList::remove(int n,
			      void (*item_free)(EST_UItem *item))
{
    return remove(nth_pointer(n), item_free);
}

// This should check if the incoming prev_item actually is in the list

EST_UItem *EST_UList::insert_after(EST_UItem *prev_item, EST_UItem *new_item)
{
    if (new_item == 0)
	return new_item;
    if (prev_item == 0)	// insert it at start of list
    {
	new_item->n = h;
	h = new_item;
    }
    else
    {
	new_item->n = prev_item->n;
	prev_item->n = new_item;
    }
    new_item->p = prev_item;
    if (new_item->n == 0)
	t = new_item;
    else
	new_item->n->p = new_item;
    
    return new_item;
}

EST_UItem *EST_UList::insert_before(EST_UItem *next_item, EST_UItem *new_item)
{
    if (new_item == 0)
	return new_item;
    if (next_item == 0)	// put it on the end of the list
    {
	new_item->p = t;
	t = new_item;
    }
    else
    {
	new_item->p = next_item->p;
	next_item->p = new_item;
    }
    new_item->n = next_item;
    if (new_item->p == 0)
	h = new_item;
    else
	new_item->p->n = new_item;
    
    return next_item;
}

void EST_UList::exchange(EST_UItem *a, EST_UItem *b)
{

    if (a==b)
	return;
    
    if ((a==0) || (b==0))
    {
	cerr << "EST_UList:exchange: can't exchange NULL items" << endl;
	return;
    }

    // I know this isn't very readable but there are eight pointers
    // that need to be changed, and half of them are trivial back pointers
    // care need only be taken when b and a are adjacent, this actual
    // sets p and n twice if they are adjacent but still gets the right answer
    EST_UItem *ap=a->p,*an=a->n,*bn=b->n,*bp=b->p;

    a->n = bn == a ? b : bn;
    if (a->n)
      a->n->p = a;
    a->p = bp == a ? b : bp;
    if (a->p)
      a->p->n = a;

    b->n = an == b ? a : an;
    if (b->n)
      b->n->p = b;
    b->p = ap == b ? a : ap;
    if (b->p)
      b->p->n = b;

    // Fix t and h 
    if (a == h)
	h = b;
    else if (b == h)
	h = a;
    else if (a == t)
	t = b;
    else if (b == t)
	t = a;
	     
}

void EST_UList::exchange(int i, int j)
{
    
    EST_UItem *p;
    EST_UItem *a=0,*b=0;
    int k;
    
    for (k=0,p = head(); p != 0; p = p->next(),k++)
    {
	if(i==k)
	    a = p;
	if(j==k)
	    b = p;
    }
    
    if ((a==0) || (b==0))
    {
	cerr << "EST_UList:exchange: can't exchange items " << i << 
	    " and " << j << " (off end of list)" << endl;
	return;
    }
    
    exchange(a,b);
}

void EST_UList::reverse()
{
    EST_UItem *p,*q;

    for (p=head(); p != 0; p=q)
    {
	q = p->n;
	p->n = p->p;
	p->p = q;
    }
    q = h;
    h = t;
    t = q;
}

void EST_UList::append(EST_UItem *new_item)
{

    if (new_item == 0) return;

    new_item->n = 0;
    new_item->p = t;
    if (t == 0)
	h = new_item;
    else
	t->n = new_item;
    t = new_item;
}

void EST_UList::prepend(EST_UItem *new_item)
{
    if (new_item == 0) return;

    new_item->p = 0;
    new_item->n = h;
    if (h == 0)
	t = new_item;
    else
	h->p = new_item;
    h = new_item;
}

bool EST_UList::operator_eq(const EST_UList &a, 
			    const EST_UList &b, 
			    bool (*eq)(const EST_UItem *item1, const EST_UItem *item2))
{
    EST_UItem *p,*q;
    q=b.head();
    for (p = a.head(); p != NULL; p = p->next()){
	if(q == NULL)
	    return false;
	if(eq(q, p))
	    q=q->next();
	else
	    return false;
    }
    
    if(q == NULL)
	return true;
    else
	return false;
}

int EST_UList::index(const EST_UList &l, 
		     const EST_UItem &val, 
		     bool (*eq)(const EST_UItem *item1, const EST_UItem *item2))
{
    
    EST_UItem *ptr;
    int n = 0;
    
    for (ptr = l.head(); ptr != 0; ptr = ptr->next(), ++n)
	if (eq(&val,ptr))
	    return n;
    
    return -1;
}

void EST_UList::sort(EST_UList &l,
		     bool (*gt)(const EST_UItem *item1, 
				const EST_UItem *item2))
{
    
    // just bubble sort for now
    // use EST_String::operator > for comparisons
    
    EST_UItem *l_ptr,*m_ptr;
    bool sorted=false;
    
    while(!sorted){
	sorted=true;
	
	for(l_ptr=l.head(); l_ptr != 0; l_ptr=l_ptr->next()){
	    
	    m_ptr=l_ptr->next();
	    if(m_ptr != 0)
		if(gt(l_ptr, m_ptr)){
		    l.exchange(l_ptr,m_ptr);
		    sorted=false;
		}
	}
    }
    
}

// quicksort from 'Algorithms'
// by Cormen, Leiserson & Rivest

static EST_UItem *partition(EST_UItem *p, EST_UItem *r,
			    bool (*gt)(const EST_UItem *item1, const EST_UItem *item2),
			    void (*exchange)(EST_UItem *item1, EST_UItem *item2))
{
    // this can be tidied up / sped up
    
    EST_UItem *i,*j,*i2,*j2;
    EST_UItem *x = p;
    
    i = p;
    j = r;
    
    while(true){
	
	while(gt(j, x) )
	    j = j->prev();
	
	while(gt(x, i))
	    i = i->next();
	
	if((i != j) && (i->prev() != j)){
	    i2=i;
	    j2=j;
	    i=i->next();
	    j=j->prev();
	    exchange(i2,j2);
	    
	}else
	    return j;
	
    }
    return NULL;
}


static void qsort_sub(EST_UList &l, EST_UItem *p, EST_UItem *r,
		      bool (*gt)(const EST_UItem *item1, const EST_UItem *item2),
		      void (*exchange)(EST_UItem *item1, EST_UItem *item2))
{
    EST_UItem *q;
    if(p != r){
	q = partition(p,r, gt, exchange);
	qsort_sub(l,p,q, gt, exchange);
	qsort_sub(l,q->next(),r, gt, exchange);
    }
}

void EST_UList::qsort(EST_UList &l,
		      bool (*gt)(const EST_UItem *item1, const EST_UItem *item2),
		      void (*exchange)(EST_UItem *item1, EST_UItem *item2))
{
    qsort_sub(l,l.head(),l.tail(), gt, exchange);
}


void EST_UList::sort_unique(EST_UList &l,
			    bool (*eq)(const EST_UItem *item1, const EST_UItem *item2),
			    bool (*gt)(const EST_UItem *item1, const EST_UItem *item2),
			    void (*item_free)(EST_UItem *item))
{
    // as sort(..) but delete any repeated items
    
    EST_UItem *l_ptr,*m_ptr;
    bool sorted=false;
    
    while(!sorted){
	sorted=true;
	
	for(l_ptr=l.head(); l_ptr != 0; l_ptr=l_ptr->next()){
	    
	    m_ptr=l_ptr->next();
	    if(m_ptr != 0)
            {
		if(gt(l_ptr, m_ptr)){
		    l.exchange(l_ptr,m_ptr);
		    sorted=false;
		} else if(eq(l_ptr,  m_ptr)){
		    l.remove(m_ptr, item_free);
		    sorted=false;
		}
            }
	}
    }
}

void EST_UList::merge_sort_unique(EST_UList &l, EST_UList &m,
				  bool (*eq)(const EST_UItem *item1, const EST_UItem *item2),
				  bool (*gt)(const EST_UItem *item1, const EST_UItem *item2),
				  void (*item_free)(EST_UItem *item))
{
    // keep all unique items in l, and add any new items from m to l
    
    EST_UItem *l_ptr,*m_ptr;
    bool flag;
    
    // make sure
    sort_unique(l, eq, gt, item_free);
    
    for(m_ptr=m.head(); m_ptr != 0; m_ptr=m_ptr->next()){
	
	// try and put item from m in list 
	flag=false;
	for(l_ptr=l.head(); l_ptr != 0; l_ptr=l_ptr->next()){
	    if( gt(l_ptr, m_ptr) ){
		l.insert_before(l_ptr, m_ptr);
		flag=true;
		break;
	    }else if( eq(m_ptr, l_ptr) ){
		flag=true;
		break;
	    }
	}
	// or try and append it
	if(!flag && ( gt(m_ptr, l.tail()) ) )
	    l.append(m_ptr);
    }
}
