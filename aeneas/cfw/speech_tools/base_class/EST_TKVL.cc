/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                      Copyright (c) 1995,1996                          */
/*                        All Rights Reserved.                           */
/*                                                                       */
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
/*                                                                       */
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
/*                      Author :  Paul Taylor                            */
/*                      Date   :  May 1995                               */
/*-----------------------------------------------------------------------*/
/*                 Key/value List Class source file                      */
/*                                                                       */
/*=======================================================================*/

#include <cstdlib>
#include "EST_TKVL.h"
#include "EST_error.h"

template <class K, class V> EST_TKVL<K, V>::EST_TKVL(const EST_TKVL<K, V> &kv)
{
    list = kv.list;
}

template <class K, class V> 
void EST_TKVL<K, V>::clear(void)
{
    list.clear();
}

template <class K, class V> 
EST_Litem *EST_TKVL<K, V>::find_pair_key(const K &key) const
{
    EST_Litem *ptr;

    for (ptr = list.head(); ptr != 0; ptr= ptr->next())
	if (list.item(ptr).k == key)
	    return ptr;
    return 0;
}

template <class K, class V> 
EST_Litem *EST_TKVL<K, V>::find_pair_val(const V &val) const
{
    EST_Litem *ptr;

///    cout << "function list\n" << endl;

    for (ptr = list.head(); ptr != 0; ptr= ptr->next())
    {
//	cout << "ff: " << list.item(ptr).k << endl;
	if (list.item(ptr).v == val)
	    return ptr;
    }
    return 0;
}


// look for pointer kptr in list. If found, change its value to rval and
// return true, otherwise return false.

template <class K, class V> 
int EST_TKVL<K, V>::change_val(EST_Litem *kptr, const V &rval)
{
    if (list.index(kptr) == -1)
	return 0;
    else
    {
	list.item(kptr).v = rval;
	return 1;
    }
}

template <class K, class V> 
int EST_TKVL<K, V>::change_key(EST_Litem *kptr, const K &rkey)
{
    if (list.index(kptr) == -1)
	return 0;
    else
    {
	list.item(kptr).k = rkey;
	return 1;
    }
}

// look for key rkey in list. If found, change its value to rval and
// return true, otherwise return false.
template <class K, class V> 
int EST_TKVL<K, V>::change_val(const K &rkey,const V &rval)
{
    EST_Litem *ptr=find_pair_key(rkey);
    if (ptr == 0)
	return 0;
    else
    {
	list.item(ptr).v = rval;
	return 1;
    }
}

// NOTE: This _MUST_NOT_ change the EST_TKVL,  if it needs to, a separate
// const version will need to replace the dummy one below.

template<class K, class V> 
V &EST_TKVL<K, V>::val(const K &rkey, bool must)
{ 
    EST_Litem *ptr = find_pair_key(rkey);

    if (ptr == 0)
    {
	if (must)
	  EST_error("No value set for '%s'", error_name(rkey));

	return *default_val;
    }
    else
	return list.item(ptr).v;
}

template<class K, class V> 
const V &EST_TKVL<K, V>::val(const K &rkey, bool must) const
{ 
   return ((EST_TKVL<K, V> *)this)->val(rkey, must);
}

template<class K, class V> 
const V &EST_TKVL<K, V>::val_def(const K &rkey, const V &def) const
{ 
    EST_Litem *ptr = find_pair_key(rkey);
    if (ptr == 0)
	return def;
    else
	return list.item(ptr).v;
}

// NOTE: This _MUST_NOT_ change the EST_TKVL,  if it needs to, a separate
// const version will need to replace the dummy one below.

template<class K, class V> 
V &EST_TKVL<K, V>::val(EST_Litem *kptr, bool must)
{ 
    if (must == 0)
	return list.item(kptr).v;
    /* check kptr is one of mine */
    if (list.index(kptr) == -1)
    {
      if (must)
	EST_error("No value set in EST_TKVL");
      return *default_val;
    }
    else
	return list.item(kptr).v;
}


template<class K, class V> 
const V &EST_TKVL<K, V>::val(EST_Litem *kptr, bool must) const
{ 
  return ((EST_TKVL<K, V> *)this)->val(kptr, must);
}

// NOTE: This _MUST_NOT_ change the EST_TKVL,  if it needs to, a separate
// const version will need to replace the dummy one below.

template<class K, class V> 
K &EST_TKVL<K, V>::key(EST_Litem *kptr, int must)
{ 
    if (must == 0)
	return list.item(kptr).k;
    if (list.index(kptr) == -1)
      EST_error("No value set in EST_TKVL");

    return list.item(kptr).k;
}

template<class K, class V> 
const K &EST_TKVL<K, V>::key(EST_Litem *kptr, int must) const
{ 
  return ((EST_TKVL<K, V> *)this)->key(kptr, must);
}

template<class K, class V> 
const K &EST_TKVL<K, V>::key(const V &v, int must) const
{ 
    EST_Litem *ptr = find_pair_val(v);
    if (ptr == 0)
    {
	if (must)
	  EST_error("No value set for '%s'", error_name(v));

	return *default_key;
    }
    
    return list.item(ptr).k;
}

template<class K, class V> 
const int EST_TKVL<K, V>::present(const K &rkey) const
{ 
    if (find_pair_key(rkey) == 0)
	return 0;
    else
	return 1;
}

// map a function over the pairs

template<class K, class V>
void EST_TKVL<K,V>::map(void (*func)(K&, V&))
{
    EST_Litem *p;
    for(p=list.head(); p; p=p->next())
    {
	EST_TKVI<K,V> item = list.item(p);
	(*func)(item.k, item.v);
    }
}

// add item to list. By default, the list is searched to see if the
// item exists already. If so, its value is overwritten. This facility
// can be turned off by setting no_search = 1;

template<class K, class V> 
int EST_TKVL<K, V>::add_item(const K &rkey, const V &rval, int no_search)
{
    if (!no_search)
	if (change_val(rkey, rval)) // first see if key exists
	    return 1;
    
    EST_TKVI<K,V>  item;
    item.k = rkey;
    item.v = rval;
    
    list.append(item);
    return 1;
}

template<class K, class V> 
int EST_TKVL<K, V>::remove_item(const K &rkey, int quiet)
{
    EST_Litem *ptr = find_pair_key(rkey);
    const char *en;
    if (ptr == 0)
    {
	if (!quiet)
	{
	    en = error_name(rkey);
	    EST_warning("EST_TKVL: no item labelled '%s'", en);
	}
	return -1;
    }
    else 
    {
	list.remove(ptr);
	return 0;
    }
}

template<class K, class V> EST_TKVL<K, V> &EST_TKVL<K, V>::operator = 
(const EST_TKVL<K, V> &kv)
{
    list = kv.list;
    return *this;
}

template<class K, class V> EST_TKVL<K, V> &EST_TKVL<K, V>::operator += 
(const EST_TKVL<K, V> &kv)
{
    list += kv.list;
    return *this;
}

template<class K, class V> EST_TKVL<K, V> EST_TKVL<K, V>::operator + (const EST_TKVL<K, V> &kv)
{
    EST_TKVL<K, V> result;
    result = *this;
    result += kv;
    return result;
}

