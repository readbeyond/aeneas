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
 /*                 Author: Richard Caley (rjc@cstr.ed.ac.uk)            */
 /*                   Date: Fri Apr  4 1997                              */
 /************************************************************************/
 /*                                                                      */
 /* Simple Hash classes.                                                 */
 /*                                                                      */
 /************************************************************************/


#include "EST_THash.h"

template<class K, class V>
EST_THash<K,V>::EST_THash(int size,  unsigned int (*hash_function)(const K &key, unsigned int size))
{
    p_num_entries =0;

    p_num_buckets = size;

    p_buckets = new EST_Hash_Pair<K,V> *[size];
    for(int i=0; i<size;i++)
	p_buckets[i] = NULL;

    p_hash_function = hash_function;
}

template<class K, class V>
EST_THash<K,V>::EST_THash(const EST_THash<K,V> &from)
{
  p_buckets=NULL;
  copy(from);
}

template<class K, class V>
void EST_THash<K,V>::clear(void)
{
  if (p_buckets != NULL)
    {
      for(unsigned int i=0; i<p_num_buckets;i++)
	{
	  EST_Hash_Pair<K,V> *p, *n;
	  for(p=p_buckets[i]; p != NULL; p=n)
	    {
	      n = p->next;
	      delete p;
	    }
	  p_buckets[i]=NULL;
	}
    }
  p_num_entries=0;
}

template<class K, class V>
EST_THash<K,V>::~EST_THash(void)
{
  if (p_buckets)
    {
    clear();
    delete[] p_buckets;
    }
}


template<class K, class V>
int EST_THash<K,V>::present(const K &key) const
{
  unsigned int b;
  if (p_hash_function)
    b = (*p_hash_function)(key, p_num_buckets);
  else
    b = DefaultHashFunction((void *)&key, sizeof(key), p_num_buckets);

  EST_Hash_Pair<K,V> *p;

  for(p=p_buckets[b]; p!=NULL; p=p->next)
    if (p->k == key)
      return TRUE;
  
return FALSE;
}

template<class K, class V>
V &EST_THash<K,V>::val(const K &key, int &found) const
{
  unsigned int b;
  if (p_hash_function)
    b = (*p_hash_function)(key, p_num_buckets);
  else
    b = DefaultHashFunction((void *)&key, sizeof(key), p_num_buckets);

  EST_Hash_Pair<K,V> *p;

  for(p=p_buckets[b]; p!=NULL; p=p->next)
    if (p->k == key)
      {
	found=1;
	return p->v;
      }
  
found=0;
return Dummy_Value;
}

template<class K, class V>
const K &EST_THash<K,V>::key(const V &val, int &found) const
{

  for(unsigned int b=0; b<p_num_buckets; b++)
	{
	EST_Hash_Pair<K,V> *p;
	  for(p=p_buckets[b]; p!=NULL; p=p->next)
	    if (p->v == val)
	      {
		found=1;
		return p->k;
	      }
	}
found=0;
return Dummy_Key;
}

template<class K, class V>
void EST_THash<K,V>::map(void (*func)(K&, V&))
{
  for(unsigned int i=0; i<p_num_buckets; i++)
    {
      EST_Hash_Pair<K,V> *p;

      for(p=p_buckets[i]; p!=NULL; p=p->next)
	(*func)(p->k, p->v);
    }
  
}

template<class K, class V>
int EST_THash<K,V>::add_item(const K &key, const V &value, int no_search)
{
  unsigned int b;
  if (p_hash_function)
    b = (*p_hash_function)(key, p_num_buckets);
  else
    b = DefaultHashFunction((void *)&key, sizeof(key), p_num_buckets);

  EST_Hash_Pair<K,V> *p;

  if (!no_search)
    for(p=p_buckets[b]; p!=NULL; p=p->next)
      if (p->k == key)
	{
	  p->v = value;
	  return FALSE;
	}
      
  p = new EST_Hash_Pair<K,V>;
  p->k = key;
  p->v = value;
  p->next = p_buckets[b];
  p_buckets[b] = p;
  p_num_entries++;
  return TRUE;
}

template<class K, class V>
int EST_THash<K,V>::remove_item(const K &rkey, int quiet)
{
  unsigned int b;
  if (p_hash_function)
    b = (*p_hash_function)(rkey, p_num_buckets);
  else
    b = DefaultHashFunction((void *)&rkey, sizeof(rkey), p_num_buckets);

  EST_Hash_Pair<K,V> **p;

  for (p = &(p_buckets[b]); *p!=NULL; p=&((*p)->next))
    if ( (*p)->k == rkey )
      {
	EST_Hash_Pair<K,V> *n = (*p)->next;
	delete *p;
	*p = n;
	p_num_entries--;
	return 0;
      }
      
  if (!quiet)
    cerr << "THash: no item labelled \"" << rkey << "\"" << endl;
  return -1;
}

template<class K, class V>
EST_THash<K,V> &EST_THash<K,V>::operator = (const EST_THash<K,V> &from)
{
  copy(from);
  return *this;
}

template<class K, class V>
void EST_THash<K,V>::dump(ostream &stream, int all)
{
  for(unsigned int i=0; i<p_num_buckets; i++)
    if (all || p_buckets[i])
      {
	stream << i << ": ";
	EST_Hash_Pair<K,V> *p;
	for(p=p_buckets[i]; p!=NULL; p=p->next)
	  stream << "[" << p->k << "],(" << p->v << ") ";
	stream << "\n";
      }
}

template<class K, class V>
void EST_THash<K,V>::copy(const EST_THash<K,V> &from)
{
  clear();
  p_num_entries = from.p_num_entries;
  p_num_buckets = from.p_num_buckets;
  p_hash_function = from.p_hash_function;

  if (p_buckets != NULL)
    delete [] p_buckets;

  p_buckets = new EST_Hash_Pair<K,V> *[p_num_buckets];


  for(unsigned int b=0; b<p_num_buckets; b++)
    {
      p_buckets[b]=NULL;
      for(EST_Hash_Pair<K,V> *p=from.p_buckets[b]; p; p=p->next)
	{
	  EST_Hash_Pair<K,V> *n = new EST_Hash_Pair<K,V>(*p);
	  n->next = p_buckets[b];
	  p_buckets[b]=n;
	}
    }
}





