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
 /************************************************************************/

#ifndef __EST_THASH_H__
#define __EST_THASH_H__

#include <iostream>

using namespace std;

#include "EST_String.h"
#include "EST_system.h"
#include "EST_bool.h"
#include "EST_TIterator.h"

#include "instantiate/EST_THashI.h"
#include "instantiate/EST_TStringHashI.h"

/**@name Hash Tables
  *
  * @author Richard Caley <rjc@cstr.ed.ac.uk>
  * @version $Id: EST_THash.h,v 1.6 2004/09/29 08:24:17 robert Exp $
  */
//@{

/** This is just a convenient place to put some useful hash functions.
  */
class EST_HashFunctions {
public:
  /// A generally useful hash function.
  static unsigned int DefaultHash(const void *data, size_t size, unsigned int n);

  /// A hash function for strings.
  static  unsigned int StringHash(const EST_String &key, unsigned int size);
};

template<class K, class V>  class EST_THash;

/** This class is used in hash tables to hold a key value pair.
  * Not much to say beyond that.
  */
template<class K, class V>
class EST_Hash_Pair {
public:
  /// The key
  K k;
  /// The value
  V v;

private:
  /// Pointer used to chain entries into buckets.
  EST_Hash_Pair<K,V> *next;

  /// The hash table must be a friend so it can see the pointer.
  friend class EST_THash<K, V>;
};

/** An open hash table. The number of buckets should be set to allow
  * enough space that there are relatively few entries per bucket on
  * average.
  */

template<class K, class V>
class EST_THash : protected EST_HashFunctions {

private:
  /// Something to return when there is no entry in the table.
  static V Dummy_Value;
  static K Dummy_Key;

  /// Total number of entries.
  unsigned int p_num_entries;

  /// Number of buckets.
  unsigned int p_num_buckets;

  /// Pointer to an array of <variable>p_num_buckets</variable>buckets.
  EST_Hash_Pair<K,V> **p_buckets;

  /// The hash function to use on this table.
  unsigned int (*p_hash_function)(const K &key, unsigned int size);

public:
  /** Create a table with the given number of buckets. Optionally setting
    * a custom hash function.
    */
  EST_THash(int size,  
	    unsigned int (*hash_function)(const K &key, unsigned int size)= NULL);

  /// Create a copy
  EST_THash(const EST_THash<K,V> &from);

  /// Destroy the table.
  ~EST_THash(void);

  /// Empty the table.
  void clear(void);

  /// Return the total number of entries in the table.
  unsigned int num_entries(void) const 
    { return p_num_entries; };

  /// Does the key have an entry?
  int present(const K &key) const;

  /** Return the value associated with the key.
    * <parameter>found</parameter> is set to whether such an entry was found.
    */
  V &val(const K &key, int &found) const;

  /// Return the value associated with the key.
  V &val(const K &key) const {int x; return val(key, x); }

  const K &key(const V &val, int &found) const;
  const K &key(const V &val) const {int x; return key(val, x); }

  /// Copy all entries
  void copy(const EST_THash<K,V> &from);

  /// Apply <parameter>func</parameter> to each entry in the table.
  void map(void (*func)(K&, V&));
    
  /// Add an entry to the table.
  int add_item(const K &key, const V &value, int no_search = 0);

  /// Remove an entry from the table.
  int remove_item(const K &rkey, int quiet = 0);

  /// Assignment is a copy operation
  EST_THash<K,V> &operator = (const EST_THash<K,V> &from);

  /// Print the table to <parameter>stream</parameter> in a human readable format.
  void dump(ostream &stream, int all=0);

  /**@name Pair Iteration
    *
    * This iterator steps through the table returning key-value pairs. 
    */
  //@{
protected:
  /** A position in the table is given by a bucket number and a
    * pointer into the bucket.
    */
    // struct IPointer{  unsigned int b; EST_Hash_Pair<K, V> *p; };
    struct IPointer_s{  unsigned int b; EST_Hash_Pair<K, V> *p; };

    typedef struct IPointer_s IPointer;

  /// Shift to point at something.
  void skip_blank(IPointer &ip) const 
    {
      while (ip.p==NULL && ip.b<p_num_buckets)
	{ip.b++; ip.p = ip.b<p_num_buckets?p_buckets[ip.b]:0; } 
    }
  
  /// Go to start of the table.
  void point_to_first(IPointer &ip) const 
    { ip.b=0; ip.p = ip.b<p_num_buckets?p_buckets[ip.b]:0; 
    skip_blank(ip);}

  /// Move pointer forwards, at the end of the bucket, move down.
  void move_pointer_forwards(IPointer &ip) const 
    { 
      ip.p = ip.p->next; 
      skip_blank(ip);
    }

  /// We are at the end if the pointer ever becomes NULL
  bool points_to_something(const IPointer &ip) const { return ip.b<p_num_buckets; }

  /// Return the contents of this entry.
  EST_Hash_Pair<K, V> &points_at(const IPointer &ip) { return *(ip.p); }

  /// The iterator must be a friend to access this private interface.
  friend class EST_TStructIterator< EST_THash<K, V>, IPointer, EST_Hash_Pair<K, V> >;
  friend class EST_TRwStructIterator< EST_THash<K, V>, IPointer, EST_Hash_Pair<K, V> >;
  friend class EST_TIterator< EST_THash<K, V>, IPointer, EST_Hash_Pair<K, V> >;
  friend class EST_TRwIterator< EST_THash<K, V>, IPointer, EST_Hash_Pair<K, V> >;

public:
  /// An entry returned by the iterator is a key value pair.
  typedef EST_Hash_Pair<K, V> Entry;

  /// Give the iterator a sensible name.
  typedef EST_TStructIterator< EST_THash<K, V>, IPointer, EST_Hash_Pair<K, V> > Entries;
  typedef EST_TRwStructIterator< EST_THash<K, V>, IPointer, EST_Hash_Pair<K, V> > RwEntries;
  //@}

  /**@name Key Iteration
    *
    * This iterator steps through the table returning just keys.
    */
  //@{
protected:
  /** A position in the table is given by a bucket number and a
    * pointer into the bucket.
    */
  struct IPointer_k_s {  unsigned int b; EST_Hash_Pair<K, V> *p; };

  typedef struct IPointer_k_s IPointer_k;

  /// Shift to point at something.
  void skip_blank(IPointer_k &ip) const 
    {
      while (ip.p==NULL && ip.b<p_num_buckets)
	{ip.b++; ip.p = ip.b<p_num_buckets?p_buckets[ip.b]:0; } 
    }
  
  /// Go to start of the table.
  void point_to_first(IPointer_k &ip) const 
    { ip.b=0; ip.p = ip.b<p_num_buckets?p_buckets[ip.b]:0; 
    skip_blank(ip);}

  /// Move pointer forwards, at the end of the bucket, move down.
  void move_pointer_forwards(IPointer_k &ip) const 
    { 
      ip.p = ip.p->next; 
      skip_blank(ip);
    }

   /// We are at the end if the pointer ever becomes NULL
  bool points_to_something(const IPointer_k &ip) const { return ip.b<p_num_buckets; }

  /// Return the key of this entry.
  K &points_at(const IPointer_k &ip) { return (ip.p)->k; }

  /// The iterator must be a friend to access this private interface.
  friend class EST_TIterator< EST_THash<K, V>, IPointer_k, K >;
  friend class EST_TRwIterator< EST_THash<K, V>, IPointer_k, K >;

public:
  /// An entry returned by this iterator is just a key.
  typedef K KeyEntry;

  /// Give the iterator a sensible name.
  typedef EST_TIterator< EST_THash<K, V>, IPointer_k, K > KeyEntries;
  typedef EST_TRwIterator< EST_THash<K, V>, IPointer_k, K > KeyRwEntries;
  //@}

};

/** A specialised hash table for when the key is an EST_String.
  *
  * This is just a version of <classname>EST_THash</classname> which
  * has a different default hash function.
  */

template<class V>
class EST_TStringHash : public EST_THash<EST_String, V> {
public:

  /// Create a string hash table of <parameter>size</parameter> buckets.
  EST_TStringHash(int size) : EST_THash<EST_String, V>(size, EST_HashFunctions::StringHash) {};

  /// An entry returned by the iterator is a key value pair.
  typedef EST_Hash_Pair<EST_String, V> Entry;

/*    struct IPointer_s{  unsigned int b; Entry *p; };
      typedef struct IPointer_s IPointer; */

  // Convince GCC that the IPointer we're going to use is a typename
  typedef typename EST_THash<EST_String, V>::IPointer TN_IPointer;

  /// Give the iterator a sensible name.
   typedef EST_TStructIterator< EST_THash<EST_String, V>, typename EST_THash<EST_String, V>::IPointer,
 				    EST_Hash_Pair<EST_String, V> > Entries;
  
   typedef EST_TRwStructIterator< EST_THash<EST_String, V>, typename EST_THash<EST_String, V>::IPointer,
 				    EST_Hash_Pair<EST_String, V> > RwEntries;
  //@}

  typedef EST_String KeyEntry;

/*  struct IPointer_k_s {  unsigned int b; EST_Hash_Pair<EST_String, V> *p; };
    typedef struct IPointer_k_s IPointer_k; */

  /// Give the iterator a sensible name.

  // Convince GCC that the IPointer_k we're going to use is a typename 
  typedef typename EST_THash<EST_String, V>::IPointer_k TN_IPointer_k;

  typedef EST_TIterator< EST_THash<EST_String, V>, typename EST_THash<EST_String, V>::IPointer_k,
 			    EST_String > KeyEntries;
  typedef EST_TRwIterator< EST_THash<EST_String, V>, typename EST_THash<EST_String, V>::IPointer_k,
 			    EST_String > KeyRwEntries;
};


/** The default hash function used by <classname>EST_THash</classname>
  */
inline static unsigned int DefaultHashFunction(const void *data, size_t size, unsigned int n)
{
  unsigned int x=0;
  const char *p = (const char *)data;
  for(; size>0 ; p++, size--)
      x = ((x+*p)*33) % n;
  return x;
}

//@}
#endif
