/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                         Copyright (c) 1998                            */
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
/*                       Author :  Alan W Black                          */
/*                       Date   :  February 1998                         */
/* --------------------------------------------------------------------- */
/*                        another architecture                           */
/*                                                                       */
/*************************************************************************/
#ifndef __EST_RELATION_H__
#define __EST_RELATION_H__

#include "EST_String.h"
#include "EST_TList.h"
#include "EST_TKVL.h"
#include "EST_THash.h"
#include "EST_Val.h"
#include "EST_types.h"
#include "EST_Token.h"
#include "EST_Features.h"
#include "ling_class/EST_Item.h"

class EST_Utterance;

class EST_Relation_Iterator;

/** Relations are a container class for EST_Items. Three types of
relation structure are supported:

<variablelist>

<varlistentry><term>Linear lists</term><listitem></listitem></varlistentry>
<varlistentry><term>Trees</term><listitem></listitem></varlistentry>
<varlistentry><term>Multi-linear structures</term><listitem> as used
in autosegmental phonology etc</listitem></varlistentry>

</variablelist>
*/

class EST_Relation
{
    EST_String p_name;
    EST_Utterance *p_utt;
    EST_Item *p_head;   
    EST_Item *p_tail;   // less meaningful in a tree

    EST_Item *get_item_from_name(EST_THash<int,EST_Val> &inames,int name);
    EST_Item *get_item_from_name(EST_TVector< EST_Item * > &inames,int name);
    EST_write_status save_items(EST_Item *item, 
				ostream &outf,
				EST_TKVL<void *,int> &contentnames,
				EST_TKVL<void *,int> &itemnames,
				int &node_count) const;

    static void node_tidy_up_val(int &k, EST_Val &v);
    static void node_tidy_up(int &k, EST_Item *node);

    EST_read_status load_items(EST_TokenStream &ts,
			       const EST_THash<int,EST_Val> &contents);
    EST_read_status load_items(EST_TokenStream &ts,
			       const EST_TVector < EST_Item_Content * > &contents
			       );
    void copy(const EST_Relation &r);
  public:

    /** default constructor */
    EST_Relation();
    /** Constructor which sets name of relation */
    EST_Relation(const EST_String &name);
    /** Constructor which copies relation r */
    EST_Relation(const EST_Relation &r) { copy(r); }
    /** default destructor */
    ~EST_Relation();

    /** Features which belong to the relation rather than its items*/
    EST_Features f;

    /** Evaluate the relation's feature functions */
    //void evaluate_features();
    /** Evaluate the feature functions of all the items in the relation */
    void evaluate_item_features();

    /** Clear the relation of items */
    void clear();

    /** Return the <link linkend="est-utterance">EST_Utterance</link>
	to which this relation belongs */
    EST_Utterance *utt(void) { return p_utt; }

    /** Set the <link linkend="est-utterance">EST_Utterance</link>
	to which this relation belongs */
    void set_utt(EST_Utterance *u) { p_utt = u; }

    /** Return the name of the relation */
    const EST_String &name() const { return (this == 0) ? EST_String::Empty : p_name; }

    /** Return the head (first) item of the relation */
    EST_Item *head() const {return (this == 0) ? 0 : p_head;}

    /** Return the root item of the relation */
    EST_Item *root() const {return head();}

    /** Return the tail (last) item of the relation */
    EST_Item *tail() const {return (this == 0) ? 0 : p_tail;}

    // This have been replaced by Relation_Tree functions
    EST_Item *first() const { return head(); }
    EST_Item *first_leaf() const;
    EST_Item *last() const { return tail(); }
    EST_Item *last_leaf() const;

    /** Return the tail (last) item of the relation */
//    EST_Item *id(int i);

    /** number of items in this relation */
    int length() const;
//    int num_nodes() const;
//    int num_leafs() const;
    /** return true if relation does not contain any items */
    int empty() const { return p_head == 0; }

    /** remove EST_Item <parameter>item</parameter> from relation */
    void remove_item(EST_Item *item);

    /** remove all occurrences of feature 
	<parameter>name</parameter> from relation's items
    */
    void remove_item_feature(const EST_String &name);

    /** Load relation from file */
    EST_read_status load(const EST_String &filename,
			 const EST_String &type="esps");

    /** Load relation from already open tokenstream */
//    EST_read_status load(EST_TokenStream &ts,
//			 const EST_THash<int,EST_Val> &contents);

    /** Load relation from already open tokenstream */
    EST_read_status load(EST_TokenStream &ts,
			 const EST_TVector < EST_Item_Content * > &contents
			 );

    /** Load relation from already open tokenstream */
    EST_read_status load(const EST_String &filename,
			 EST_TokenStream &ts,
			 const EST_String &type);

    /** Save relation to file */
    EST_write_status save(const EST_String &filename, 
			  bool evaluate_ff = false) const;

    /** Save relation to file, evaluating all feature functions before hand */
    EST_write_status save(const EST_String &filename, 
			  const EST_String &type,
			  bool evaluate_ff = false) const;

    /** Save relation from already open ostream */
    EST_write_status save(ostream &outf,EST_TKVL<void *,int> contents) const;

    /** Save relation from already open ostream */
    EST_write_status save(ostream &outf,
			  const EST_String &type,
			  bool evaluate_ff) const;
    /** Iteration */
    typedef EST_Relation_Iterator Iterator;

    EST_Relation &operator=(const EST_Relation &s);
    friend ostream& operator << (ostream &s, const EST_Relation &u);

    EST_Item *append(EST_Item *si);
    EST_Item *append(); 
    EST_Item *prepend(EST_Item *si);
    EST_Item *prepend(); 

    friend class EST_Item;
};

VAL_REGISTER_CLASS_DCLS(relation,EST_Relation)

inline bool operator==(const EST_Relation &a, const EST_Relation &b)
   { return (&a == &b); }

void copy_relation(const EST_Relation &from, EST_Relation &to);

EST_Utterance *get_utt(EST_Item *s);

class EST_Relation_Iterator
{
private:
  const EST_Relation &rel;
  EST_Item *next;

public:
  EST_Relation_Iterator(const EST_Relation &r)
    : rel(r), next(NULL) { reset();};

  void reset() 
    { next=rel.head(); }
  bool has_more_elements()
    { return next != NULL; }

  EST_Item *next_element();
};

typedef EST_TList<EST_Relation> EST_RelationList;

#endif
