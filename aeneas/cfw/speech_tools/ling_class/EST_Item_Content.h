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
/*                    Author :  Alan W Black                             */
/*                    Date   :  May 1998                                 */
/*-----------------------------------------------------------------------*/
/*                                                                       */
/*  The shared part of an EST_Item, containing the linguistic            */
/*  contents, such as part of speech, stress etc.  Basically holds a     */
/*  list of feature value pairs are required.                            */
/*                                                                       */
/*  This class is effectively private to the EST_Item class and          */
/*  shouldn't be referenced outside that clase                           */
/*                                                                       */
/*=======================================================================*/
#ifndef __EST_ITEM_CONTENT_H__
#define __EST_ITEM_CONTENT_H__

#include "EST_String.h"
#include "EST_Features.h"

VAL_REGISTER_CLASS_DCLS(icontent,EST_Item_Content)
VAL_REGISTER_CLASS_DCLS(item,EST_Item)
class EST_Item;

/** A class for containing individual linguistic features and references
to relations.

This class contents the potentially shared part of an \Ref{EST_Item}.
It contains a list of features allowing string names to be related to
string, floats, ints and arbitrary objects.  It also contains a reference
list to the \Ref{EST_Item}s indexed by the relation names.

This class should not normally be accessed by anyone other than the
\Ref{EST_Item}.

*/
class EST_Item_Content {
 private:
    void copy(const EST_Item_Content &x);
 public:

    /**@name Constructor Functions */
    //@{
    /// Default constructor
    EST_Item_Content() {}
    /// Copy constructor
    EST_Item_Content(const EST_Item_Content &content) { copy(content); }
    /// destructor
    ~EST_Item_Content();
    //@}

    /// General features for this item
    EST_Features f;

    /** return the name of the item, e.g. the name of the phone or the
	text of the word*/

    // this should be changed back to a reference once we figure
    // out how to do it.
    const EST_String name() const {return f.S("name");}
    
    /// set name
    void set_name(const EST_String &s) {f.set("name",s);}

    // Links to ling_items that share this information
    EST_TKVL<EST_String, EST_Val> relations;

    EST_Item_Content& operator = (const EST_Item_Content& a);
    friend ostream& operator << (ostream &s, const EST_Item_Content &a);

    /**@name Relation related member functions */
    //@{

    EST_Item *Relation(const char *name)
    {
	EST_Item *d = 0;
	return ::item(relations.val_def(name,est_val(d)));
    }

    int in_relation(const EST_String &name) const 
         { return relations.present(name); }
    // Unreference this ling_content from the named relation
    int unref_relation(const EST_String &relname);
    // Unreference this ling_content from all relations and delete
    int unref_and_delete();
    //@}

    friend class EST_Item;
};

#endif

