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
/*                    Author :  Paul Taylor                              */
/*                    Date   :  May 1995 (redone May 1998)               */
/*-----------------------------------------------------------------------*/
/*                  EST_Utterance Class header file                      */
/*                                                                       */
/*=======================================================================*/
#ifndef __Utterance_H__
#define __Utterance_H__

#include "EST_String.h"
#include "EST_TList.h"
#include "ling_class/EST_Relation.h"
#include "ling_class/EST_Item.h"
#include "EST_Features.h"

/** A class that contains <link linkend="est-item">EST_Items</link>
and <link linkend="est-relation">EST_Relations</link> between them.
Used for holding interrelated linguistic structures.

*/

class EST_Utterance{
private:
    void copy(const EST_Utterance &u);
    int highest_id;
public:
    /**@name Constructor and initialisation Functions */
    //@{
    /// default constructor
    EST_Utterance();
    EST_Utterance(const EST_Utterance &u) { copy(u); }
    ~EST_Utterance() {clear();}
    //@}
    ///

    /**@name Utility Functions */
    //@{
    ///  initialise utterance
    void init();

    /// remove everything in utterance
    void clear();

    /// clear the contents of the relations only
    void clear_relations();

    /// set the next id to be <parameter>n</parameter>
    void set_highest_id(int n) {highest_id=n;}
    /// return the id of the next item
    int next_id();
    //@}

    /**@name File i/o */
    //@{
    /** load an utterance from an ascii file
      */
    EST_read_status load(const EST_String &filename);
    /** load an utterance from a already opened token stream
      */
    EST_read_status load(EST_TokenStream &ts);

    /** save an utterance to an ascii file
      */
    EST_write_status save(const EST_String &filename,
			  const EST_String &type="est_ascii") const;

    /** save an utterance to an ostream
      */
    EST_write_status save(ostream &outf,const EST_String &type) const;
    //@}  

    EST_Utterance &operator=(const EST_Utterance &s);
    friend ostream& operator << (ostream &s, const EST_Utterance &u);
    EST_Relation * operator() (const EST_String &name)
    { return relation(name);}

    /** Utterance access
     */
    //@{

    /// Utterance level features
    EST_Features f;

    /// Evaluate all feature functions in utterance
    void evaluate_all_features();
    
    /// The list of named relations
    EST_Features relations;

    /// number of relations in this utterance
    int num_relations() const { return relations.length(); }

    /** returns true if utterance contains named relations.
      {\bf name} can be either a single string or a bracketed list
      of strings e.g. "(Word Phone Syl)".  */
    bool relation_present(const EST_String name) const;

    /** returns true if utterance contains all the relations
      named in the list {\bf name}. */
    bool relation_present(EST_StrList &names) const;

    /// get relation by name
    EST_Relation *relation(const char *name,int err_on_not_found=1) const;

    /// return EST_Item whose id is <parameter>n</parameter>.
    EST_Item *id(const EST_String &n) const;

    /// create a new relation called <parameter>n</parameter>.
    EST_Relation *create_relation(const EST_String &relname);

    /// remove the relation called <parameter>n</parameter>.
    void remove_relation(const EST_String &relname);
		
    void sub_utterance(EST_Item *i);
};

void utt_2_flat_repr( const EST_Utterance &utt,
		      EST_String &flat_repr );

int utterance_merge(EST_Utterance &utt,
		    EST_Utterance &sub_utt,
		    EST_Item *utt_root,
		    EST_Item *sub_root);

int utterance_merge(EST_Utterance &utt,
		    EST_Utterance &extra,
		    EST_String feature);

void sub_utterance(EST_Utterance &sub,EST_Item *i);

#endif
