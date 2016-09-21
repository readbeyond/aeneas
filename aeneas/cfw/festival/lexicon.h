/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                       Copyright (c) 1996,1997                         */
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
/*                     Author :  Alan W Black                            */
/*                     Date   :  April 1996                              */
/*-----------------------------------------------------------------------*/
/*                                                                       */
/*  Shared lexicon utilities                                             */
/* Top level form (simply an s-expression of the input form              */
/*                                                                       */
/*=======================================================================*/
#ifndef __LEXICON_H__
#define __LEXICON_H__

#include "EST_Pathname.h"

enum lex_type_t {lex_external, lex_internal};
class Lexicon{
 private:
    lex_type_t type;
    EST_String name;
    EST_String ps_name;
    LISP addenda;  // for personal local changes
    LISP posmap;
    int comp_num_entries;
    EST_Pathname bl_filename;
    FILE *binlexfp;
    EST_String lts_method;
    EST_String lts_ruleset;
    int blstart;
    LISP index_cache;
    void binlex_init(void);
    LISP lookup_addenda(const EST_String &word, LISP features);
    LISP lookup_complex(const EST_String &word, LISP features);
    LISP lookup_lts(const EST_String &word, LISP features);
    LISP bl_bsearch(const EST_String &word,LISP features,
		    int start,int end,int depth);
    LISP bl_find_next_entry(int pos);
    LISP bl_find_actual_entry(int pos,const EST_String &word,LISP features);
    int lex_entry_match;
    LISP matched_lexical_entries;
public:
    LISP pre_hooks;
    LISP post_hooks;

    Lexicon();
    ~Lexicon();
    const EST_String &lex_name() const {return name;}
    const EST_String &phoneset_name() const {return ps_name;}
    void set_lex_name(const EST_String &p) {name = p;}
    const EST_String &get_lex_name(void) const { return name; }
    void set_phoneset_name(const EST_String &p) {ps_name = p;}
    void set_lts_method(const EST_String &p) {lts_method = p;}
    void set_lts_ruleset(const EST_String &p) {lts_ruleset = p;}
    void set_pos_map(LISP p) {posmap = p;}
    const EST_String &get_lts_ruleset(void) const { return lts_ruleset; }
    void set_bl_filename(const EST_String &p) 
       {bl_filename = p;
        if (binlexfp != NULL) fclose(binlexfp);
	binlexfp=NULL;}
    void add_addenda(LISP entry) {addenda = cons(entry,addenda);}
    LISP lookup(const EST_String &word,const  LISP features);
    LISP lookup_all(const EST_String &word);
    EST_String str_lookup(const EST_String &word,const  LISP features);
    int in_lexicon(const EST_String &word,LISP features);
    int num_matches() { return lex_entry_match; }
    void bl_lookup_cache(LISP cache, const EST_String &word, 
			 int &start, int &end, int &depth);
    void add_to_cache(LISP index_cache,
		      const EST_String &word,
		      int start,int mid, int end);

    inline friend ostream& operator<<(ostream& s, Lexicon &p);

    Lexicon & operator =(const Lexicon &a);
};

inline ostream& operator<<(ostream& s, Lexicon &p)
{
    s << "[LEXICON " << p.lex_name() << "]" ;
    return s;
}

inline Lexicon &Lexicon::operator = (const Lexicon &a)
{
    name = a.name;
    addenda = a.addenda;
    bl_filename = a.bl_filename;
    binlexfp = NULL;
    lts_method = a.lts_method;
    return *this;
}

LISP lex_lookup_word(const EST_String &word,LISP features);
EST_String lex_current_phoneset(void);
LISP lex_select_lex(LISP lexname);
LISP lex_syllabify(LISP phones);
LISP lex_syllabify_phstress(LISP phones);
int in_current_lexicon(const EST_String &word,LISP features);

#endif /* __LEXICON_H__ */
