/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                         Copyright (c) 1997                            */
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
/*             Author :  Alan W Black                                    */
/*             Date   :  November 1997                                   */
/*-----------------------------------------------------------------------*/
/*                                                                       */
/* Weighted Finite State Transducers                                     */
/*                                                                       */
/*=======================================================================*/
#ifndef __EST_WFST_H__
#define __EST_WFST_H__

#include "EST_simplestats.h"
#include "EST_rw_status.h"
#include "EST_Option.h"
#include "EST_TList.h"
#include "EST_TVector.h"
#include "EST_THash.h"
#include "siod.h"
#define wfst_error_msg(WMESS) (cerr << WMESS << endl,siod_error())

#define WFST_ERROR_STATE -1

class EST_WFST_State;
class EST_WFST;

/** an internal class for \Ref{EST_WFST} for representing transitions 
    in an WFST
 */
class EST_WFST_Transition {
  private:
    float p_weight;
    int p_state;
    int p_in_symbol;
    int p_out_symbol;
  public:
    EST_WFST_Transition();
    EST_WFST_Transition(const EST_WFST_Transition &t)
       {  p_weight=t.p_weight; p_state=t.p_state; 
	  p_in_symbol = t.p_in_symbol; p_out_symbol=t.p_out_symbol; }
    EST_WFST_Transition(float w, int s, int i, int o)
       { p_weight=w; p_state=s; p_in_symbol=i; p_out_symbol=o;}
    ~EST_WFST_Transition() { };
    float weight() const { return p_weight; }
    int state() const { return p_state; }
    int in_symbol() const { return p_in_symbol; }
    int out_symbol() const { return p_out_symbol; }
    void set_weight(float f) { p_weight = f; }
    void set_state(int s) { p_state = s; }
    
};
typedef EST_TList<EST_WFST_Transition *> wfst_translist;

enum wfst_state_type {wfst_final, wfst_nonfinal, wfst_error, wfst_licence};
/** I'd like to use the enums but I need to binary read/write them **/
/** I don't believe that's portable so we need to have ints for these **/
#define WFST_FINAL    0
#define WFST_NONFINAL 1
#define WFST_ERROR    2
#define WFST_LICENCE  3


/** an internal class for \Ref{EST_WFST} used to represent a 
    state in a WFST 
*/
class EST_WFST_State {
  private:
    int p_name;
    enum wfst_state_type p_type;
    int p_tag;   // for marking in traversing
  public:
    wfst_translist transitions;

    EST_WFST_State(int name);
    EST_WFST_State(const EST_WFST_State &state);
    ~EST_WFST_State();

    EST_WFST_Transition *add_transition(float w,
					int end, 
					int in,
					int out);
    int name() const { return p_name; }
    int num_transitions() const { return transitions.length(); }
    enum wfst_state_type type() const { return p_type; }
    void set_type(wfst_state_type t) { p_type = t; }
    void set_tag(int v) { p_tag = v;}
    int tag() const { return p_tag;}
};
typedef EST_TVector<EST_WFST_State *> wfst_state_vector;

typedef EST_TStringHash<int> EST_WFST_MultiStateIndex;
enum wfst_mstate_type {wfst_ms_set, wfst_ms_list};

/** an internal class to \Ref{EST_WFST} used in holding multi-states
    when determinizing and find the intersections of other WFSTs.
 */
class EST_WFST_MultiState : public EST_IList {
  private:
    int p_name;
    float p_weight;
    enum wfst_mstate_type p_type;
  public:
    EST_WFST_MultiState() : EST_IList() 
       { p_name = -1; p_weight = 0.0;  p_type = wfst_ms_set; }
    EST_WFST_MultiState(enum wfst_mstate_type ty) : EST_IList() 
       { p_name = -1; p_weight = 0.0;  p_type = ty; }
    int name() const { return p_name; }
    void set_name(int i) { p_name = i; }
    float weight() const { return p_weight; }
    void set_weight(float w) { p_weight = w; }
    void set_type(enum wfst_mstate_type s) { p_type = s; }
    enum wfst_mstate_type type() const { return p_type; }
    void add(int i);
};

int multistate_index(EST_WFST_MultiStateIndex &i,EST_WFST_MultiState *ms);

/** a call representing a weighted finite-state transducer
 */
class EST_WFST {
  private:
    EST_Discrete p_in_symbols;
    EST_Discrete p_out_symbols;
    int p_start_state;
    int current_tag;
    int p_num_states;
    int p_cumulate;
    wfst_state_vector p_states;

    int operator_and(LISP l);
    int operator_or(LISP l);
    int operator_star(LISP l);
    int operator_plus(LISP l);
    int operator_optional(LISP l);
    int operator_not(LISP l);
    int terminal(LISP l);
    EST_WFST_State *copy_and_map_states(const EST_IVector &state_map,
					const EST_WFST_State *s,
					const EST_WFST &b) const; 
    void extend_alphabets(const EST_WFST &b);
    int deterministiconstartstates(const EST_WFST &a, const EST_WFST &b) const;
    EST_read_status load_transitions_from_lisp(int s, LISP trans);
    void more_states(int new_max);

    int can_reach_final(int state);
    static int traverse_tag;
  public:
    /**@name Constructor and initialisation functions */
    //@{
    /// ?
    EST_WFST();
    /// ?
    EST_WFST(const EST_WFST &wfst) { p_num_states = 0; copy(wfst); }
    ~EST_WFST();
    //@}

    /**@name Reseting functions */
    //@{
    /// Clear with (estimation of number of states required)
    void init(int init_num_states=10);
    /// clear an initialise with given input and out alphabets
    void init(LISP in, LISP out);
    /// Copy from existing wfst
    void copy(const EST_WFST &wfst);
    /// clear removing existing states if any
    void clear();
    //@}

    /**@name General utility functions */
    //@{
    int num_states() const { return p_num_states; }
    int start_state() const { return p_start_state; }
    /// Map input symbol to input alphabet index
    int in_symbol(const EST_String &s) const
        { return p_in_symbols.name(s); }
    /// Map input alphabet index to input symbol
    const EST_String &in_symbol(int i) const
        { return p_in_symbols.name(i); }
    /// Map output symbol to output alphabet index
    int out_symbol(const EST_String &s) const
        { return p_out_symbols.name(s); }
    /// Map output alphabet index to output symbol 
    const EST_String &out_symbol(int i) const
        { return p_out_symbols.name(i); }
    /// LISP for on epsilon symbols
    LISP epsilon_label() const { return rintern("__epsilon__"); }
    /// Internal index for input epsilon
    int in_epsilon() const { return p_in_symbols.name("__epsilon__"); }
    /// Internal index for output epsilon
    int out_epsilon() const { return p_out_symbols.name("__epsilon__"); }
    /// Return internal state information
    const EST_WFST_State *state(int i) const { return p_states(i); }
    /// Return internal state information (non-const)
    EST_WFST_State *state_non_const(int i) { return p_states(i); }
    /// True if state {\tt i} is final
    int final(int i) const 
       { return ((i != WFST_ERROR_STATE) && (state(i)->type() == wfst_final));}
    /// Accessing the input alphabet
    const EST_Discrete &in_symbols() const { return p_in_symbols; }
    /// Accessing the output alphabet
    const EST_Discrete &out_symbols() const { return p_out_symbols; }

    //@}

    /**@name file i/o */
    //@{
    /// ?
    EST_write_status save(const EST_String &filename,
			  const EST_String type = "ascii");
    EST_write_status save_binary(FILE *fd);
    /// ?
    EST_read_status load(const EST_String &filename);

    EST_read_status load_binary(FILE *fd, 
				EST_Option &hinfo, 
				int num_states,
				int swap);
    //@}

    /**@name transduction functions */
    //@{
    /// Find (first) new state given in and out symbols
    int transition(int state,int in, int out) const;
    int transition(int state,int in, int out, float &prob) const;
    /// Find (first) transition given in and out symbols
    EST_WFST_Transition *find_transition(int state,int in, int out) const;
    /// Find (first) new state given in and out strings
    int transition(int state,const EST_String &in,const EST_String &out) const;
    /// Find (first) new state given in/out string
    int transition(int state,const EST_String &inout) const;
    /// Transduce in to out from state
    int transduce(int state,int in,int &out) const;
    /// Transduce in to out (strings) from state
    int transduce(int state,const EST_String &in,EST_String &out) const;
    /// Transduce in to list of transitions
    void transduce(int state,int in,wfst_translist &out) const;
    /// Find all possible transitions for given state/input/output
    void transition_all(int state,int in, int out,
			EST_WFST_MultiState *ms) const;

    //@}

    /**@name Cumulation functions for adding collective probabilities
       for transitions from data */
    //@{
    /// Cumulation condition
    int cumulate() const {return p_cumulate;}
    /// Clear and start cumulation
    void start_cumulate();
    /// Stop cumulation and calculate probabilities on transitions
    void stop_cumulate();
    //@}

    /**@name WFST construction functions from external representations **/
    //@{
    /// Add a new state, returns new name
    int add_state(enum wfst_state_type state_type);
    /// Given a multi-state return type (final, ok, error)
    enum wfst_state_type ms_type(EST_WFST_MultiState *ms) const;

    /// Basic regex constructor
    void build_wfst(int start, int end,LISP regex);
    /// Basic conjunction constructor
    void build_and_transition(int start, int end, LISP conjunctions);
    /// Basic disjunction constructor
    void build_or_transition(int start, int end, LISP disjunctions);

    // from standard REGEX
    void build_from_regex(LISP inalpha, LISP outalpha, LISP regex);
    // Kay/Kaplan/Koskenniemi rule compile
    void kkrule_compile(LISP inalpha, LISP outalpha, LISP fp, 
			LISP rule, LISP sets);
    // Build from regular (or pseudo-CF) grammar
    void build_from_rg(LISP inalpha, LISP outalpha, 
		       LISP distinguished, LISP rewrites,
		       LISP sets, LISP terms,
		       int max_depth);
    // Build simple tree lexicon
    void build_tree_lex(LISP inalpha, LISP outalpha, 
			 LISP wlist);
    //@}

    /**@name Basic WFST operators */
    //@{
    /// Build determinized form of a
    void determinize(const EST_WFST &a);
    /// Build minimized form of a
    void minimize(const EST_WFST &a);
    /// Build complement of a
    void complement(const EST_WFST &a);
    /** Build intersection of all WFSTs in given list.  The new WFST
	recognizes the only the strings that are recognized by all WFSTs
        in the given list */
    void intersection(EST_TList<EST_WFST> &wl);
    /** Build intersection of WFSTs a and b   The new WFST
	recognizes the only the strings that are recognized by both a
        and b list */
    void intersection(const EST_WFST &a, const EST_WFST &b);
    /** Build union of all WFSTs in given list.  The new WFST 
	recognizes the only the strings that are recognized by at least
        one WFSTs in the given list */
    void uunion(EST_TList<EST_WFST> &wl);
    /** Build union of WFSTs a and b.  The new WFST 
	recognizes the only the strings that are recognized by either
        a or b */
    void uunion(const EST_WFST &a, const EST_WFST &b);
    /** Build new WFST by composition of a and b.  Outputs of a are
        fed to b, given a new WFSTs which has a's input language and b's
        output set.  a's output and b's input alphabets must be the same */
    void compose(const EST_WFST &a,const EST_WFST &b);
    /** Build WFST that accepts only strings in a that aren't also accepted
        by strings in b. */
    void difference(const EST_WFST &a,const EST_WFST &b);
    /** Build WFST that accepts a language that consists of any string in
        a followed by any string in b **/
    void concat(const EST_WFST &a,const EST_WFST &b);
    //@}

    /**@name construction support functions */
    //@{
    /// True if WFST is deterministic
    int deterministic() const;
    /// Transduce a multi-state given n and out
    EST_WFST_MultiState *apply_multistate(const EST_WFST &wfst,
					  EST_WFST_MultiState *ms,
					  int in, int out) const;
    /// Extend multi-state with epsilon reachable states
    void add_epsilon_reachable(EST_WFST_MultiState *ms) const;
    /// Remove error states from the WFST.
    void remove_error_states(const EST_WFST &a);
    
    EST_String summary() const;
    /// ?
    EST_WFST & operator = (const EST_WFST &a) { copy(a); return *this; }
};
typedef EST_TList<EST_WFST> wfst_list;

// The basic operations on WFST
void minimize(EST_WFST &a,EST_WFST &b);
void determinize(EST_WFST &a,EST_WFST &b);
void concat(EST_WFST &a,EST_WFST &b,EST_WFST &c);
void compose(EST_WFST &a,EST_WFST &b,EST_WFST &c);
void intersect(wfst_list &wl,EST_WFST &wfst);
void complement(EST_WFST &a,EST_WFST &b);
void difference(EST_WFST &a,EST_WFST &b,EST_WFST &c);
void concatenate(EST_WFST &a,EST_WFST &b,EST_WFST &c);

// Compile a set of KK rules
void kkcompile(LISP ruleset, EST_WFST &all_wfst);
// Compile a set of LTS rules
void ltscompile(LISP lts_rules, EST_WFST &all_wfst);
// Compile a regular grammar 
void rgcompile(LISP rg, EST_WFST &all_wfst);
// Compile a tree lexicon
void tlcompile(LISP rg, EST_WFST &all_wfst);

// Transduction and recognition functions
int transduce(const EST_WFST &wfst,const EST_StrList &in,EST_StrList &out);
int transduce(const EST_WFST &wfst,const EST_IList &in,EST_IList &out);
int recognize(const EST_WFST &wfst,const EST_StrList &in,int quiet);
int recognize(const EST_WFST &wfst,const EST_IList &in, 
	      const EST_IList &out,int quite);
int recognize_for_perplexity(const EST_WFST &wfst,
			     const EST_StrList &in,
			     int quiet,
			     float &count,
			     float &sumlogp);
int recognize_for_perplexity(const EST_WFST &wfst,
			     const EST_IList &in, 
			     const EST_IList &out, 
			     int quiet,
			     float &count,
			     float &sumlogp);

VAL_REGISTER_CLASS_DCLS(wfst,EST_WFST)

#endif
