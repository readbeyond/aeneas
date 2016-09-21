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
/*             Date   :  October 1997                                    */
/*-----------------------------------------------------------------------*/
/*                                                                       */
/* Stochastic context free grammars                                      */
/*                                                                       */
/*=======================================================================*/
#ifndef __EST_SCFG_H__
#define __EST_SCFG_H__

#include "EST_simplestats.h"
#include "EST_rw_status.h"
#include "EST_TList.h"
#include "siod.h"

/** This class represents a bracketed string used in training of SCFGs.

    An object in this class builds an index of valid bracketing of
    the string, thus offering both a tree like access and direct
    access to the leafs of the tree.  The definition of ``valid
    bracketing'' is any substring \[ W_{i,j} \] that doesn't cross any
    brackets.
*/
class EST_bracketed_string {
  private:
    int p_length;
    LISP *symbols;
    LISP bs;
    int **valid_spans;   // triangular matrix
    int find_num_nodes(LISP string);
    int set_leaf_indices(LISP string,int i,LISP *symbols);
    int num_leafs(LISP l) const;
    void find_valid(int i,LISP t) const;
    void init();
  public:
    ///
    EST_bracketed_string();
    ///
    EST_bracketed_string(LISP string);
    ///
    ~EST_bracketed_string();

    ///
    void set_bracketed_string(LISP string);
    ///
    int length() const {return p_length;}
    ///
    LISP string() const { return bs; }
    /// The nth symbol in the string.
    const EST_String symbol_at(int i) const 
       { return EST_String(get_c_string(car(symbols[i]))); }
    /// If a bracketing from i to k is valid in string
    int valid(int i,int k) const { return valid_spans[i][k]; }

    ///
    int operator !=(const EST_bracketed_string &a) const 
       { return (!(this == &a)); }
    int operator ==(const EST_bracketed_string &a) const 
       { return ((this == &a)); }
    ///
    friend ostream& operator << (ostream &s, const EST_bracketed_string &a)
       { (void)a; s << "[a bracketed string]" << endl; return s; }

};

typedef EST_TVector<EST_bracketed_string> EST_Bcorpus;

// Only support Chomsky Normal Form at present
enum est_scfg_rtype {est_scfg_unset, est_scfg_binary_rule,
		     est_scfg_unary_rule};

/** A stochastic context free grammar rule.  

    At present only two types of rule are supported: 
    {\tt est\_scfg\_binary\_rule} and {\tt est\_scfg\_unary\_rule}.
    This is sufficient for the representation of grammars in 
    Chomsky Normal Form.  Each rule also has a probability associated
    with it.  Terminals and noterminals are represented as ints using
    the \Ref{EST_Discrete}s in \Ref{EST_SCFG} to reference the actual
    alphabets.

    Although this class includes a ``probability'' nothing in the rule
    itself enforces it to be a true probability.  It is responsibility
    of the classes that use this rule to enforce that condition if
    desired.

    @author Alan W Black (awb@cstr.ed.ac.uk): October 1997
*/
class EST_SCFG_Rule {
  private:
    int p_mother;
    int p_daughter1;
    int p_daughter2;
    est_scfg_rtype p_type;
    double p_prob;
  public:
    ///
    EST_SCFG_Rule() {p_type=est_scfg_unset; p_prob=0;}
    ///
    EST_SCFG_Rule(const EST_SCFG_Rule &r) 
      {p_mother = r.p_mother; p_daughter1 = r.p_daughter1;
       p_daughter2 = r.p_daughter2; p_type=r.p_type; p_prob = r.p_prob;}
    /// Create a unary rule.
    EST_SCFG_Rule(double prob,int p,int m);
    /// Create a binary rule.
    EST_SCFG_Rule(double prob,int p, int q, int r);
    /// The rule's probability
    double prob() const {return p_prob;}
    /// set the probability
    void set_prob(double p) { p_prob=p;}
    /// rule type
    est_scfg_rtype type() const { return p_type; }
    ///
    int mother() const {return p_mother;}
    /** In a unary rule this is a terminal, in a binary rule it 
        is a nonterminal
    */
    int daughter1() const {return p_daughter1;}
    ///
    int daughter2() const {return p_daughter2;}
    ///
    void set_rule(double prob,int p, int m);
    ///
    void set_rule(double prob,int p, int q, int r);
};

typedef EST_TList<EST_SCFG_Rule> SCFGRuleList;

/** A class representing a stochastic context free grammar (SCFG).

    This class includes the representation of the grammar itself and
    methods for training and testing it against some corpus.

    At presnet of grammars in Chomsky Normal Form are supported.  That
    is rules may be binary or unary.  If binary the mother an two
    daughters are nonterminals, if unary the mother must be nonterminal
    and daughter a terminal symbol.  

    The terminals and nonterminals symbol sets are derived automatically
    from the LISP representation of the rules at initialization time
    and are represented as \Ref{EST_Discrete}s.  The distinguished
    symbol is assumed to be the first mother of the first rule in
    the given grammar.

*/
class EST_SCFG {
  private:
    EST_Discrete nonterminals;
    EST_Discrete terminals;
    int p_distinguished_symbol;
    // Index of probabilities for binary rules in grammar
    double ***p_prob_B;
    // Index of probabilities for unary rules in grammar
    double **p_prob_U;
    // Build rule probability caches
    void rule_prob_cache();
    // Delete rule probability caches
    void delete_rule_prob_cache();
  public:
    /**@name Constructor and initialisation functions */
    //@{
    EST_SCFG();
    /// Initialize from a set of rules
    EST_SCFG(LISP rules);
    ~EST_SCFG();
    //@}

    /**@name utility functions */
    //@{
    /// Set (or reset) rules from external source after construction
    void set_rules(LISP rules);
    /// Return rules as LISP list.
    LISP get_rules();
    /// The rules themselves
    SCFGRuleList rules;
    int distinguished_symbol() const { return p_distinguished_symbol; }
    /** Find the terminals and nonterminals in the given grammar, adding
        them to the appropriate given string lists.
    */
    void find_terms_nonterms(EST_StrList &nt, EST_StrList &t,LISP rules);
    /// Convert nonterminal index to string form
    EST_String nonterminal(int p) const { return nonterminals.name(p); }
    /// Convert terminal index to string form
    EST_String terminal(int m) const { return terminals.name(m); }
    /// Convert nonterminal string to index
    int nonterminal(const EST_String &p) const { return nonterminals.name(p); }
    /// Convert terminal string to index
    int terminal(const EST_String &m) const { return terminals.name(m); }
    /// Number of nonterminals
    int num_nonterminals() const { return nonterminals.length(); }
    /// Number of terminals
    int num_terminals() const { return terminals.length(); }
    /// The rule probability of given binary rule
    double prob_B(int p, int q, int r) const { return p_prob_B[p][q][r]; }
    /// The rule probability of given unary rule
    double prob_U(int p, int m) const { return p_prob_U[p][m]; }
    /// (re-)set rule probability caches
    void set_rule_prob_cache();
    //@}

    /**@name file i/o functions */
    //@{
    /// Load grammar from named file
    EST_read_status load(const EST_String &filename);
    /// Save current grammar to named file
    EST_write_status save(const EST_String &filename);
    //@}
};

/** A class used to train (and test) SCFGs is an extension of 
    \Ref{EST_SCFG}.

    This offers an implementation of Pereira and Schabes ``Inside-Outside
    reestimation from partially bracket corpora.''  ACL 1992.

    A SCFG maybe trained from a corpus (optionally) containing brackets
    over a series of passes reestimating the grammar probabilities
    after each pass.   This basically extends the \Ref{EST_SCFG} class
    adding support for a bracket corpus and various indexes for efficient
    use of the grammar.
*/
class EST_SCFG_traintest : public EST_SCFG {
  private:
    /// Index for inside probabilities
    double ***inside;
    /// Index for outside probabilities
    double ***outside;
    EST_Bcorpus corpus;
    /// Partial (numerator) for reestimation
    EST_DVector n;
    /// Partial (denominator) for reestimation
    EST_DVector d;

    /// Calculate inside probability.
    double f_I_cal(int c, int p, int i, int k);
    /// Lookup or calculate inside probability.
    double f_I(int c, int p, int i, int k)
    { double r; 
      if ((r=inside[p][i][k]) != -1) return r;
      else return f_I_cal(c,p,i,k); }
    /// Calculate outside probability.
    double f_O_cal(int c, int p, int i, int k);
    /// Lookup or calculate outside probability.
    double f_O(int c, int p, int i, int k)
    { double r; 
      if ((r=outside[p][i][k]) != -1) return r;
      else return f_O_cal(c,p,i,k); }
    /// Find probability of parse of corpus sentence {\tt c}
    double f_P(int c);
    /** Find probability of parse of corpus sentence {\tt c} for
        nonterminal {\tt p}
    */
    double f_P(int c,int p);
    /// Re-estimate probability of binary rule using inside-outside algorithm
    void reestimate_rule_prob_B(int c, int ri, int p, int q, int r);
    /// Re-estimate probability of unary rule using inside-outside algorithm
    void reestimate_rule_prob_U(int c, int ri, int p, int m);
    /// Do grammar re-estimation
    void reestimate_grammar_probs(int passes,
				  int startpass,
				  int checkpoint,
				  int spread,
				  const EST_String &outfile);
    ///
    double cross_entropy();
    /// Initialize the cache for inside/outside values for sentence {\tt c}
    void init_io_cache(int c,int nt);
    /// Clear the cache for inside/outside values for sentence {\tt c}
    void clear_io_cache(int c);
  public:
    EST_SCFG_traintest();
    ~EST_SCFG_traintest();

    /** Test the current grammar against the current corpus print summary.

        Cross entropy measure only is given.
    */
    void test_corpus();
    /** Test the current grammar against the current corpus.

        Summary includes percentage of cross bracketing accuracy 
        and percentage of fully correct parses.
    */
    void test_crossbrackets();

    /** Load a corpus from the given file.

        Each sentence in the corpus should be contained in parentheses.
        Additional parenthesis may be used to denote phrasing within
        a sentence.  The corpus is read using the LISP reader so LISP
        conventions shold apply, notable single quotes should appear
        within double quotes.
    */
    void load_corpus(const EST_String &filename);

    /** Train a grammar using the loaded corpus.

        @param passes     the number of training passes desired.
	@param startpass  from which pass to start from
	@param checkpoint save the grammar every n passes
	@param spread     Percentage of corpus to use on each pass,
             this cycles through the corpus on each pass.
    */
    void train_inout(int passes,
		     int startpass,
		     int checkpoint,
		     int spread, 
		     const EST_String &outfile);
};

/** From a full parse, extract the string with bracketing only.
*/
LISP scfg_bracketing_only(LISP parse);
/** Cummulate cross bracketing information between ref and test.
 */
void count_bracket_crossing(const EST_bracketed_string &ref,
			    const EST_bracketed_string &test,
			    EST_SuffStats &vs);

#endif
