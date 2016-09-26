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
/*             Date   :  June 1997                                       */
/*-----------------------------------------------------------------------*/
/*                                                                       */
/* A SCFG chart parser, general functions                                */
/*                                                                       */
/*=======================================================================*/
#ifndef __EST_SCFG_CHART_H__
#define __EST_SCFG_CHART_H__

#include "EST_String.h"
#include "EST_simplestats.h"
#include "EST_string_aux.h"
#include "EST_SCFG.h"
#include "ling_class/EST_Relation.h"

class EST_SCFG_Chart_Edge;

/** An internal class for \Ref{EST_SCFG_Chart} for representing edges
    in the chart during parsing with SCFGs.

    A standard Earley type chart edge, with representations for two
    daughters and a position or what has been recognised.  A probability
    is also included.
*/
class EST_SCFG_Chart_Edge {
  private:
    int p_d1;
    int p_d2;
    int p_pos;
    double p_prob;
  public:
    /**@name Constructor and initialisation functions */
    //@{
    EST_SCFG_Chart_Edge();
    EST_SCFG_Chart_Edge(double prob, int d1, int d2, int pos);
    ~EST_SCFG_Chart_Edge();
    //@}

    /// Postion, 0 1 or 2, where 0 is empty, 1 is incomplete 2 is complete.
    int pos(void) { return p_pos; }
    /// Edge probability
    double prob(void) { return p_prob; }
    /// (Non)terminal of daughter 1
    int d1() { return p_d1; }
    /// (Non)terminal of daughter 2
    int d2() { return p_d2; }

};

/** A class for parsing with a probabilistic grammars.

    The chart (sort of closer to CKY table) consists of indexes of
    edges indexed by vertex number of mother non-terminal.

    The initial values (well-formed substring table) are taken from
    an \Ref{EST_Stream} with a given feature.  The grammar may be
    specified as LISP rules or as an already constructed \Ref{EST_SCFG}.

    This produces a single best parse.  It treats the grammar as
    strictly context free in that the probability of a nonterminal
    over vertex n to m, is the sum of all the possible analyses
    of that sub-tree.  Only the best analysis is kept for the
    resulting parse tree.

    @author Alan W Black (awb@cstr.ed.ac.uk): October 1997
*/
class EST_SCFG_Chart {
  private:
    /// pointer to grammar
    EST_SCFG *grammar;
    /// TRUE is grammar was created internally, FALSE is can't be freed
    int grammar_local;
    /// Number of vertices (number of words + 1)
    int n_vertices;
    /// Index of edges by vertex start x vertex end x nonterminal
    EST_SCFG_Chart_Edge ****edges;
    /// Index of basic symbols indexed by (start) vertex.
    EST_SCFG_Chart_Edge **wfst;
    /// An empty edge, denotes 0 probability edge.
    EST_SCFG_Chart_Edge *emptyedge;

    // Find the best analysis of nonterminal {\tt p} from {\tt start} 
    // to {\tt end}.  Used after parsing
    double find_best_tree(int start,int end,int p)
       { EST_SCFG_Chart_Edge *r;
	 if ((r=edges[start][end][p]) != 0) return r->prob();
	 else return find_best_tree_cal(start,end,p); }
    // Calculate best tree/probability
    double find_best_tree_cal(int start,int end,int p);
    void setup_edge_table();
    void delete_edge_table();
    LISP print_edge(int start, int end, int name, EST_SCFG_Chart_Edge *e);
    // Extract edge from chart and add it to stream
    void extract_edge(int start, int end, int p,
		      EST_SCFG_Chart_Edge *e,
		      EST_Item *s,
		      EST_Item **word);
    // Build parse from distinguished symbol alone
    void extract_forced_parse(int start, int end, EST_Item *s, EST_Item *w);
  public:
    /**@name Constructor and initialisation functions */
    //@{
    EST_SCFG_Chart();
    ~EST_SCFG_Chart();
    //@}

    /**@name Grammar and parse string initialisation functions */
    //@{
    /// Initialize from LISP rules set
    void set_grammar_rules(LISP r);
    /// Initialize from existing \Ref{EST_SCFG} grammar
    void set_grammar_rules(EST_SCFG &grammar);
    /** Initialize for parsing from relation using {\tt name} feature
        setting up the "Well Formed Substring Table" */
    void setup_wfst(EST_Relation *s,const EST_String &name="name");
    /** Initialize for parsing from s to e using {\tt name} feature
        setting up the "Well Formed Substring Table" */
    void setup_wfst(EST_Item *s, EST_Item *e,const EST_String &name="name");
    //@}

    /**@name parsing functions */
    //@{
    /// Parses the loaded WFST with the loaded grammar.
    void parse();
    /// Return the parse in full LISP form.
    LISP find_parse();
    /// Extract parse tree and add it to syn linking leafs to word
    void extract_parse(EST_Relation *syn,EST_Relation *word,int force=0);
    /// Extract parse tree and add it to syn linking leafs to items s to e
    void extract_parse(EST_Relation *syn,EST_Item *s, 
		       EST_Item *e,int force=0);
    //@}
};

/** Build a relation from a LISP list of items.
*/
void EST_SCFG_chart_load_relation(EST_Relation *s,LISP sent);

/** Parse a given string using the given grammar.
*/
LISP scfg_parse(LISP string,LISP grammar);
/** Parse the given string using the given \Ref{EST_SCFG}.
*/
LISP scfg_parse(LISP string,EST_SCFG &grammar);
/** Parse named features in (list) relation Word into (tree)
 ** relation Syntax
 */
void scfg_parse(class EST_Relation *Word, const EST_String &name, 
		class EST_Relation *Syntax, EST_SCFG &grammar);

#endif
