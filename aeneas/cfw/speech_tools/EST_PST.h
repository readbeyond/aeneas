/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                         Copyright (c) 1996                            */
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
/*                     Date   :  July 1996                               */
/*-----------------------------------------------------------------------*/
/*                                                                       */
/*             A general class for PredictionSuffixTrees                 */
/*                                                                       */
/*=======================================================================*/

#ifndef __PredictionSuffixTree_H__
#define __PredictionSuffixTree_H__

#include "EST_simplestats.h"
#include "EST_types.h"
#include "EST_Features.h"

class EST_PredictionSuffixTree_tree_node {
private:
    
protected:
    
    int p_level;
    int state;
    EST_DiscreteProbDistribution pd;
    EST_String path;  /* context */
    void delete_node(void *n) { if (n != 0) delete (EST_PredictionSuffixTree_tree_node *)n;}

public:

//    EST_StringTrie nodes;
//    EST_TKVL <EST_String, EST_PredictionSuffixTree_tree_node *> nodes;
    EST_Features nodes;
    EST_PredictionSuffixTree_tree_node() {p_level=0;}
    ~EST_PredictionSuffixTree_tree_node();
    void clear(void);
    const EST_String &get_path(void) const {return path;}
    void set_path(const EST_String &s) {path=s;}
    void set_level(int l) {p_level=l;}
    void set_state(int s) {state=s;}
    int get_state(void) const {return state;}
    int get_level(void) const {return p_level;}
    void cumulate(const EST_String &s,double count=1) {pd.cumulate(s,count);}
    void cumulate(const int i,double count=1) {pd.cumulate(i,count);}
    const EST_String &most_probable(double *p) const;
    const EST_DiscreteProbDistribution &prob_dist() const {return pd;}
    void print_freqs(ostream &os);
    void print_probs(ostream &os);
};

VAL_REGISTER_CLASS_DCLS(pstnode,EST_PredictionSuffixTree_tree_node)

class EST_PredictionSuffixTree {
    
private:
    
    enum EST_filetype {PredictionSuffixTree_ascii, PredictionSuffixTree_binary};

protected:

    int p_order;
    int num_states;
    EST_PredictionSuffixTree_tree_node *nodes;
    EST_DiscreteProbDistribution *pd; // distribution of predictees
    const EST_String &ppredict(EST_PredictionSuffixTree_tree_node *node,
			       const EST_StrVector &words,
			       double *prob, int *state,
			       const int index=0) const;

    void p_accumulate(EST_PredictionSuffixTree_tree_node *node,
		      const EST_StrVector &words,
		      double count, 
		      const int index=0);

    const EST_DiscreteProbDistribution &p_prob_dist(
	EST_PredictionSuffixTree_tree_node *node, 
	const EST_StrVector &words,
	const int index=0) const;
public:
    EST_PredictionSuffixTree();
    EST_PredictionSuffixTree(const int order) {init(order);}
    EST_PredictionSuffixTree(const EST_String filename);
    EST_PredictionSuffixTree(const EST_TList<EST_String> &vocab,int order=2);
    ~EST_PredictionSuffixTree(); 
    void clear(void);
    void init(const int order);
    double samples() const { return pd->samples(); }
    int states() const { return num_states; }
    int order(void) const {return p_order;}
    void accumulate(const EST_StrVector &words,const double count=1,const int index=0);

    int load(const EST_String filename);
    int save(const EST_String filename,const EST_PredictionSuffixTree::EST_filetype type=PredictionSuffixTree_ascii);

    // build EST_PredictionSuffixTree from train data
    void build(const EST_String filename,
	       const EST_String prev,
	       const EST_String prev_prev,
	       const EST_String last);

    void build(const EST_StrList &input); // to go

    void test(const EST_String filename);   // test EST_PredictionSuffixTree against test data
    void print_freqs(ostream &os);
    void print_probs(ostream &os);

    const EST_String &predict(const EST_StrVector &words) const;
    const EST_String &predict(const EST_StrVector &words,double *prob) const;
    const EST_String &predict(const EST_StrVector &words,double *prob,int *state) const;
    const EST_DiscreteProbDistribution &prob_dist(const EST_StrVector &words)
	const 
    {return p_prob_dist(nodes,words);}
    /* Reverse probability, given X what is prob of EST_PredictionSuffixTree Y */
    double rev_prob(const EST_StrVector &words) const;
    double rev_prob(const EST_StrVector &words,
		    const EST_DiscreteProbDistribution &pd) const;
    /* print frequency or probability models */
    /* build model from file */
    /* predict and measure success */

};

#endif // __PredictionSuffixTree_H__
