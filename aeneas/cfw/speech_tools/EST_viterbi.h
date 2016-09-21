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
/*                 Authors:  Alan W Black                                */
/*                 Date   :  July 1996                                   */
/*-----------------------------------------------------------------------*/
/*  A viterbi decoder                                                    */
/*                                                                       */
/*  User provides the candidates, target and combine score function      */
/*  and it searches for the best path through the candidates             */
/*                                                                       */
/*=======================================================================*/

#ifndef __VERTERBI_H__
#define __VERTERBI_H__

#include "EST_cutils.h"
#include "EST_Features.h"
#include "ling_class/EST_Relation.h"

/**  Internal class to \Ref{EST_Viterbi_Decoder} used to a represent 
     a candidate.

     These objects need to be created and set by the user of the Viterbi
     decoder.

     @author Alan W Black (awb@cstr.ed.ac.uk): July 1996
*/
class EST_VTCandidate {
  private:
  public:
    EST_VTCandidate() {score=0.0; next=0; s=0; }
    ~EST_VTCandidate() {if (next != 0) delete next;}
    float score;
    EST_Val name;
    int pos;
    EST_Item *s;
    EST_VTCandidate *next;
};


/**  Internal class to \Ref{EST_Viterbi_Decoder} used to a represent 
     a link in a path the candidates.

     @author Alan W Black (awb@cstr.ed.ac.uk): July 1996
*/
class EST_VTPath {
  private:
  public:
    EST_VTPath() {score=0.0; from=0; next=0; c=0;}
    ~EST_VTPath() {if (next != 0) delete next;}
    double score;   /* cumulative score for path */
    int state;
    EST_Features f;
    EST_VTCandidate *c;
    EST_VTPath *from;
    EST_VTPath *next;
};

/**  Internal class to \Ref{EST_Viterbi_Decoder used to a node in
     the decoder table

    @author Alan W Black (awb@cstr.ed.ac.uk): July 1996
*/
class EST_VTPoint {
  private:
  public:
    EST_VTPoint() {next=0; s=0; paths=0; num_paths=0; cands=0; st_paths=0; num_states=0;}
    ~EST_VTPoint();
    EST_Item *s;
    int num_states;
    int num_paths;
    EST_VTCandidate *cands;
    EST_VTPath *paths;
    EST_VTPath **st_paths;
    EST_VTPoint *next;
};    

typedef EST_VTCandidate *(*uclist_f_t)(EST_Item *s,EST_Features &f);
typedef EST_VTPath *(*unpath_f_t)(EST_VTPath *p,EST_VTCandidate *c,
				  EST_Features &f);

/** A class that offers a generalised Viterbi decoder.

    This class can be used to find the best path through a set
    of candidates based on likelihoods of the candidates and 
    some combination function.  The candidate list and joining
    are not included in the decoder itself but are user defined functions
    that are specified at construction time.  

    Those functions need to return a list of candidates and score
    a join of a path to a candidate and (optionally define a state).

    Although this offers a full Viterbi search it may also be used as
    a generalised beam search.

    See {\tt viterbi_main.cc} for an example of using this.

    @author Alan W Black (awb@cstr.ed.ac.uk): July 1996
*/
class EST_Viterbi_Decoder {
  private:
    int num_states;

    /// very detailed info - for developers
    int debug;

    /// less detailed info than debug - for users
    int trace;

    int beam_width;
    int cand_width;
    int big_is_good;
    uclist_f_t user_clist;
    unpath_f_t user_npath;
    EST_VTPoint *timeline;

    /// pruning parameters
    bool do_pruning;
    float overall_path_pruning_envelope_width;
    float candidate_pruning_envelope_width;

    void add_path(EST_VTPoint *p, EST_VTPath *np);
    void vit_add_path(EST_VTPoint *p, EST_VTPath *np);
    void vit_add_paths(EST_VTPoint *p, EST_VTPath *np);
    EST_VTPath *find_best_end() const;
    const int betterthan(const float a,const float b) const;
    void prune_initialize(EST_VTPoint *p,
			  double &best_score, double &best_candidate_score,
			  double &score_cutoff, double &candidate_cutoff,
			  int &cand_count);
  public:

    /// For holding values to pass to user called functions
    EST_Features f;

    /// Unfortunately using MAX_DOUBLE doesn't do the right thing
    /// (e.g. comparison don't work with MAX_DOUBLE on alphas), so
    /// we declare our own large number.
    const double vit_a_big_number;

    /** Construct a decoder with given candidate function and join function,
        as number of states is given this implies a beam search
    */
    EST_Viterbi_Decoder(uclist_f_t a, unpath_f_t b);
    /** Construct a decoder with given candidate function and join function
        with a state size as specified.
    */
    EST_Viterbi_Decoder(uclist_f_t a, unpath_f_t b, int num_states);
    ///
    ~EST_Viterbi_Decoder();
    ///  Only for use in beam search mode: number of paths to consider
    void set_beam_width(int w) {beam_width = w;}
    ///  Only for use in beam search mode: number of candidates to consider
    void set_cand_width(int w) {cand_width = w;}
    ///  Output some debugging information
    void set_debug(int d) {debug = d;}

    /** Define whether good scores are bigger or smaller.  This allows
        the search to work for likelihoods probabilities, scores or
        whatever
    */
    void set_big_is_good(int flag) { big_is_good = flag; }
    /** Add a new candidate to list if better than others, pruning
        the list if required.
    */
    EST_VTCandidate *add_cand_prune(EST_VTCandidate *newcand,
				    EST_VTCandidate *allcands);

    bool vit_prune_path(double path_score, double score_cutoff);

    /// Build the initial table from a \Ref{EST_Relation}
    void initialise(EST_Relation *r);

    /// set beam widths for pruning
    void set_pruning_parameters(float beam, float ob_beam);

    void turn_on_debug();
    void turn_on_trace();

    /// Do the the actual search
    void search(void);
    /** Extract the result from the table and store it as a feature
        on the related \Ref{EST_Item} in the given \Ref{EST_Relation}
        named as {\tt n}. Return FALSE if no path is found.
    */
    bool result(const EST_String &n);

    /** Extract the end point of the best path found during search. 
	Return FALSE if no path is found.
     */
    bool result( EST_VTPath **bestPathEnd );

    /// Copy named feature from the best path to related stream item
    void copy_feature(const EST_String &n);
};


#endif // __VERTERBI_H__
