/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                      Copyright (c) 1996,1997                          */
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
/*                     Date   :  May 1996                                */
/*-----------------------------------------------------------------------*/
/*                                                                       */
/* Public declarations for Wagon (CART builder)                          */
/*                                                                       */
/*=======================================================================*/
#ifndef __WAGON_H__
#define __WAGON_H__

#include "EST_String.h"
#include "EST_Val.h"
#include "EST_TVector.h"
#include "EST_TList.h"
#include "EST_simplestats.h"  /* For EST_SuffStats class */
#include "EST_Track.h"
#include "siod.h"
#define wagon_error(WMESS) (cerr << WMESS << endl,exit(-1))

// I get floating point exceptions of Alphas when I do any comparisons
// with HUGE_VAL or FLT_MAX so I'll make my own
#define WGN_HUGE_VAL 1.0e20

class WVector : public EST_FVector
{
  public:
    WVector(int n) : EST_FVector(n) {}
    int get_int_val(int n) const { return (int)a_no_check(n); }
    float get_flt_val(int n) const { return a_no_check(n); }
    void set_int_val(int n,int i) { a_check(n) = (int)i; }
    void set_flt_val(int n,float f) { a_check(n) = f; }
};

typedef EST_TList<WVector *> WVectorList;
typedef EST_TVector<WVector *> WVectorVector;

/* Different types of feature */
enum wn_dtype {/* for predictees and predictors */
               wndt_binary, wndt_float, wndt_class, 
               /* for predictees only */
               wndt_cluster, wndt_vector, wndt_matrix, wndt_trajectory,
               wndt_ols, 
               /* for ignored features */
               wndt_ignore};

class WDataSet : public WVectorList {
  private:
    int dlength;
    EST_IVector p_type;
    EST_IVector p_ignore;
    EST_StrVector p_name;
  public:
    void load_description(const EST_String& descfname,LISP ignores);
    void ignore_non_numbers();

    int ftype(const int &i) const {return p_type(i);}
    int ignore(int i) const {return p_ignore(i); }
    void set_ignore(int i,int value) { p_ignore[i] = value; }
    const EST_String &feat_name(const int &i) const {return p_name(i);}
    int samples(void) const {return length();}
    int width(void) const {return dlength;}
};    
enum wn_oper {wnop_equal, wnop_binary, wnop_greaterthan, 
		  wnop_lessthan, wnop_is, wnop_in, wnop_matches};

class WQuestion {
  private:
    int feature_pos;
    wn_oper op;
    int yes;
    int no;
    EST_Val operand1;
    EST_IList operandl;
    float score;
  public:
    WQuestion() {;}
    WQuestion(const WQuestion &s) 
       { feature_pos=s.feature_pos;
         op=s.op; yes=s.yes; no=s.no; operand1=s.operand1;
	 operandl = s.operandl; score=s.score;}
    ~WQuestion() {;}
    WQuestion(int fp, wn_oper o,EST_Val a)
       { feature_pos=fp; op=o; operand1=a; }
    void set_fp(const int &fp) {feature_pos=fp;}
    void set_oper(const wn_oper &o) {op=o;}
    void set_operand1(const EST_Val &a) {operand1 = a;}
    void set_yes(const int &y) {yes=y;}
    void set_no(const int &n) {no=n;}
    int get_yes(void) const {return yes;}
    int get_no(void) const {return no;}
    const int get_fp(void) const {return feature_pos;}
    const wn_oper get_op(void) const {return op;}
    const EST_Val get_operand1(void) const {return operand1;}
    const EST_IList &get_operandl(void) const {return operandl;}
    const float get_score(void) const {return score;}
    void set_score(const float &f) {score=f;}
    const int ask(const WVector &w) const;
    friend ostream& operator<<(ostream& s, const WQuestion &q);
};

enum wnim_type {wnim_unset, wnim_float, wnim_class, 
                wnim_cluster, wnim_vector, wnim_matrix, wnim_ols,
                wnim_trajectory};

//  Impurity measure for cumulating impurities from set of data
class WImpurity {
  private:
    wnim_type t;
    EST_SuffStats a;
    EST_DiscreteProbDistribution p;

    float cluster_impurity();
    float cluster_member_mean(int i);
    float vector_impurity();
    float trajectory_impurity();
    float ols_impurity();
  public:
    EST_IList members;            // Maybe there should be a cluster class
    EST_FList member_counts;      // AUP: Implement counts for vectors
    EST_SuffStats **trajectory;
    const WVectorVector *data;          // Needed for ols
    float score;
    int l,width;

    WImpurity() { t=wnim_unset; a.reset(); trajectory=0; l=0; width=0; data=0;}
    ~WImpurity();
    WImpurity(const WVectorVector &ds);
    void copy(const WImpurity &s) 
    {
        int i,j; 
        t=s.t; a=s.a; p=s.p; members=s.members; member_counts = s.member_counts; l=s.l; width=s.width;
        score = s.score;
        data = s.data;
        if (s.trajectory)
        {
            trajectory = new EST_SuffStats *[l];
            for (i=0; i<l; i++)
            {
                trajectory[i] = new EST_SuffStats[width];
                for (j=0; j<width; j++)
                    trajectory[i][j] = s.trajectory[i][j];
            }
        }
    }
    WImpurity &operator = (const WImpurity &a) { copy(a); return *this; }

    float measure(void);
    double samples(void);
    wnim_type type(void) const { return t;}
    void cumulate(const float pv,double count=1.0);
    EST_Val value(void);
    EST_DiscreteProbDistribution &pd() { return p; }
    float cluster_distance(int i); // distance i from centre in sds
    int in_cluster(int i);       // distance i from centre < most remote member
    float cluster_ranking(int i);  // position in closeness to centre
    friend ostream& operator<<(ostream &s, WImpurity &imp);
};

class WDlist {
  private:
    float p_score;
    WQuestion p_question;
    EST_String p_token;
    int p_freq;
    int p_samples;
    WDlist *next;
  public:
    WDlist() { next=0; }
    ~WDlist() { if (next != 0) delete next; }
    void set_score(float s) { p_score = s; }
    void set_question(const WQuestion &q) { p_question = q; }
    void set_best(const EST_String &t,int freq, int samples)
	{ p_token = t; p_freq = freq; p_samples = samples;}
    float score() const {return p_score;}
    const EST_String &token(void) const {return p_token;}
    const WQuestion &question() const {return p_question;}
    EST_Val predict(const WVector &w);
    friend WDlist *add_to_dlist(WDlist *l,WDlist *a);
    friend ostream &operator<<(ostream &s, WDlist &d);
};

class WNode {
  private:
    WVectorVector data;
    WQuestion question;
    WImpurity impurity;
    WNode *left;
    WNode *right;
    void print_out(ostream &s, int margin);
    int leaf(void) const { return ((left == 0) || (right == 0)); }
    int pure(void);
  public:
    WNode() { left = right = 0; }
    ~WNode() { if (left != 0) {delete left; left=0;}
	       if (right != 0) {delete right; right=0;} }
    WVectorVector &get_data(void) { return data; }
    void set_subnodes(WNode *l,WNode *r) { left=l; right=r; }
    void set_impurity(const WImpurity &imp) {impurity=imp;}
    void set_question(const WQuestion &q) {question=q;}
    void prune(void);
    void held_out_prune(void);
    WImpurity &get_impurity(void) {return impurity;}
    WQuestion &get_question(void) {return question;}
    EST_Val predict(const WVector &w);
    WNode *predict_node(const WVector &d);
    int samples(void) const { return data.n(); }
    friend ostream& operator<<(ostream &s, WNode &n);
};

extern Discretes wgn_discretes;
extern WDataSet wgn_dataset;
extern WDataSet wgn_test_dataset;
extern EST_FMatrix wgn_DistMatrix;
extern EST_Track wgn_VertexTrack;
extern EST_Track wgn_UnitTrack;
extern EST_Track wgn_VertexFeats;

void wgn_load_datadescription(EST_String fname,LISP ignores);
void wgn_load_dataset(WDataSet &ds,EST_String fname);
WNode *wgn_build_tree(float &score);
WNode *wgn_build_dlist(float &score,ostream *output);
WNode *wagon_stepwise(float limit);
float wgn_score_question(WQuestion &q, WVectorVector &ds);
void wgn_find_split(WQuestion &q,WVectorVector &ds,
		WVectorVector &y,WVectorVector &n);
float summary_results(WNode &tree,ostream *output);

extern int wgn_min_cluster_size;
extern int wgn_held_out;
extern int wgn_prune;
extern int wgn_quiet;
extern int wgn_verbose;
extern int wgn_predictee;
extern int wgn_count_field;
extern EST_String wgn_count_field_name;
extern EST_String wgn_predictee_name;
extern float wgn_float_range_split;
extern float wgn_balance;
extern EST_String wgn_opt_param;
extern EST_String wgn_vertex_output;

#define wgn_ques_feature(X) (get_c_string(car(X)))
#define wgn_ques_oper_str(X) (get_c_string(car(cdr(X))))
#define wgn_ques_operand(X) (car(cdr(cdr(X))))

int wagon_ask_question(LISP question, LISP value);

int stepwise_ols(const EST_FMatrix &X,
		 const EST_FMatrix &Y,
		 const EST_StrList &feat_names,
		 float limit,
		 EST_FMatrix &coeffs,
		 const EST_FMatrix &Xtest,
		 const EST_FMatrix &Ytest,
                 EST_IVector &included,
                 float &best_score);
int robust_ols(const EST_FMatrix &X,
	       const EST_FMatrix &Y, 
	       EST_IVector &included,
	       EST_FMatrix &coeffs);
int ols_apply(const EST_FMatrix &samples,
	      const EST_FMatrix &coeffs,
	      EST_FMatrix &res);
int ols_test(const EST_FMatrix &real,
	     const EST_FMatrix &predicted,
	     float &correlation,
	     float &rmse);

#endif /* __WAGON_H__ */
