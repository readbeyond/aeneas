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
/*                     Author :  Simon King & Alan W Black               */
/*                     Date   :  February 1997                           */
/*-----------------------------------------------------------------------*/
/*                                                                       */
/* A general class for ngrams (bi-gram, tri-gram etc)                    */
/*                                                                       */
/*=======================================================================*/
#ifndef __EST_NGRAMMAR_H__
#define __EST_NGRAMMAR_H__

#include <cstdarg>
#include <cstdlib>

using namespace std;

#include "EST_String.h"
#include "EST_Val.h"
#include "EST_rw_status.h"
#include "EST_types.h"
#include "EST_FMatrix.h"
#include "EST_TList.h"
#include "EST_StringTrie.h"
#include "EST_simplestats.h"
#include "EST_PST.h"
#include "EST_string_aux.h"
#include "EST_math.h"

// HTK style
#define SENTENCE_START_MARKER "!ENTER"
#define SENTENCE_END_MARKER "!EXIT"
#define OOV_MARKER "!OOV"

#define EST_NGRAMBIN_MAGIC 1315402337

// for compressed save/load
#define GZIP_FILENAME_EXTENSION "gz"
#define COMPRESS_FILENAME_EXTENSION "Z"

// Ultimate floor 
#define TINY_FREQ 1.0e-10

// ngram state - represents the N-1 word history and contains
// the pdf of the next word

class EST_NgrammarState {

private:

protected:
    EST_DiscreteProbDistribution p_pdf;
    int p_id; // a 'name'

public:
    EST_NgrammarState() : 
      
      p_pdf() 

      { 
	init(); 
      };
    EST_NgrammarState(int id,EST_Discrete *d){clear();init(id,d);};
    EST_NgrammarState(int id,const EST_DiscreteProbDistribution &pdf)
              {clear();init(id,pdf);};
    EST_NgrammarState(const EST_NgrammarState &s);
    EST_NgrammarState(const EST_NgrammarState *const s);
    ~EST_NgrammarState();

    EST_IVector path;  // how we got here

    // initialise
    void clear();
    void init();
    void init(int id, EST_Discrete *d);
    void init(int id, const EST_DiscreteProbDistribution &pdf);

    // build
    void cumulate(const int index, const double count=1)
                  {p_pdf.cumulate(index,count);};
    void cumulate(const EST_String &word, const double count=1)
                  {p_pdf.cumulate(word,count);};
    
    // access
    int id() const {return p_id; };
    const EST_DiscreteProbDistribution &pdf_const() const {return p_pdf; };
    EST_DiscreteProbDistribution &pdf() {return p_pdf; };
    double probability(const EST_String &w) const
      {return p_pdf.probability(w);}
    double probability(int w) const {return p_pdf.probability(w);}
    double frequency(const EST_String &w) const
      {return p_pdf.frequency(w);}
    double frequency(int w) const {return p_pdf.frequency(w);}
    const EST_String &most_probable(double *prob = NULL) const
      {return p_pdf.most_probable(prob);}
    
friend ostream&  operator<<(ostream& s, const EST_NgrammarState &a);
    
};

class EST_BackoffNgrammarState {

private:

protected:
  int p_level; // = 0 for root node
  double backoff_weight;
  EST_DiscreteProbDistribution p_pdf;
  EST_StringTrie children;
  
  EST_BackoffNgrammarState* add_child(const EST_Discrete *d,
				      const EST_StrVector &words);
  EST_BackoffNgrammarState* add_child(const EST_Discrete *d,
				      const EST_IVector &words);
public:
  EST_BackoffNgrammarState()
    { init(); };
  EST_BackoffNgrammarState(const EST_Discrete *d,int level)
    {clear();init(d,level);};
  EST_BackoffNgrammarState(const EST_DiscreteProbDistribution &pdf,int level)
    {clear();init(pdf,level);};
  EST_BackoffNgrammarState(const EST_BackoffNgrammarState &s);
  EST_BackoffNgrammarState(const EST_BackoffNgrammarState *const s);
  ~EST_BackoffNgrammarState();
  
  // initialise
  void clear();
  void init();
  void init(const EST_Discrete *d, int level);
  void init(const EST_DiscreteProbDistribution &pdf, int level);
  
  // build
  bool accumulate(const EST_StrVector &words,
		  const double count=1);
  bool accumulate(const EST_IVector &words,
		  const double count=1);
  // access
  const EST_DiscreteProbDistribution &pdf_const() const {return p_pdf; };
  EST_DiscreteProbDistribution &pdf() {return p_pdf; };
  double probability(const EST_String &w) const
    {return p_pdf.probability(w);}
  double frequency(const EST_String &w) const
    {return p_pdf.frequency(w);}
  const EST_String &most_probable(double *prob = NULL) const
    {return p_pdf.most_probable(prob);}

  const int level() const {return p_level;}
  
  EST_BackoffNgrammarState* get_child(const EST_String &word) const
    {
	return (EST_BackoffNgrammarState*)children.lookup(word);
    }
  EST_BackoffNgrammarState* get_child(const int word) const
    {
	return (EST_BackoffNgrammarState*)children.lookup(p_pdf.get_discrete()->name(word));
    }
  
  void remove_child(EST_BackoffNgrammarState *child,
		    const EST_String &name);

  // recursive delete of contents and children
  void zap();

  const EST_BackoffNgrammarState *const get_state(const EST_StrVector &words) const;

  bool ngram_exists(const EST_StrVector &words,
		    const double threshold) const;
  const double get_backoff_weight() const {return backoff_weight; }
  const double get_backoff_weight(const EST_StrVector &words) const;
  bool set_backoff_weight(const EST_StrVector &words, const double w);
  void frequency_of_frequencies(EST_DVector &ff);
  
  void print_freqs(ostream &os,const int order,EST_String followers="");
  
friend ostream&  operator<<(ostream& s, const EST_BackoffNgrammarState &a);
  
};

class EST_Ngrammar {
    
public:

    // 3 representations : sparse, dense and backed off. User specifies which.
    enum representation_t {sparse, dense, backoff};
    
    // now only keep frequencies (or log frequencies)
    // probabilities (or log probabilities) can be done
    // on the fly quickly enough
    enum entry_t {frequencies, log_frequencies};
    
protected:
    
    // each instance of an EST_Ngrammar is a grammar of fixed order
    // e.g. a bigram (order = 2)
    int p_order;
    int p_num_samples;

    double p_number_of_sentences; // which were used to build this grammar


    EST_String p_sentence_start_marker;
    EST_String p_sentence_end_marker;

    // only one representation in use at a time 
    representation_t p_representation; 
    entry_t p_entry_type;

    // sparse representation is a tree structure
    // holding only those N-grams which were seen
    EST_PredictionSuffixTree sparse_representation;
    bool init_sparse_representation();

    // dense representation is just an array of all states
    bool init_dense_representation();

    // backoff representation is also a tree structure
    // but the root state pdf is the most recent word in the
    // ngram and going down the tree is going back in time....
    // here is the root node :
    EST_BackoffNgrammarState *backoff_representation;

    double backoff_threshold;

    // need a non-zero unigram floor to enable backing off
    double backoff_unigram_floor_freq;

    // instead of simple discounting, we have a (possibly) different
    // discount per order and per frequency
    // e.g. backoff_discount[2](4) contains the discount to be
    // applied to a trigram frequency of 4
    // backoff_discount[0] is unused (we don't discount unigrams)
    EST_DVector *backoff_discount;
    const double get_backoff_discount(const int order, const double freq) const;

    bool init_backoff_representation();
    void prune_backoff_representation(EST_BackoffNgrammarState *start_state=NULL); // remove any zero frequency branches
    void backoff_restore_unigram_states();
    int p_num_states; // == p_vocab_size ^ (p_ord-1) if fully dense
    EST_NgrammarState *p_states; // state id is index into this array
    int find_dense_state_index(const EST_IVector &words, int index=0) const;

    // and the reverse
    const EST_StrVector &make_ngram_from_index(const int i) const;
    
    // vocabulary
    EST_Discrete *vocab;
    EST_Discrete *pred_vocab;  // may be different from state vocab
    bool init_vocab(const EST_StrList &wordlist);
    bool init_vocab(const EST_StrList &word_list,
		    const EST_StrList &pred_list);

    // make sure vocab matches a given wordlist
    bool check_vocab(const EST_StrList &wordlist);

    EST_DiscreteProbDistribution vocab_pdf;  // overall pdf
    
    const EST_String &lastword(const EST_StrVector &words) const
        { return words(p_order-1); }
    const int lastword(const EST_IVector &words) const
        { return words(p_order-1); }
    // are we allowing out-of-vocabulary words, or is the vocabulary closed?
    bool allow_oov; 
    
    bool sparse_to_dense();
    bool dense_to_sparse();
    
    // these aren't sorted yet ...
    void take_logs();
    void take_exps();
    void freqs_to_probs(); // just calls normalise
    
    bool build_sparse(const EST_String &filename,
		      const EST_String &prev,
		      const EST_String &prev_prev,
		      const EST_String &last);
    // for dense and backoff
    bool build_ngram(const EST_String &filename,
		     const EST_String &prev,
		     const EST_String &prev_prev,
		     const EST_String &last,
		     const EST_String &input_format);

    // go through all matching ngrams ( *(ngram[i])="" matches anything )
    void iterate(EST_StrVector &words,
                 void (*function)(EST_Ngrammar *n,
                                  EST_StrVector &words, 
                                  void *params),
		 void *params);

    // same, but with a constant Ngrammar
    void const_iterate(EST_StrVector &words,
		       void (*function)(const EST_Ngrammar *const n,
					EST_StrVector &words, 
					void *params),
		       void *params) const;

    bool p_init(int o, representation_t r);

    // new filename returned of we had to copy stdin to a
    // temporary file - must delete it later !
    bool oov_preprocess(const EST_String &filename,
			EST_String &new_filename,
			const EST_String &what);

    
    const EST_NgrammarState &find_state_const(const EST_StrVector &words)const;
    EST_NgrammarState &find_state(const EST_StrVector &words);
    const EST_NgrammarState &find_state_const(const EST_IVector &words) const;
    EST_NgrammarState &find_state(const EST_IVector &words);
    
    // special versions for backoff grammars
    const EST_DiscreteProbDistribution &backoff_prob_dist(const EST_StrVector &words) const;    
    const double backoff_reverse_probability_sub(const EST_StrVector &words,
				    const EST_BackoffNgrammarState *root) const;
    const double backoff_probability(const EST_StrVector &words,
				     const bool trace=false) const;
    const double backoff_reverse_probability(const EST_StrVector &words) const;
    const EST_String & backoff_most_probable(const EST_StrVector &words,
					     double *prob = NULL) const;

    // backoff representation isn't a nice array of states
    // so use this to visit every node in the tree
    // and apply the function to that node
    void backoff_traverse(EST_BackoffNgrammarState *start_state,
			  void (*function)(EST_BackoffNgrammarState *s,
					   void *params),
			  void *params);
    
    // visit every node at a given level
    void backoff_traverse(EST_BackoffNgrammarState *start_state,
			  void (*function)(EST_BackoffNgrammarState *s,
					   void *params),
			  void *params, const int level);
public:

    EST_Ngrammar() {default_values();}

    EST_Ngrammar(int o, representation_t r, 
		 const EST_StrList &wordlist)
    { 
	default_values(); init(o,r,wordlist); 
    }

    // When state trans vocab differs from prediction vocab
    EST_Ngrammar(int o, representation_t r, 
		 const EST_StrList &wordlist,
		 const EST_StrList &predlist)
    { 
	default_values(); init(o,r,wordlist,predlist); 
    }

    EST_Ngrammar(int o, representation_t r, EST_Discrete &v)
    {
	default_values(); init(o,r,v); 
    }
    ~EST_Ngrammar();
    
    void default_values();
    void clear();
    bool init(int o, representation_t r, 
	      const EST_StrList &wordlist);
    bool init(int o, representation_t r, 
	      const EST_StrList &wordlist,
	      const EST_StrList &predlist);
    bool init(int o, representation_t r, EST_Discrete &v);
    bool init(int o, representation_t r, 
	      EST_Discrete &v,EST_Discrete &pv);
    
    // access
    int num_states(void) const { return p_num_states;}
    double samples(void) const { return p_num_samples;}
    int order() const { return p_order; }
    int get_vocab_length() const { return vocab?vocab->length():0; }
    EST_String get_vocab_word(int i) const;
    int get_vocab_word(const EST_String &s) const;
    int get_pred_vocab_length() const { return pred_vocab->length(); }
    EST_String get_pred_vocab_word(int i) const { return pred_vocab->name(i); }
    int get_pred_vocab_word(const EST_String &s) const 
       { return pred_vocab->name(s); }
    int closed_vocab() const {return !allow_oov; }
    entry_t entry_type() const {return p_entry_type;}
    representation_t representation() const 
       { return p_representation;}
    
    // build
    bool build(const EST_StrList &filenames,
	       const EST_String &prev = SENTENCE_START_MARKER,
	       const EST_String &prev_prev = SENTENCE_END_MARKER,
	       const EST_String &last = SENTENCE_END_MARKER,
	       const EST_String &input_format = "",
	       const EST_String &oov_mode = "",
	       const int mincount=1,
	       const int maxcount=10);
    
    // Accumulate ngrams
    void accumulate(const EST_StrVector &words,
		    const double count=1);
    //const int index=0);
    void accumulate(const EST_IVector &words,
		    const double count=1);
    //const int index=0);
    
    // hack - fix enter/exit probs s.t. P(...,!ENTER)=P(!EXIT,...)=0, for all x
    void make_htk_compatible();

    // I/O functions 
    EST_read_status load(const EST_String &filename);
    EST_read_status load(const EST_String &filename, const EST_StrList &wordlist);
    EST_write_status save(const EST_String &filename, 
			  const EST_String type="cstr_ascii", 
			  const bool trace=false,
			  double floor=0.0);
    
    int wordlist_index(const EST_String &word, const bool report=true) const;
    const EST_String &wordlist_index(int i) const;
    int predlist_index(const EST_String &word) const;
    const EST_String &predlist_index(int i) const;
    
    // set
    bool set_entry_type(entry_t new_type);
    bool set_representation(representation_t new_representation);

    // probability distributions
    // -------------------------
    // flag 'force' forces computation of probs on-the-fly if necessary
    double probability(const EST_StrVector &words, bool force=false,
		       const bool trace=false) const;
    double frequency(const EST_StrVector &words, bool force=false,
		     const bool trace=false) const;

    const EST_String &predict(const EST_StrVector &words,
			      double *prob,int *state) const;
    const EST_String &predict(const EST_StrVector &words) const
       {double p; int state; return predict(words,&p,&state); }
    const EST_String &predict(const EST_StrVector &words,double *prob) const
       {int state; return predict(words,prob,&state); }
    
    const EST_String &predict(const EST_IVector &words,double *prob,int *state) const;
    const EST_String &predict(const EST_IVector &words) const
       {double p; int state; return predict(words,&p,&state); }
    const EST_String &predict(const EST_IVector &words,double *prob) const
       {int state; return predict(words,prob,&state); }
    
    int find_state_id(const EST_StrVector &words) const;
    int find_state_id(const EST_IVector &words) const;
    int find_next_state_id(int state, int word) const;
    // fast versions for common N
    //const double probability(const EST_String w1);
    //const double probability(const EST_String w1,const EST_String w2);
    //const double probability(const EST_String w1,const EST_String w2,
    //const EST_String w2);

    // reverse - probability of words[0..order-2] given word[order-1]
    double reverse_probability(const EST_StrVector &words,
			       bool force=false) const;
    double reverse_probability(const EST_IVector &words,
			       bool force=false) const;
    
    // predict, where words has 'order' elements and the last one is "" or NULL
    const EST_DiscreteProbDistribution &prob_dist(const EST_StrVector &words) const;
    const EST_DiscreteProbDistribution &prob_dist(const EST_IVector &words) const;
    const EST_DiscreteProbDistribution &prob_dist(int state) const;
    
//    bool stats(const EST_String filename,
//	       double &raw_entropy, double &count,
//	       double &entropy, double &perplexity,
//	       const EST_String &prev = SENTENCE_START_MARKER,
//	       const EST_String &prev_prev = SENTENCE_END_MARKER,
//	       const EST_String &last = SENTENCE_END_MARKER,
//               const EST_String &input_format = "") const;
    
    void fill_window_start(EST_IVector &window, 
			   const EST_String &prev,
			   const EST_String &prev_prev) const;

    void fill_window_start(EST_StrVector &window, 
			   const EST_String &prev,
			   const EST_String &prev_prev) const;

    // why anybody would want to do this ....
    //EST_Ngrammar &operator =(const EST_Ngrammar &a);

    bool ngram_exists(const EST_StrVector &words) const;
    bool ngram_exists(const EST_StrVector &words, const double threshold) const;
    const double get_backoff_weight(const EST_StrVector &words) const;
    bool set_backoff_weight(const EST_StrVector &words, const double w);
    
    void print_freqs(ostream &os,double floor=0.0);
    
    // i/o functions
    // -------------
    friend ostream& operator<<(ostream& s, EST_Ngrammar &n);
    friend EST_read_status load_ngram_htk_ascii(const EST_String filename, 
						EST_Ngrammar &n);
    friend EST_read_status load_ngram_htk_binary(const EST_String filename, 
						 EST_Ngrammar &n);
    friend EST_read_status load_ngram_arpa(const EST_String filename, 
					   EST_Ngrammar &n, 
					   const EST_StrList &vocab);
    friend EST_read_status load_ngram_cstr_ascii(const EST_String filename, 
					     EST_Ngrammar &n);
    friend EST_read_status load_ngram_cstr_bin(const EST_String filename, 
					       EST_Ngrammar &n);
    
    friend EST_write_status save_ngram_htk_ascii_sub(const EST_String &word,
						     ostream *ost, 
						     EST_Ngrammar &n,
						     double floor);
    friend EST_write_status save_ngram_htk_ascii(const EST_String filename, 
						 EST_Ngrammar &n,
						 double floor);

    //friend EST_write_status save_ngram_htk_binary(const EST_String filename, 
    //					  EST_Ngrammar &n);
    friend EST_write_status save_ngram_cstr_ascii(const EST_String filename, 
						  EST_Ngrammar &n,
						  const bool trace,
						  double floor);
    friend EST_write_status save_ngram_cstr_bin(const EST_String filename, 
						EST_Ngrammar &n, 
						const bool trace,
						double floor);
    friend EST_write_status save_ngram_arpa(const EST_String filename, 
					    EST_Ngrammar &n);
    friend EST_write_status save_ngram_arpa_sub(ostream *ost, 
						EST_Ngrammar &n, 
						const EST_StrVector &words);
    friend EST_write_status save_ngram_wfst(const EST_String filename, 
					    EST_Ngrammar &n);

    // Auxiliary functions
    
    // smoothing
friend void frequency_of_frequencies(EST_DVector &ff, EST_Ngrammar &n,int this_order);
friend void map_frequencies(EST_Ngrammar &n, const EST_DVector &map, const int this_order);
friend bool Good_Turing_smooth(EST_Ngrammar &n, int maxcount, int mincount);
friend void Good_Turing_discount(EST_Ngrammar &ngrammar, const int maxcount,
				 const double default_discount);

friend void fs_build_backoff_ngrams(EST_Ngrammar *backoff_ngrams,
				    EST_Ngrammar &ngram);
friend int fs_backoff_smooth(EST_Ngrammar *backoff_ngrams,
			     EST_Ngrammar &ngram, int smooth_thresh);

    // frequencies below mincount get backed off
    // frequencies above maxcount are not smoothed(discounted)
    bool compute_backoff_weights(const int mincount=1,
				 const int maxcount=10);


    bool merge(EST_Ngrammar &n,float weight);

friend class EST_BackoffNgrammar;
    
};

void Ngram_freqsmooth(EST_Ngrammar &ngram,
		      int smooth_thresh1,
		      int smooth_thresh2);

// utils
void slide(EST_IVector &i, const int l);
void slide(EST_StrVector &i, const int l);

bool test_stats(EST_Ngrammar &ngram, 
		const EST_String &filename,
		double &raw_entropy,
		double &count,
		double &entropy,
		double &perplexity,
		const EST_String &input_format,
		const EST_String &prev = SENTENCE_START_MARKER, 
		const EST_String &prev_prev = SENTENCE_END_MARKER,
		const EST_String &last = SENTENCE_END_MARKER);

VAL_REGISTER_CLASS_DCLS(ngrammar,EST_Ngrammar)

#endif // __EST_NGRAMMAR_H__
