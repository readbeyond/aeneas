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
/* Simple statistics (for discrete probability distributions             */
/*                                                                       */
/*=======================================================================*/
#ifndef __EST_SIMPLESTATS_H__
#define __EST_SIMPLESTATS_H__

#include "EST_String.h"
#include "EST_Token.h"
#include "EST_StringTrie.h"
#include "EST_TList.h"
#include "EST_TKVL.h"
#include "EST_types.h"

typedef size_t int_iter; 

/** A class for managing mapping string names to integers and back again,
    mainly used for representing alphabets in n-grams and grammars etc.

    This offers an efficient way of mapping a known set of string names
    to integers.  It is initialised from a list of names and builds
    a index of those names to a set of integers.

    @author Alan W Black (awb@cstr.ed.ac.uk): July 1996
*/
class EST_Discrete {
private:
    // for fast index->name
    EST_StrVector namevector;
    int p_def_val;
    // for fast name->index
    EST_StringTrie nametrie;
    
public:
    ///
    EST_Discrete() {nametrie.clear(); p_def_val = -1;}
    ///
    EST_Discrete(const EST_Discrete &d) { copy(d); }
    /// Initialise discrete class from given list of strings
    EST_Discrete(const EST_StrList &vocab);
    ///
    ~EST_Discrete();
    /// 
    void copy(const EST_Discrete &d);
    /// (re-)initialise
    bool init(const EST_StrList &vocab);

    /// The number of members in the discrete
    const int length(void) const { return namevector.length(); }
    /** The int assigned to the given name, if it doesn't exists p\_def\_val
        is returned (which is -1 by default)
    */
    const int index(const EST_String &n) const { 
	int *i;
	return (((i=(int*)nametrie.lookup(n)) != NULL) ? *i : p_def_val);
    };

    /// The name given the index
    const EST_String &name(const int n) const { return namevector(n); }

    /// set the default value when a name isn't found (-1 by default)
    void def_val(const EST_String &v) { p_def_val = index(v); }
    
    /// An alternative method for getting the int form the name
    int name(const EST_String &n) const { return index(n); };

    bool operator == (const EST_Discrete &d);
    bool operator != (const EST_Discrete &d);

    EST_String print_to_string(int quote=0);
    friend ostream& operator <<(ostream& s, const EST_Discrete &d);

    ///
    EST_Discrete & operator = (const EST_Discrete &a) 
      { copy(a); return *this; }

};

class Discretes {
  private:
    int max;
    int next_free;
    EST_Discrete **discretes;
  public:
    Discretes() {max=50;next_free=0;discretes=new EST_Discrete*[max];}
    ~Discretes();
    const int def(const EST_StrList &members);
    EST_Discrete &discrete(const int t) const {return *discretes[t-10];}
    EST_Discrete &operator [] (const int t) const {return *discretes[t-10];}
};

/** A class for cummulating ``sufficient statistics'' for a set of
    numbers: sum, count, sum squared.

    This collects the number, sum and sum squared for a set of number.
    Offering mean, variance and standard deviation derived from the
    cummulated values.

    @author Alan W Black (awb@cstr.ed.ac.uk): July 1996
 */
class EST_SuffStats {
private:
    double n;  // allows frequencies to be non-integers
    double p_sum;
    double p_sumx;
public:
    ///
    EST_SuffStats() {n = p_sum = p_sumx = 0.0;}
    ///
    EST_SuffStats(double in, double isum, double isumx) 
	{n = in; p_sum = isum; p_sumx = isumx;}
    ///
    EST_SuffStats(const EST_SuffStats &s) { copy(s); }
    ///
    void copy(const EST_SuffStats &s) 
       {n=s.n; p_sum = s.p_sum; p_sumx = s.p_sumx;}
    /// reset internal values
    void reset(void) {n = p_sum = p_sumx = 0.0;}
    void set(double in, double isum, double isumx) 
	{n = in; p_sum = isum; p_sumx = isumx;}
    /// number of samples in set
    double samples(void) {return n;}
    /// sum of values
    double sum() { return p_sum; }
    /// sum of squared values 
    double sumx() { return p_sumx; }
    /// mean of currently cummulated values
    double mean(void) const { return (n==0)?0.0:(p_sum / n); }
    /// variance of currently cummulated values 
    double variance(void) const 
       { return ((n*p_sumx)-(p_sum*p_sum))/((double)n*(n-1)); }
    /// standard deviation of currently cummulated values
    double stddev(void) const { return sqrt(variance()); }

    void cumulate(double a,double count=1.0)
	{ n+=count; p_sum+=a*count; p_sumx+=count*(a*a); }

    /// Used to cummulate new values
    EST_SuffStats &operator +=(double a) 
         { cumulate(a,1.0); return *this;}
    /// Used to cummulate new values
    EST_SuffStats &operator + (double a) 
         { cumulate(a,1.0); return *this;}
    /// 
    EST_SuffStats &operator = (const EST_SuffStats &a) 
         { copy(a); return *this;}
};

enum EST_tprob_type {tprob_string, tprob_int, tprob_discrete};
/** A class for representing probability distributions for a set of
    discrete values.

    This may be used to cummulate the probability distribution of a 
    class of values.  Values are actually help as frequencies so both
    frequency and probability information may be available.   Note that
    frequencies are not integers because using smoothing and backoff
    integers are too restrictive so they are actually represented as
    doubles.

    Methods are provided to iterate over the values in a distribution,
    for example
    \begin{verbatim}
       EST_DiscreteProbistribution pdf;
       for (int i=pdf.item_start(); i < pdf.item_end(); i=pdf.item_next(i))
       {
          EST_String name;
          double prob;
          item_prob(i,name,prob);
          cout << name << ": prob " << prob << endl;
       }
    \end{verbatim}

    @author Alan W Black (awb@cstr.ed.ac.uk): July 1996
*/
class EST_DiscreteProbDistribution {
private:
    double num_samples;	   // because frequencies don't have to be integers
    EST_tprob_type type;
    /* For known vocabularies: tprob_discrete */
    const EST_Discrete *discrete;
    // was int, but frequencies don't have to be integers
    EST_DVector icounts;	
    /* For unknown vocabularies: tprob_string */
    EST_StrD_KVL scounts;
public:
    EST_DiscreteProbDistribution() : type(tprob_string), discrete(NULL), icounts(0), scounts() {init();}
    /// Create with copying from an existing distribution.
    EST_DiscreteProbDistribution(const EST_DiscreteProbDistribution &b);
    /// Create with given vocabulary
    EST_DiscreteProbDistribution(const EST_TList<EST_String> &vocab)
          {init(); (void)init(vocab);}
    /// Create using given \Ref{EST_Discrete} class as the vocabulary
    EST_DiscreteProbDistribution(const EST_Discrete *d) {init(); init(d);}
    /** Create using given \Ref{EST_Discrete} class as vocabulary plus given
        counts
    */
    EST_DiscreteProbDistribution(const EST_Discrete *d,
				 const double n_samples, 
				 const EST_DVector &counts);

    /// Destructor function
    ~EST_DiscreteProbDistribution() {clear();}
    /// Copy all data from another DPD to this
    void copy(const EST_DiscreteProbDistribution &b);

    /// Reset, clearing all counts and vocabulary
    void clear(void);
    /// Initialise using given vocabulary
    bool init(const EST_StrList &vocab);
    /// Initialise using given \Ref{EST_Discrete} as vocabulary
    void init(const EST_Discrete *d);
    /// Initialise
    void init();
    /// Total number of example found.
    double samples(void) const { return num_samples; }
    /// Add this observation, may specify number of occurrences
    void cumulate(const EST_String &s,double count=1);
    /// Add this observation, i must be with in EST\_Discrete range
    void cumulate(EST_Litem *i,double count=1);
    void cumulate(int i,double count=1);
    /// Return the most probable member of the distribution
    const EST_String &most_probable(double *prob = NULL) const;
    /** Return the entropy of the distribution
        \[ -\sum_{i=1}^N(prob(i)*log(prob(i))) \]
    */
    double entropy(void) const;
    /// 
    double probability(const EST_String &s) const; 
    /// 
    double probability(const int i) const; 
    ///
    double frequency(const EST_String &s) const; 
    /// 
    double frequency(const int i) const; 
    /// Used for iterating through members of the distribution
    EST_Litem *item_start() const;
    /// Used for iterating through members of the distribution
    EST_Litem *item_next(EST_Litem *idx) const;
    /// Used for iterating through members of the distribution
    int item_end(EST_Litem *idx) const;

    /// During iteration returns name given index 
    const EST_String &item_name(EST_Litem *idx) const;
    /// During iteration returns name and frequency given index  
    void item_freq(EST_Litem *idx,EST_String &s,double &freq) const;
    /// During iteration returns name and probability given index
    void item_prob(EST_Litem *idx,EST_String &s,double &prob) const;

    /// Returns discrete vocabulary of distribution
    inline const EST_Discrete *const get_discrete() const { return discrete; };
    
    /** Sets the frequency of named item, modifies {\tt num\_samples}
         accordingly.  This is used when smoothing frequencies.
    */
    void set_frequency(const EST_String &s,double c);
    /** Sets the frequency of named item, modifies {\tt num\_samples}
        accordingly.  This is used when smoothing frequencies.
    */
    void set_frequency(int i,double c); 
    void set_frequency(EST_Litem *i,double c); 
    
    /// Sets the frequency of named item, without modifying {\tt num\_samples}.
    void override_frequency(const EST_String &s,double c);
    /// Sets the frequency of named item, without modifying {\tt num\_samples}.
    void override_frequency(int i,double c); 
    void override_frequency(EST_Litem *i,double c); 
    
    /** Sets the number of samples.  Care should be taken on setting this
        as it will affect how probabilities are calculated.
    */
    void set_num_samples(const double c) { num_samples = c;}
    
friend ostream & operator <<(ostream &s, const EST_DiscreteProbDistribution &p);
    EST_DiscreteProbDistribution &operator=(const EST_DiscreteProbDistribution &a);
};    

#endif				// __EST_SIMPLESTATS_H__
