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
/*                    Author :  Simon King                               */
/*                    Date   :  November 1996                            */
/*-----------------------------------------------------------------------*/
/*                   Lattice/Finite State Network                        */
/*                                                                       */
/*=======================================================================*/


#ifndef __EST_LATTICE_H__
#define __EST_LATTICE_H__

#include "EST_types.h"
#include "EST_Track.h"

class Lattice {

public:

    /*
    struct quantised_label_table_entry_t{
	int index;
	float value;
    };
    */

    /*
    struct name_map_entry_t{
	int index;
	String name;
    };
    */

    struct symbol_t{
	int qmap_index;
	int nmap_index;

	symbol_t operator += (const symbol_t s2);
	bool operator != (const symbol_t &s2) const;
    };

    struct Node;
    struct Arc;

    struct Arc{
	int label;
	Node *to;
    };


    struct Node{
	EST_IList name; // list of ints, referring to the table of names
	EST_TList<Arc*> arcs_out;
    };

private:

protected:

    // not necessarily defined or used ...
    //friend inline ostream& operator<<(ostream &s, Lattice::quantised_label_table_entry_t &q);
    //friend inline ostream& operator<<(ostream& s, Lattice::name_map_entry_t &n);
    friend inline ostream& operator<<(ostream& s, const Lattice::symbol_t &sy);
    friend inline ostream& operator<<(ostream& s, const Lattice::Node &n);
    friend inline ostream& operator<<(ostream& s, const Lattice::Arc &n);


    // maps, for speed
    float qmap_error_margin; // only used in construction, so remove .. to do

    // quantised log probabilities
    EST_FVector qmap;

    // 'words'
    EST_TVector<EST_String> nmap;

    // not used
    EST_String name_as_string(EST_IList &l); // given a list of nmap indices

    // the finite state machines alphabet
    EST_TVector<symbol_t> alphabet;
    int e_move_symbol_index;
    //int enter_move_symbol_index;

    //symbol_t* alphabet_lookup(int nmap_index, int qmap_index);
    //symbol_t* alphabet_lookup_from_end(int nmap_index, int qmap_index);


    int alphabet_index_lookup(int nmap_index, int qmap_index); // return index
    //int alphabet_index_lookup_from_end(int nmap_index, int qmap_index); // return index
    

    // the nodes
    EST_TList<Node*> nodes;

    //Node* start_node; // a subset of nodes

    EST_TList<Node*> final_nodes; // a subset of nodes

    bool final(Node *n);

    // an alternative representation is a transition function
    // useful (fast) for dense networks, but inefficient for sparse ones
    int **tf; // indexed [node index][symbol index], contains destination node
    bool build_transition_function();

    bool build_distinguished_state_table(bool ** &dst);
    bool build_distinguished_state_table_direct(bool  ** &dst);
    bool build_distinguished_state_table_from_transition_function(bool ** &dst);

    void sort_arc_lists();
    bool link(Node *n1, Node *n2, int label); //, EST_TList<Arc*> *l = NULL);

    void merge_nodes(EST_TList<Node*> &l);
    void merge_arcs();
    void prune_arc(Node *node, Arc *arc);
    void prune_arcs(Node *node, EST_TList<Arc*> arcs);
    void remove_arc_from_nodes_out_list(Node *n, Arc *a);

    int node_index(Node *n); // only for output in HTK format

// SIMONK FIX THIS
//    bool build_qmap(Bigram &g, float error_margin=0);
//    bool build_nmap(Bigram &g);

public:
    Lattice();
    ~Lattice();

// SIMONK FIX THIS
//    bool construct_alphabet(Bigram &g);
//    bool construct(Bigram &g);
    bool determinise();
    bool prune();
    bool minimise();
    bool expand();

    Node *start_node();

    // traversing functions
    bool accepts(EST_TList<symbol_t*> &string);
    float viterbi_transduce(EST_TList<EST_String> &input,
			    EST_TList<Arc*> &path,
			    EST_Litem *current_symbol = NULL,
			    Node *start_node = NULL);

    // observations are indexed same as wordlist, excluding !ENTER and !EXIT
    float viterbi_transduce(EST_Track &observations,
			    EST_TList<Arc*> &path,
			    float &score,
			    int current_frame = 0,
			    Node *start_node = NULL);

    // map lookup functions

    float qmap_index_to_value(int index);
    int qmap_value_to_index(float value);

    EST_String nmap_index_to_name(int index);
    int nmap_name_to_index(const EST_String &name);

    symbol_t* alphabet_index_to_symbol(int index);
    int alphabet_symbol_to_index(symbol_t *sym);


    friend bool save(Lattice &lattice, EST_String filename);
    friend bool load(Lattice &lattice, EST_String filename);

    friend class Lattice_Language_Model;

};

/*
inline int
operator > (const Lattice::name_map_entry_t &n1,
	    const Lattice::name_map_entry_t &n2)
{
    return (n1.name > n2.name);
};

inline int
operator < (const Lattice::name_map_entry_t &n1,
	    const Lattice::name_map_entry_t &n2)
{
    return (n1.name < n2.name);
};
*/

inline int
operator > (const Lattice::symbol_t s1,
	    const Lattice::symbol_t s2)
{
    if(s1.qmap_index > s2.qmap_index)
	return true;

    if(s1.qmap_index < s2.qmap_index)
	return false;

    return (s1.nmap_index > s2.nmap_index);
}

inline int
operator < (const Lattice::symbol_t s1,
	    const Lattice::symbol_t s2)
{
    if(s1.qmap_index < s2.qmap_index)
	return true;

    if(s1.qmap_index > s2.qmap_index)
	return false;

    return (s1.nmap_index < s2.nmap_index);
}

inline Lattice::symbol_t
operator + (const Lattice::symbol_t s1,
	    const Lattice::symbol_t s2)
{
    (void) s1;
    (void) s2;
    cerr << "operator + makes no sense for Lattice::symbol_t !" << endl;
    return Lattice::symbol_t();

}

// used for sorting arcs lists
inline int operator > (Lattice::Arc a1, Lattice::Arc a2)
{
    return (a1.label > a2.label);
}

inline int operator < (Lattice::Arc a1, Lattice::Arc a2)
{
    return (a1.label < a2.label);
}

inline int operator >= (Lattice::Arc a1, Lattice::Arc a2)
{
    return (a1.label >= a2.label);
}

inline int operator <= (Lattice::Arc a1, Lattice::Arc a2)
{
    return (a1.label <= a2.label);
}

inline int operator == (Lattice::Arc a1, Lattice::Arc a2)
{
    return (a1.label == a2.label);
}

inline int operator == (Lattice::symbol_t s1, Lattice::symbol_t s2)
{
    return ((s1.nmap_index == s2.nmap_index) &&
	    (s1.qmap_index == s2.qmap_index) );
}


/*
inline ostream& operator<<(ostream &s, Lattice::quantised_label_table_entry_t &q){
    s << q.value;
    return s;
}
*/

/*
inline ostream& operator<<(ostream& s, Lattice::name_map_entry_t &n)
{
    s << n.index << "=" << n.name;
    return s;
}
*/

inline ostream& operator<<(ostream& s, const Lattice::symbol_t &sm)
{
    s << "[q=" << sm.qmap_index << ",n=" << sm.nmap_index << "]";
    return s;
}


inline ostream& operator<<(ostream& s, const Lattice::Node &n)
{
    s << "Node:" << n.name;
    return s;
}

inline ostream& operator<<(ostream &s, const Lattice::Arc &a)
{
    s << a.label;
    return s;
}


void sort_by_label(EST_TList<Lattice::Arc*> &l);


#endif
