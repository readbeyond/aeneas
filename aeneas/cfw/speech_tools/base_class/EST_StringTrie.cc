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
/*                     Date   :  June 1996                               */
/*-----------------------------------------------------------------------*/
/*                                                                       */
/* A class for building EST_String (char-based) tries for indexing       */
/* arbitrary objects by Strings                                          */
/*                                                                       */
/*=======================================================================*/
#include "EST_String.h"
#include "EST_StringTrie.h"
#include <cstring>

#define TRIEWIDTH 256

static void (* trie_delete_function)(void *n) = 0;

static inline int char2idx(unsigned char k)
{
//    return k & 0x7f;  // only seven significant bits;
    return k;
}

EST_TrieNode::EST_TrieNode(const int width)
{
    // Initialise a node of given width
    w=width; 
    d= new EST_TrieNode *[w]; 
    contents=0;
    memset(d,0,w*sizeof(EST_TrieNode *));
}

EST_TrieNode::~EST_TrieNode()
{
    int i;

    if (trie_delete_function != 0)   /* user supplied delete function */
	trie_delete_function(contents);
    for (i=0; i<w; i++)
	delete d[i];
    delete [] d;
}
    
void *EST_TrieNode::lookup(const unsigned char *key) const
{
    // find key in EST_TrieNode, 0 if not found 

    if (*key == '\0')
	return contents;  // base case
    else
    {
	int idx = char2idx(*key);
	if (d[idx] == 0)
	    return 0;     // not there
	else
	    return d[idx]->lookup(key+1);
    }
}

void EST_TrieNode::copy_into(EST_StringTrie &trie,
			     const EST_String &path) const
{
    // find all items and add them to trie

    if (contents != 0)
	trie.add(path,contents);

    for (int i=0; i < w; i++)
    {
	if (d[i] != 0)
	{
	    char tail[2];
	    tail[0] = (char)i;
	    tail[1] = '\0';
	    d[i]->copy_into(trie,path+tail);
	}
    }
}

void EST_TrieNode::add(const unsigned char *key,void *value)
{
    // add this value

    if (*key == '\0')
	contents = value;
    else
    {
	int idx = char2idx(*key);
	if (d[idx] == 0) // need new subnode
	    d[idx] = new EST_TrieNode(w);
	d[idx]->add(key+1,value);
    }
}

EST_StringTrie::EST_StringTrie()
{
    tree = new EST_TrieNode(TRIEWIDTH);
}

void EST_StringTrie::copy(const EST_StringTrie &trie)
{
    // This can't work because of the void* pointers in contents
    delete tree;
    tree = new EST_TrieNode(TRIEWIDTH);
    trie.tree->copy_into(*this,"");
}

EST_StringTrie::~EST_StringTrie()
{
    delete tree;
}

void *EST_StringTrie::lookup(const EST_String &key) const
{
    const unsigned char *ckey = (const unsigned char *)(void *)(const char *)key;
    return tree->lookup(ckey);
}

void EST_StringTrie::add(const EST_String &key,void *item)
{
    const unsigned char *ckey = (const unsigned char *)(void *)(const char *)key;
    tree->add(ckey,item);
    return;
}
    
void EST_StringTrie::clear(void)
{
    delete tree;
    tree = new EST_TrieNode(TRIEWIDTH);
}

void EST_StringTrie::clear(void (*deletenode)(void *n))
{
    // This wont work if we go multi-thread
    trie_delete_function = deletenode;
    delete tree;
    trie_delete_function = 0;
    tree = new EST_TrieNode(TRIEWIDTH);
}

