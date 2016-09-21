/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                         Copyright (c) 1998                            */
/*                        All Rights Reserved.                           */
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
/*                       Author :  Alan W Black                          */
/*                       Date   :  February 1998                         */
/* --------------------------------------------------------------------- */
/*   Functions for LIST relations                                        */
/*                                                                       */
/*************************************************************************/
#ifndef __EST_RELATION_LIST_H__
#define __EST_RELATION_LIST_H__


#if 0
/**@name Functions for building and traversing list relations
 */

//@{
/**@name List traversal functions */
//@{

/** return next item of <parameter>n</parameter>
 */
inline EST_Item *next(const EST_Item *n) { return n->next(); }

/** return previous item of <parameter>n</parameter>
 */
inline EST_Item *prev(const EST_Item *n) { return n->prev(); }

/** return last item in <parameter>n</parameter>'s relation
 */
inline EST_Item *last(const EST_Item *n) { return n->last(); }

/** return first item in <parameter>n</parameter>'s relation
 */
inline EST_Item *first(const EST_Item *n) { return n->first(); }

/** return next item of <parameter>n</parameter> as seen from relation
<parameter>relname</parameter> */
inline EST_Item *next(const EST_Item *n,const char *relname)
    { return next(as(n,relname)); }

/** return previous item of <parameter>n</parameter> as seen from relation
<parameter>relname</parameter> */
inline EST_Item *prev(const EST_Item *n,const char *relname)
    { return prev(as(n,relname)); }

/** return first item of <parameter>n</parameter> as seen from relation
<parameter>relname</parameter> */
inline EST_Item *first(const EST_Item *n,const char *relname) 
    { return first(as(n,relname)); }

/** return last item of <parameter>n</parameter> as seen from relation
<parameter>relname</parameter> */
inline EST_Item *last(const EST_Item *n,const char *relname) 
    { return last(as(n,relname)); }

#endif

/** Given a node <parameter>l</parameter>, return true if
    <parameter>c</parameter> after it in a list relation. */
int in_list(const EST_Item *c, const  EST_Item *l);


/** Add a item after node <parameter>n</parameter>, and return the new
item. If <parameter>n</parameter> is the first item in the list, the 
new item becomes the head of the list, otherwise it is inserted between
<parameter>n</parameter> and it's previous current item.
If <parameter>p</parameter> is 0, make a new node for the new
item, otherwise add <parameter>p</parameter> to this relation as the
next item in <parameter>n</parameter>'s relation.  */

EST_Item *add_after(const EST_Item *n, EST_Item *p=0);

/** Add a item before node <parameter>n</parameter>, and return the new
item. If <parameter>n</parameter> is the first item in the list, the 
new item becomes the head of the list, otherwise it is inserted between
<parameter>n</parameter> and it's previous current item.
If <parameter>p</parameter> is 0, make a new node for the new
item, otherwise add <parameter>p</parameter> to this relation as the
previous item in <parameter>n</parameter>'s relation.  */

EST_Item *add_before(const EST_Item *n, EST_Item *p=0);

/** Remove the given item.
*/

void remove_item_list(EST_Relation *rel, EST_Item *n);

//@}
//@}
#endif
