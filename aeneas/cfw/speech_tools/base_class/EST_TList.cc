 /*************************************************************************/
 /*                                                                       */
 /*                Centre for Speech Technology Research                  */
 /*                     University of Edinburgh, UK                       */
 /*                      Copyright (c) 1995,1996                          */
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
 /*                                                                       */
 /*                      Author :  Paul Taylor                            */
 /*                      Date   :  April 1995                             */
 /* --------------------------------------------------------------------- */
 /*                       Template List Class                             */
 /*                                                                       */
 /* Modified by RJC, 21/7/97. Now much of the working code is in the      */
 /* UList class, this template class provides a type safe front end to    */
 /* the untyped list.                                                     */
 /*                                                                       */
 /*************************************************************************/

#include "EST_TList.h"

template<class T> EST_TItem<T> *EST_TItem<T>::make(const T &val)
{
  EST_TItem<T> *it=NULL;
  if (s_free!=NULL)
    {
      void *mem = s_free;
      s_free=(EST_TItem<T> *)s_free->n;
      s_nfree--;

      // Create an item in the retrieved memory.
      it=new (mem) EST_TItem<T>(val);
    }
  else
    it = new EST_TItem<T>(val);

  return it;
}

template<class T> void EST_TItem<T>::release(EST_TItem<T> *it)
{
    if (0) // (s_nfree < s_maxFree)
    {
      // Destroy the value in case it holds resources.
      it->EST_TItem<T>::~EST_TItem();

      // I suppose it's a bit weird to use 'n' after calling the destructor.
      it->n=s_free;
      s_free=it;
      s_nfree++;
    }
  else
  {
      delete it;
  }
}

template<class T> void EST_TList<T>::copy_items(const EST_TList<T> &l)
{
    EST_UItem *p;
    for (p = l.head();  p; p = p->next())
	append(l.item(p));
}

template<class T> void EST_TList<T>::free_item(EST_UItem *item)
{ EST_TItem<T>::release((EST_TItem<T> *)item); }


template<class T> EST_TList<T>::EST_TList(const EST_TList<T> &l)
{
    init();
    copy_items(l);
}

template<class T> void EST_TList<T>::exchange_contents(EST_Litem *a,EST_Litem *b)
{

    if(a==b)
	return;

    T temp;

    temp = ((EST_TItem<T> *)a)->val;
    ((EST_TItem<T> *)a)->val = ((EST_TItem<T> *)b)->val;
    ((EST_TItem<T> *)b)->val = temp;

}


template<class T> EST_TList<T> &EST_TList<T>::operator=(const EST_TList<T> &a) 
{
    clear();			// clear out all current items in list.
    copy_items(a);
    return *this;
}


template<class T> EST_TList<T> &EST_TList<T>::operator+=(const EST_TList<T> &a)
{
    if (this == &a)
    {
	cerr << "EST_TList: error: tried to add list to itself\n";
	return *this;
    }
    copy_items(a);
    return *this;
}



