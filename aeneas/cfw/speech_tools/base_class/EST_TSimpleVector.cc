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
 /*                                                                       */
 /*                 Author: Richard Caley (rjc@cstr.ed.ac.uk)             */
 /*                   Date: Fri Oct 10 1997                               */
 /* --------------------------------------------------------------------  */
 /* A subclass of TVector which copies using memcopy. This isn't          */
 /* suitable for matrices of class objects which have to be copied        */
 /* using a constructor or specialised assignment operator.               */
 /*                                                                       */
 /*************************************************************************/

#include "EST_TSimpleVector.h"
#include "EST_matrix_support.h"
#include <fstream>
#include <cstring>
#include "EST_cutils.h"

using std::memset;
using std::memcpy;

template<class T> void EST_TSimpleVector<T>::copy(const EST_TSimpleVector<T> &a)
{
  if (this->p_column_step==1 && a.p_column_step==1)
    {
    resize(a.n(), FALSE);
    memcpy((void *)(this->p_memory), (const void *)(a.p_memory), this->n() * sizeof(T));
    }
else
  ((EST_TVector<T> *)this)->copy(a);
}

template<class T> EST_TSimpleVector<T>::EST_TSimpleVector(const EST_TSimpleVector<T> &in)
{
    this->default_vals();
    copy(in);
}

// should copy from and delete old version first
template<class T> void EST_TSimpleVector<T>::resize(int newn, int set)
{
  int oldn = this->n();
  T *old_vals =NULL;
  int old_offset = this->p_offset;
  unsigned int q;

  this->just_resize(newn, &old_vals);

  if (set && old_vals)
    {
      int copy_c = 0;
      if (this->p_memory != NULL)
	{
	  copy_c = Lof(this->n(), oldn);
          for (q=0; q<copy_c* sizeof(T); q++) /* for memcpy */
              ((char *)this->p_memory)[q] = ((char *)old_vals)[q];
	}
      
      for (int i=copy_c; i < this->n(); ++i)
	this->p_memory[i] = *this->def_val;
    }
  
  if (old_vals != NULL && old_vals != this->p_memory && !this->p_sub_matrix)
    delete [] (old_vals - old_offset);

}

template<class T>
void EST_TSimpleVector<T>::copy_section(T* dest, int offset, int num) const
{
  unsigned int q;
  if (num<0)
    num = this->num_columns()-offset;

  if (!EST_vector_bounds_check(num+offset-1, this->num_columns(), FALSE))
    return;

  if (!this->p_sub_matrix && this->p_column_step==1)
  {
      for (q=0; q<num* sizeof(T); q++)  /* for memcpy */
          ((char *)dest)[q] = ((char *)(this->p_memory+offset))[q];
  }
  else
    for(int i=0; i<num; i++)
      dest[i] = this->a_no_check(offset+i);
}

template<class T>
void EST_TSimpleVector<T>::set_section(const T* src, int offset, int num)
{
  unsigned int q;
  if (num<0)
    num = this->num_columns()-offset;

  if (!EST_vector_bounds_check(num+offset-1, this->num_columns(), FALSE))
    return;
  
  if (!this->p_sub_matrix && this->p_column_step==1)
  {
      for (q=0; q<num* sizeof(T); q++)  /* for memcpy */
          ((char *)(this->p_memory+offset))[q] = ((char *)(src))[q];
  }
  else
    for(int i=0; i<num; i++)
      this->a_no_check(offset+i) = src[i];
}

template<class T> EST_TSimpleVector<T> &EST_TSimpleVector<T>::operator=(const EST_TSimpleVector<T> &in)
{
    copy(in);
    return *this;
}

template<class T> void EST_TSimpleVector<T>::zero()
{
  if (this->p_column_step==1)
    memset((void *)(this->p_memory), 0, this->n() * sizeof(T));
  else
    ((EST_TVector<T> *)this)->fill(*this->def_val);
}


