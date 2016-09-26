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
 /*                     Author :  Paul Taylor                             */
 /*                     Date   :  April 1995                              */
 /* --------------------------------------------------------------------- */
 /*                     Template Vector Class                             */
 /*                                                                       */
 /*************************************************************************/


#include <iostream>
#include <fstream>
#include "EST_TVector.h"
#include "EST_matrix_support.h"
#include "EST_cutils.h"
#include "EST_error.h"

template<class T>
void EST_TVector<T>::default_vals()
{
  p_num_columns = 0;
  p_offset=0;
  p_column_step=0;

  p_memory = NULL;
  p_sub_matrix=FALSE;
}

template<class T>
EST_TVector<T>::EST_TVector()
{
    default_vals();
}

template<class T>
EST_TVector<T>::EST_TVector(int n)
{
    default_vals();
    resize(n);
}

template<class T>
EST_TVector<T>::EST_TVector(const EST_TVector<T> &in)
{
  default_vals();
  copy(in);
}

template<class T>
EST_TVector<T>::EST_TVector(int n,
			    T *memory, int offset, int free_when_destroyed)
{
  default_vals();

  set_memory(memory, offset, n, free_when_destroyed);
}

template<class T>
EST_TVector<T>::~EST_TVector()
{
  p_num_columns = 0;
  p_offset=0;
  p_column_step=0;

  if (p_memory != NULL && !p_sub_matrix)
    {
      delete [] (p_memory-p_offset);
      p_memory = NULL;
    }
}


template<class T>
void  EST_TVector<T>::fill(const T &v)
{
    for (int i = 0; i < num_columns(); ++i)
	fast_a_v(i) = v;
}

template<class T>
void EST_TVector<T>::set_memory(T *buffer, int offset, int columns, 
				int free_when_destroyed)
{
  if (p_memory != NULL && !p_sub_matrix)
    delete [] (p_memory-p_offset);
  
  p_memory = buffer-offset;
  p_offset=offset;
  p_num_columns = columns;
  p_column_step=1;
  p_sub_matrix = !free_when_destroyed;
}

template<class T>
void EST_TVector<T>::set_values(const T *data, 
				 int step,
				 int start_c, 
				 int num_c)
{
  for(int i=0, c=start_c, p=0; i<num_c; i++, c++, p+=step)
    a_no_check(c) = data[p];
}


template<class T>
void EST_TVector<T>::get_values(T *data, 
				 int step,
				 int start_c, 
				 int num_c) const
{
  for(int i=0, c=start_c, p=0; i<num_c; i++, c++, p+=step)
   data[p] = a_no_check(c);
}


template<class T>
void EST_TVector<T>::copy_data(const EST_TVector<T> &a)
{
  set_values(a.p_memory, a.p_column_step, 0, num_columns());
}

template<class T>
void EST_TVector<T>::copy(const EST_TVector<T> &a)
{
    resize(a.n(), FALSE);
    copy_data(a);
}

template<class T>
void EST_TVector<T>::just_resize(int new_cols, T** old_vals)
{
  T *new_m;
  
  if (num_columns() != new_cols || p_memory == NULL )
    {
      if (p_sub_matrix)
	EST_error("Attempt to resize Sub-Vector");

      if (new_cols < 0)
	EST_error("Attempt to resize vector to negative size: %d",
		  new_cols);

      new_m = new T[new_cols];

      if (p_memory != NULL)
      {
	if (old_vals != NULL)
	  *old_vals = p_memory;
	else if (!p_sub_matrix)
	  delete [] (p_memory-p_offset);
      }

      p_memory = new_m;
      //cout << "vr: mem: " << p_memory << " (" << (int)p_memory << ")\n";
      p_offset=0;
      p_num_columns = new_cols;
      p_column_step=1;
    }
  else
    *old_vals = p_memory;
}


template<class T>
void EST_TVector<T>::resize(int new_cols, int set)
{
  int i;
  T * old_vals = p_memory;
  int old_cols = num_columns();
  int old_offset = p_offset;
  int old_column_step = p_column_step;

  just_resize(new_cols, &old_vals);

  if (set)
    {
      int copy_c = 0;

      if (!old_vals)
	copy_c=0;
      else if (old_vals != p_memory)
	{
	  copy_c = Lof(num_columns(), old_cols);

	  for(i=0; i<copy_c; i++)
	      a_no_check(i) 
		= old_vals[vcell_pos(i,
				    old_column_step)];
	}
      else 
	copy_c = old_cols;
      
      for(i=copy_c; i<new_cols; i++)
	  a_no_check(i) =  *def_val;
    }

  if (old_vals && old_vals != p_memory && !p_sub_matrix)
    delete [] (old_vals-old_offset);
}

template<class T>
EST_TVector<T> &EST_TVector<T>::operator=(const EST_TVector<T> &in)
{
    copy(in);
    return *this;
}

template<class T>
T &EST_TVector<T>::a_check(int n)
{
  if (!EST_vector_bounds_check(n, num_columns(), FALSE))
    return *error_return;

  return fast_a_v(n);
}

template<class T>
const T &EST_TVector<T>::a_check(int n) const
{
  return ((EST_TVector<T> *)this)->a(n);
}

template<class T>
int EST_TVector<T>::operator == (const EST_TVector<T> &v) const
{
    if (num_columns() != v.num_columns())
	return 0;

    for(int i=0; i<num_columns() ; i++)
    {
      if (fast_a_v(i) == v.fast_a_v(i))
          continue;
      else
          return 0;
    }
    return 1;
}

template<class T>
void EST_TVector<T>::copy_section(T* dest, int offset, int num) const
{
  if (num<0)
    num = num_columns()-offset;

  if (!EST_vector_bounds_check(num+offset-1, num_columns(), FALSE))
    return;
  

  for(int i=0; i<num; i++)
    dest[i] = a_no_check(offset+i);
}

template<class T>
void EST_TVector<T>::set_section(const T* src, int offset, int num)
{
  if (num<0)
    num = num_columns()-offset;

  if (!EST_vector_bounds_check(num+offset-1, num_columns(), FALSE))
    return;
  
  for(int i=0; i<num; i++)
    a_no_check(offset+i) = src[i];
}

template<class T>
void EST_TVector<T>::sub_vector(EST_TVector<T> &sv,
				int start_c, int len)
{
  if (len < 0)
    len = num_columns()-start_c;

  if (sv.p_memory != NULL && ! sv.p_sub_matrix)
    delete [] (sv.p_memory - sv.p_offset);

  sv.p_sub_matrix = TRUE;
  sv.p_offset = p_offset + start_c*p_column_step;
  sv.p_memory = p_memory - p_offset + sv.p_offset;
  sv.p_column_step=p_column_step;
  sv.p_num_columns = len;
}

template<class T>
void EST_TVector<T>::integrity() const
{
  cout << "integrity: p_memory=" << p_memory << endl;
  if(p_memory == (T *)0x00080102) 
    {
      cout << "fatal value!!!\n";
    }
}


