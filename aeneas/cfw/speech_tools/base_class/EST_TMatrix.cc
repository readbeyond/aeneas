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
 /*                      Author :  Paul Taylor                            */
 /*                   Rewritten :  Richard Caley                          */
 /* -------------------------------------------------------------------   */
 /*                      Template EST_TMatrix Class                       */
 /*                                                                       */
 /*************************************************************************/

#include "EST_TMatrix.h"
#include <fstream>
#include <iostream>
#include "EST_bool.h"
#include "EST_matrix_support.h"
#include "EST_TVector.h"
#include "EST_cutils.h"
#include "EST_error.h"

/* Construction and destruction
 */

template<class T>
void EST_TMatrix<T>::default_vals()
{
  EST_TVector<T>::default_vals();
  p_num_rows = 0;
  p_row_step=0;
}

template<class T>
EST_TMatrix<T>::EST_TMatrix()
{
  default_vals();
}

template<class T>
EST_TMatrix<T>::EST_TMatrix(const EST_TMatrix<T> &in)
{
  default_vals();
  copy(in);
}

template<class T>
EST_TMatrix<T>::EST_TMatrix(int rows, int cols)
{
  default_vals();
  resize(rows, cols);
}

template<class T>
EST_TMatrix<T>::EST_TMatrix(int rows, int cols, 
			    T *memory, int offset, int free_when_destroyed)
{
  default_vals();
  set_memory(memory, offset, rows, cols, free_when_destroyed);
}

template<class T>
EST_TMatrix<T>::~EST_TMatrix()
{
    p_num_rows = 0;
    p_row_step=0;
}

/* Basic access
 */

template<class T>
T &EST_TMatrix<T>::a_check(int row, int col) 
{

  if (!EST_matrix_bounds_check(row, col, num_rows(), num_columns(), FALSE))
    return *this->error_return;
    
  return a_no_check(row,col);
}

/* Since we know a() itself doesn't change the matrix, we can cast away
 * the const here. Isn't the C++ syntax beautiful!
 */
template<class T>
const T &EST_TMatrix<T>::a_check(int row, int col) const
{
  return ((EST_TMatrix<T> *)this)->a(row,col);
}

template<class T>
void EST_TMatrix<T>::copy_data(const EST_TMatrix<T> &a)
{

  set_values(a.p_memory,
	     a.p_row_step, a.p_column_step,
	     0, a.num_rows(),
	     0, a.num_columns());
}

template<class T>
void EST_TMatrix<T>::set_values(const T *data, 
				int r_step, int c_step,
				int start_r, int num_r,
				int start_c, int num_c
				)
{
  for(int r=start_r, i=0, rp=0; i< num_r; i++, r++, rp+=r_step)
    for(int c=start_c, j=0, cp=0; j< num_c; j++, c++, cp+=c_step)
      a_no_check(r,c) = data[rp+cp];
}

template<class T>
void EST_TMatrix<T>::get_values(T *data, 
				int r_step, int c_step,
				int start_r, int num_r,
				int start_c, int num_c
				) const
{
  for(int r=start_r, i=0, rp=0; i< num_r; i++, r++, rp+=r_step)
    for(int c=start_c, j=0, cp=0; j< num_c; j++, c++, cp+=c_step)
      data[rp+cp] = a_no_check(r,c);
}

template<class T>
void EST_TMatrix<T>::copy(const EST_TMatrix<T> &a)
{
  resize(a.num_rows(), a.num_columns(), 0);
  copy_data(a);
}

template<class T>
EST_TMatrix<T> &EST_TMatrix<T>::operator=(const EST_TMatrix<T> &in)
{
    copy(in);
    return *this;
}

template<class T>
EST_TMatrix<T> &EST_TMatrix<T>::add_rows(const EST_TMatrix<T> &in)
{
  if (in.num_columns() != num_columns())
    EST_error("Can't add rows with differnet number of columns (%d vs %d)",
	      in.num_columns(),
	      num_columns()
	      );
  else
    {
      int old_num_rows = num_rows();
      resize(num_rows()+in.num_rows(), num_columns(), TRUE);

      for(int i=old_num_rows, i1=0; i<num_rows(); i++, i1++)
	for(int j=0; j<num_columns(); j++)
	  a(i,j) = in.a(i1,j);
      
    }
  return *this;
}

template<class T>
EST_TMatrix<T> &EST_TMatrix<T>::add_columns(const EST_TMatrix<T> &in)
{
  if (in.num_rows() != num_rows())
    EST_error("Can't add columns with differnet number of rows (%d vs %d)",
	      in.num_rows(),
	      num_rows()
	      );
  else
    {
      int old_num_columns = num_columns();
      resize(num_columns()+in.num_columns(), num_rows(), TRUE);

      for(int i=old_num_columns, i1=0; i<num_columns(); i++, i1++)
	for(int j=0; j<num_rows(); j++)
	  a(i,j) = in.a(i1,j);
      
    }
  return *this;
}

template<class T>
void EST_TMatrix<T>::just_resize(int new_rows, 
				 int new_cols, 
				 T** old_vals)
{
    T *new_m;

    if (num_rows() != new_rows || num_columns() != new_cols || this->p_memory == NULL )
      {
	if (this->p_sub_matrix)
	  EST_error("Attempt to resize Sub-Matrix");

	if (new_cols < 0 || new_rows < 0)
	  EST_error("Attempt to resize matrix to negative size: %d x %d",
		    new_rows,
		    new_cols);

	
	new_m = new T[new_rows*new_cols];

	if (this->p_memory != NULL)
        {
	  if (old_vals != NULL)
	    *old_vals = this->p_memory;
	  else  if (!this->p_sub_matrix)
	    delete [] (this->p_memory-this->p_offset);
        }
    
	p_num_rows = new_rows;
	this->p_num_columns = new_cols;
	this->p_offset=0;
	p_row_step=this->p_num_columns; 
	this->p_column_step=1;
	
	this->p_memory = new_m;
      }
    else
      *old_vals = this->p_memory;
	
}

template<class T>
void EST_TMatrix<T>::resize(int new_rows, int new_cols, int set)
{
  int i,j;
  T * old_vals = this->p_memory;
  int old_rows = num_rows();
  int old_cols = num_columns();
  int old_row_step = p_row_step;
  int old_offset = this->p_offset;
  int old_column_step = this->p_column_step;

  if (new_rows<0)
    new_rows = old_rows;
  if (new_cols<0)
    new_cols = old_cols;

  just_resize(new_rows, new_cols, &old_vals);

  if (set)
    {
      int copy_r = 0;
      int copy_c = 0;

      if (old_vals != NULL)
	{
	  copy_r = Lof(num_rows(), old_rows);
	  copy_c = Lof(num_columns(), old_cols);

	  set_values(old_vals,
		     old_row_step, old_column_step,
		     0, copy_r,
		     0, copy_c);
	}
      else
	{
	  copy_r = old_rows;
	  copy_c = old_cols;
	}
      
      for(i=0; i<copy_r; i++)
	for(j=copy_c; j<new_cols; j++)
	  a_no_check(i,j) =  *this->def_val;
      
      for(i=copy_r; i<new_rows; i++)
	for(j=0; j<new_cols; j++)
	  a_no_check(i,j) =  *this->def_val;
    }

  if (old_vals && old_vals != this->p_memory && !this->p_sub_matrix)
    delete [] (old_vals-old_offset);
}

template<class T>
bool EST_TMatrix<T>::have_rows_before(int n) const
{
  return this->p_offset >= n*p_row_step;
}

template<class T>
bool EST_TMatrix<T>::have_columns_before(int n) const
{
  return this->p_offset >= n*this->p_column_step;
}

template<class T>
void EST_TMatrix<T>::fill(const T &v)
{
    int i, j;
    for (i = 0; i < num_rows(); ++i)
	for (j = 0; j < num_columns(); ++j)
	    fast_a_m(i,j) = v;
}


template<class T>
EST_write_status EST_TMatrix<T>::save(const EST_String &filename) const
{
    int i, j;
    ostream *outf;
    if (filename == "-" || filename == "")
	outf = &cout;
    else
      outf = new ofstream(filename);

    for (i = 0; i < num_rows(); ++i)
    {
      for (j = 0; j < num_columns(); ++j)
	{
	  *outf 
#if 0
	    << "{" <<i<<","<<j
		<<",m="<<((int)this->p_memory)<<","
		<<"r'="<<((int)((T *) mx_move_pointer_f(this->p_memory, sizeof(T)*p_row_step, i)))<<","
		<<"r="<<((int)mx_move_pointer(this->p_memory, T, p_row_step, i))<<","
		<<"c="<<((int)mx_move_pointer(this->p_memory, T, this->p_column_step, j))<<","
		<<((int)(&fast_a_m_gcc(i,j)))
		<<"}"
#endif
		<< a_no_check(i,j) << "\t";
	}
      *outf << endl;
    }
    
    if (outf != &cout)
	delete outf;

    return write_ok;
}

template<class T>
EST_read_status
EST_TMatrix<T>::load(const EST_String &filename)
{
    // this function can only be written if we can find a way of parsing
    // an unknown type;
    (void) filename;
    EST_error("Matrix loading not implemented yet.");
    return misc_read_error;

}

template<class T>
void EST_TMatrix<T>::set_memory(T *buffer, int offset, 
				int rows, int columns, 
				int free_when_destroyed)
{
  EST_TVector<T>::set_memory(buffer, offset, columns, free_when_destroyed);
  p_num_rows = rows;
  p_row_step = columns;
}

template<class T>
void EST_TMatrix<T>::copy_row(int r, T *buf, 
			      int offset, int num) const
{
    int to = num >= 0 ? offset + num : num_columns();

    if (!EST_matrix_bounds_check(r, 0, num_rows(), num_columns(), FALSE))
    {
      if (num_rows()>0)
	r=0;
      else
	return;
    }

    for (int j = offset; j < to; j++)
      buf[j-offset] = fast_a_m(r, j);
}

template<class T>
void EST_TMatrix<T>::copy_row(int r, EST_TVector<T> &buf,
                              int offset, int num) const
{
  int to = num >= 0 ? offset + num : num_columns();

  if (!EST_matrix_bounds_check(r, 0, num_rows(), num_columns(), FALSE))
  {
    if (num_rows()>0)
      r=0;
    else
      return;
  }
  
  buf.resize(to - offset);
  
  for (int j = offset; j < to; j++)
    buf[j - offset] = fast_a_m(r, j);
}


template<class T>
void EST_TMatrix<T>::copy_column(int c, T *buf, 
				 int offset, int num) const
{
  if (num_rows() == 0) 
    return;

  int to = num >= 0 ? offset + num : num_rows();

  if (!EST_matrix_bounds_check(0, c, num_rows(), num_columns(), FALSE))
  {
    if (num_columns()>0)
      c=0;
    else
      return;
  }
  
  for (int i = offset; i < to; i++)
    buf[i-offset] = fast_a_m(i, c);
}


template<class T>
void EST_TMatrix<T>::copy_column(int c, EST_TVector<T> &buf,
				 int offset, int num) const
{
  if (num_rows() == 0) 
    return;

  int to = num >= 0 ? offset + num : num_rows();

  if (!EST_matrix_bounds_check(0, c, num_rows(), num_columns(), FALSE))
  {
    if( num_columns()>0 )
      c=0;
    else
      return;
  }
  
  buf.resize(to - offset);
  
  for (int i = offset; i < to; i++)
    buf[i-offset] = fast_a_m(i, c);
}


template<class T>
void EST_TMatrix<T>::set_row(int r, const T *buf, int offset, int num)
{
    int to = num>=0?offset+num:num_columns();

    if (!EST_matrix_bounds_check(r, 0, num_rows(), num_columns(), TRUE))
      return;

    for(int j=offset; j<to; j++)
      fast_a_m(r, j) = buf[j-offset];
}

template<class T>
void EST_TMatrix<T>::set_column(int c, const T *buf, int offset, int num)
{
    int to = num>=0?offset+num:num_rows();

    if (!EST_matrix_bounds_check(0, c, num_rows(), num_columns(), TRUE))
      return;

    for(int i=offset; i<to; i++)
      fast_a_m(i, c) = buf[i-offset];
}

template<class T>
void  EST_TMatrix<T>::set_row(int r, 
	     const EST_TMatrix<T> &from, int from_r, int from_offset, 
	     int offset, int num)
{
  int to = num>=0?offset+num:num_columns();

  if (!EST_matrix_bounds_check(r, 0, num_rows(), num_columns(), TRUE))
    return;

  if (!EST_matrix_bounds_check(from_r, 0, from.num_rows(), from.num_columns(), FALSE))
  {
    if (from.num_rows()>0)
      from_r=0;
    else 
      return;
  }

  for(int j=offset; j<to; j++)
    fast_a_m(r, j) = from.fast_a_m(from_r, (j-offset)+from_offset);
}

template<class T>
void  EST_TMatrix<T>::set_column(int c, 
	     const EST_TMatrix<T> &from, int from_c, int from_offset, 
	     int offset, int num)
{
  int to = num>=0?offset+num:num_rows();

  if (!EST_matrix_bounds_check(0, c, num_rows(), num_columns(), TRUE))
    return;

  if (!EST_matrix_bounds_check(0, from_c, from.num_rows(), from.num_columns(), FALSE))
  {
    if (from.num_columns()>0)
      from_c=0;
    else 
      return;
  }

  for(int i=offset; i<to; i++)
    fast_a_m(i, c) = from.fast_a_m((i-offset)+from_offset, from_c);
}

template<class T>
void EST_TMatrix<T>::row(EST_TVector<T> &rv, int r, int start_c, int len)
{
  if (len < 0)
    len = num_columns()-start_c;

  if (!EST_matrix_bounds_check(r, 1, start_c, len, num_rows(), num_columns(), 0))
    return;

  if (rv.p_memory != NULL && ! rv.p_sub_matrix)
    delete [] (rv.p_memory - rv.p_offset);

  rv.p_sub_matrix = TRUE;
  rv.p_num_columns = len;
  rv.p_offset = this->p_offset + start_c*this->p_column_step + r*p_row_step;
  rv.p_memory = this->p_memory - this->p_offset + rv.p_offset;
//  cout << "mrow: mem: " << rv.p_memory << " (" << (int)rv.p_memory << ")\n";
//  cout << "mrow: ofset: " << rv.p_offset << " (" << (int)rv.p_offset << ")\n";

  rv.p_column_step=this->p_column_step;
}

template<class T>
void EST_TMatrix<T>::column(EST_TVector<T> &cv, int c, int start_r, int len)
{
  if (len < 0)
    len = num_rows()-start_r;

  if (!EST_matrix_bounds_check(start_r, len, c, 1,num_rows(), num_columns(), 0))
    return;
  
  if (cv.p_memory != NULL && ! cv.p_sub_matrix)
    delete [] (cv.p_memory - cv.p_offset);

  cv.p_sub_matrix = TRUE;
  cv.p_num_columns = len;
  cv.p_offset = this->p_offset + c*this->p_column_step + start_r*p_row_step;
  cv.p_memory = this->p_memory - this->p_offset + cv.p_offset;
//  cout << "mcol: mem: " << cv.p_memory << " (" << (int)cv.p_memory << ")\n";
//  cout << "mcol: offset: " << cv.p_offset << " (" << (int)cv.p_offset << ")\n";

  cv.p_column_step=p_row_step;
}

template<class T>
void EST_TMatrix<T>::sub_matrix(EST_TMatrix<T> &sm, 
				int r, int len_r, int c, int len_c)
{
  if (len_r < 0)
    len_r = num_rows()-r;
  if (len_c < 0)
    len_c = num_columns()-c;

  if (!EST_matrix_bounds_check(r, len_r, c, len_c, num_rows(), num_columns(), 0))
    return;
  
  if (sm.p_memory != NULL && ! sm.p_sub_matrix)
    delete [] (sm.p_memory - sm.p_offset);

  sm.p_sub_matrix = TRUE;
  sm.p_offset = this->p_offset + c*this->p_column_step + r*p_row_step;
  sm.p_memory = this->p_memory - this->p_offset + sm.p_offset;
  sm.p_row_step=p_row_step;
  sm.p_column_step=this->p_column_step;
  sm.p_num_rows = len_r;
  sm.p_num_columns = len_c;
  
}

