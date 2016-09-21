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
 /*                                                                       */
 /*                     Author :  Paul Taylor                             */
 /*                  Rewritten :  Richard Caley                           */
 /* --------------------------------------------------------------------- */
 /*                         Matrix class                                  */
 /*                                                                       */
 /*************************************************************************/

#ifndef __TMatrix_H__
#define __TMatrix_H__

#include <iostream>

using namespace std;

#include "EST_rw_status.h"
#include "EST_TVector.h"
#include "instantiate/EST_TMatrixI.h"

/* When set bounds checks (safe but slow) are done on matrix access */
#ifndef TMATRIX_BOUNDS_CHECKING
#    define TMATRIX_BOUNDS_CHECKING 0
#endif

#if TMATRIX_BOUNDS_CHECKING
#define A_CHECK a_check
#else
#define A_CHECK a_no_check
#endif

#define INLINE inline 

/* This doesn't work as I thought so I have disabled it for now.
 */

#if defined(__GNUC__) && 0
#    define mx_move_pointer(P, TY, STEP, N) \
            ((TY *)\
	     ((void *) (((char (*) [sizeof(TY)*STEP])P) + N) ) \
	      )
#    define fast_a_m_gcc(R,C) \
	( * mx_move_pointer(mx_move_pointer(p_memory,T,p_column_step,C),T,p_row_step,R))
#    define fast_a_m_x(R,C) (fast_a_m_gcc(R,C))
#else
#    define fast_a_m_x(R,C) (fast_a_m(R,C))
#endif



/** Template Matrix class.
  *
  * This is an extension of the EST_TVector class to two dimensions.
  *
  * @see matrix_example
  * @see EST_TVector
  */

template <class T> 
class EST_TMatrix : public EST_TVector<T>
{

protected:
  /// Visible shape
  unsigned int p_num_rows; 
  
  /// How to access the memory
  unsigned int p_row_step;

  INLINE unsigned int mcell_pos(int r, int c,
			       int rs, int cs) const
    { return (rs==1?r:(r*rs)) + (cs==1?c:(c*cs));}


  INLINE unsigned int mcell_pos(int r, int c) const
    {

      return mcell_pos(r, c, 
		       this->p_row_step, this->p_column_step);
    }

  INLINE unsigned int mcell_pos_1(int r, int c) const
    {

      (void)r;
      return c;
    }

  /// quick method for returning {\tt x[m][n]}
  INLINE const T &fast_a_m(int r, int c) const 
    { return this->p_memory[mcell_pos(r,c)]; }
  INLINE T &fast_a_m(int r, int c) 
    { return this->p_memory[mcell_pos(r,c)]; }

  INLINE const T &fast_a_1(int r, int c) const 
    { return this->p_memory[mcell_pos_1(r,c)]; }
  INLINE T &fast_a_1(int r, int c) 
    { return this->p_memory[mcell_pos_1(r,c)]; }
  

    /// Get and set values from array
  void set_values(const T *data, 
		  int r_step, int c_step,
		  int start_r, int num_r,
		  int start_c, int num_c
		  );
  void get_values(T *data, 
		  int r_step, int c_step,
		  int start_r, int num_r,
		  int start_c, int num_c
		  ) const;

  /// private resize and copy function. 
  void copy(const EST_TMatrix<T> &a);
  /// just copy data, no resizing, no size check.
  void copy_data(const EST_TMatrix<T> &a); 

  /// resize the memory and reset the bounds, but don't set values.
  void just_resize(int new_rows, int new_cols, T** old_vals);

  /// sets data and length to default values (0 in both cases).
  void default_vals();
public:

  ///default constructor
  EST_TMatrix(); 

  /// copy constructor
  EST_TMatrix(const EST_TMatrix<T> &m); 

  /// "size" constructor
  EST_TMatrix(int rows, int cols); 

  /// construct from memory supplied by caller
  EST_TMatrix(int rows, int cols, 
	      T *memory, int offset=0, int free_when_destroyed=0);

  /// EST_TMatrix

  ~EST_TMatrix();

  /**@name access
    * Basic access methods for matrices.
    */
  //@{

  /// return number of rows
  int num_rows() const {return this->p_num_rows;}
  /// return number of columns
  int num_columns() const {return this->p_num_columns;}

  /// const access with no bounds check, care recommend
  INLINE const T &a_no_check(int row, int col) const 
    { return fast_a_m_x(row,col); }
  /// access with no bounds check, care recommend
  INLINE T &a_no_check(int row, int col) 
    { return fast_a_m_x(row,col); }

  INLINE const T &a_no_check_1(int row, int col) const { return fast_a_1(row,col); }
  INLINE T &a_no_check_1(int row, int col) { return fast_a_1(row,col); }

  /// const element access function 
  const T &a_check(int row, int col) const;
  /// non-const element access function 
  T &a_check(int row, int col);

  const T &a(int row, int col) const { return A_CHECK(row,col); }
  T &a(int row, int col) { return A_CHECK(row,col); }

  /// const element access operator
  const T &operator () (int row, int col) const { return a(row,col); }
  /// non-const element access operator
  T &operator () (int row, int col) { return a(row,col); }
  
  //@}

  bool have_rows_before(int n) const;
  bool have_columns_before(int n) const;

  /** resize matrix. If {\tt set=1}, then the current values in
      the matrix are preserved up to the new size {\tt n}. If the
      new size exceeds the old size, the rest of the matrix is
      filled with the {\tt def_val}
  */
    void resize(int rows, int cols, int set=1); 

  /// fill matrix with value v
  void fill(const T &v);
  void fill() { fill(*this->def_val); }

  /// assignment operator
  EST_TMatrix &operator=(const EST_TMatrix &s); 

  /// The two versions of what might have been operator +=
  EST_TMatrix &add_rows(const EST_TMatrix &s); 
  EST_TMatrix &add_columns(const EST_TMatrix &s); 

  /**@name Sub-Matrix/Vector Extraction
    *
    * All of these return matrices and vectors which share
    * memory with the original, so altering values them alters
    * the original. 
    */
  //@{

  /// Make the vector {\tt rv} a window onto row {\tt r}
  void row(EST_TVector<T> &rv, int r, int start_c=0, int len=-1);
  /// Make the vector {\tt cv} a window onto column {\tt c}
  void column(EST_TVector<T> &cv, int c, int start_r=0, int len=-1);
  /// Make the matrix {\tt sm} a window into this matrix.
  void sub_matrix(EST_TMatrix<T> &sm,
		  int r=0, int numr=EST_ALL, 
		  int c=0, int numc=EST_ALL);
  //@}

  /**@name Copy in and out
    * Copy data between buffers and the matrix.
    */
  //@{
    /** Copy row {\tt r} of matrix to {\tt buf}. {\tt buf}
        should be pre-malloced to the correct size.
        */
    void copy_row(int r, T *buf, int offset=0, int num=-1) const;

    /** Copy row <parameter>r</parameter> of matrix to
        <parameter>buf</parameter>. <parameter>buf</parameter> should be
        pre-malloced to the correct size.  */
    
    void copy_row(int r, EST_TVector<T> &t, int offset=0, int num=-1) const;

    /** Copy column {\tt c} of matrix to {\tt buf}. {\tt buf}
        should be pre-malloced to the correct size.
        */
    void copy_column(int c, T *buf, int offset=0, int num=-1) const;

    /** Copy column <parameter>c</parameter> of matrix to
        <parameter>buf</parameter>. <parameter>buf</parameter> should
        be pre-malloced to the correct size.  */

    void copy_column(int c,  EST_TVector<T> &t, int offset=0, int num=-1)const;

    /** Copy buf into row {\tt n} of matrix. 
        */
    void set_row(int n, const T *buf, int offset=0, int num=-1);

    void set_row(int n, const EST_TVector<T> &t, int offset=0, int num=-1)
      { set_row(n, t.memory(), offset, num); }

    void set_row(int r, 
                 const EST_TMatrix<T> &from, int from_r, int from_offset=0,
                 int offset=0, int num=-1); // set nth row


    /** Copy buf into column {\tt n} of matrix.         
      */
    void set_column(int n, const T *buf, int offset=0, int num=-1);

    void set_column(int n, const EST_TVector<T> &t, int offset=0, int num=-1)
      { set_column(n, t.memory(), offset, num); }
    
    void set_column(int c, 
                    const EST_TMatrix<T> &from, int from_c, int from_offset=0, 
                    int offset=0, int num=-1); // set nth column

  /** For when you absolutely have to have access to the memory.
    */
  void set_memory(T *buffer, int offset, int rows, int columns, 
		  int free_when_destroyed=0);

  //@}

  /**@name io
    * Matrix file io.
    */
  //@{
  /// load Matrix from file - Not currently implemented.
  EST_read_status  load(const class EST_String &filename);
  /// save Matrix to file {\tt filename}
  EST_write_status save(const class EST_String &filename) const;

  /// print matrix.
  friend ostream& operator << (ostream &st,const EST_TMatrix<T> &a)
    {int i, j; 
        for (i = 0; i < a.num_rows(); ++i) {
            for (j = 0; j < a.num_columns(); ++j) 
                st << a.a_no_check(i, j) << " "; st << endl;
        }
        return st;
    }
  //@}
  
};

#undef A_CHECK

#endif

