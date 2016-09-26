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
/*                     Author :  Simon King                              */
/*                     Date   :  February 1999                           */
/* --------------------------------------------------------------------- */
/*          Double matrix class - copied from FMatrix !                  */
/*                                                                       */
/*************************************************************************/

#ifndef __DMatrix_H__
#define __DMatrix_H__

#include "EST_TSimpleMatrix.h"
#include "EST_TSimpleVector.h"
#include "EST_FMatrix.h"


class EST_DVector;

/** A matrix class for double precision floating point numbers. 
EST_DMatrix x should be used instead of double **x wherever
possible.*/
class EST_DMatrix : public EST_TSimpleMatrix<double> {
private:
public:
    /// size constructor
    EST_DMatrix(int m, int n):EST_TSimpleMatrix<double>(m, n)  {}
    /// copy constructor
    EST_DMatrix(const EST_DMatrix &a):EST_TSimpleMatrix<double>(a)  {}

    static EST_String default_file_type;
    /// CHECK  - what does this do???
    EST_DMatrix(const EST_DMatrix &a, int b);
    /// default constructor
    EST_DMatrix():EST_TSimpleMatrix<double>()  {}

    /// Save in file (ascii or binary)
    EST_write_status save(const EST_String &filename,
			  const EST_String &type = 
			        EST_DMatrix::default_file_type);
    /// Load from file (ascii or binary as defined in file)
    EST_read_status load(const EST_String &filename);
    /// Save in file in est format
    EST_write_status est_save(const EST_String &filename,
			      const EST_String &type);
    /// Load from file in est format (binary/ascii defined in file itself)
    EST_read_status est_load(const EST_String &filename);

    /// Copy 2-d array {\tt x} of size {\tt rows x cols} into matrix.
    void copyin(double **x, int rows, int cols);

    /// Add elements of 2 same sized matrices.
    EST_DMatrix &operator+=(const EST_DMatrix &a);

    /// Subtract elements of 2 same sized matrices.
    EST_DMatrix &operator-=(const EST_DMatrix &a);

    /// elementwise multiply by scalar
    EST_DMatrix &operator*=(const double f); 

    /// elementwise divide by scalar
    EST_DMatrix &operator/=(const double f); 

    /// Multiply all elements of matrix by {\tt x}.
    friend EST_DMatrix operator*(const EST_DMatrix &a, const double x);

    /// Multiply matrix by vector.
    friend EST_DVector operator*(const EST_DMatrix &a, const EST_DVector &v);

    /// Multiply vector by matrix
    friend EST_DVector operator*(const EST_DVector &v,const EST_DMatrix &a);

    /// Multiply matrix by matrix.
    friend EST_DMatrix operator*(const EST_DMatrix &a, const EST_DMatrix &b);
};


/** A vector class for double precision floating point
    numbers. {\tt EST_DVector x} should be used instead of 
    {\tt float *x} wherever possible.
*/
class EST_DVector: public EST_TSimpleVector<double> {
public:
    /// Size constructor.
    EST_DVector(int n): EST_TSimpleVector<double>(n) {}
    /// Copy constructor.
    EST_DVector(const EST_DVector &a): EST_TSimpleVector<double>(a) {}
    /// Default constructor.
    EST_DVector(): EST_TSimpleVector<double>() {}

    /// elementwise multiply
    EST_DVector &operator*=(const EST_DVector &s); 

    /// elementwise add
    EST_DVector &operator+=(const EST_DVector &s); 

    /// elementwise multiply by scalar
    EST_DVector &operator*=(const double d); 

    /// elementwise divide by scalar
    EST_DVector &operator/=(const double d); 

    EST_write_status est_save(const EST_String &filename,
			      const EST_String &type);

    /// save vector to file <tt> filename</tt>.
    EST_write_status save(const EST_String &filename,
			  const EST_String &type);

    /// load vector from file <tt> filename</tt>.
    EST_read_status load(const EST_String &filename);
    /// Load from file in est format (binary/ascii defined in file itself)
    EST_read_status est_load(const EST_String &filename);
};

int square(const EST_DMatrix &a);
/// inverse
int inverse(const EST_DMatrix &a, EST_DMatrix &inv);
int inverse(const EST_DMatrix &a, EST_DMatrix &inv, int &singularity);
/// pseudo inverse (for non-square matrices)
int pseudo_inverse(const EST_DMatrix &a, EST_DMatrix &inv); 
int pseudo_inverse(const EST_DMatrix &a, EST_DMatrix &inv,int &singularity); 

/// some useful matrix creators
/// make an identity matrix of dimension n
void eye(EST_DMatrix &a, const int n);
/// make already square matrix into I without resizing
void eye(EST_DMatrix &a);

/// the user should use est_seed to seed the random number generator
void est_seed();
void est_seed48();
/// all elements are randomised
void make_random_vector(EST_DVector &M, const double scale);
/// all elements are randomised
void make_random_matrix(EST_DMatrix &M, const double scale);
/// used for variance
void make_random_diagonal_matrix(EST_DMatrix &M, const double scale);
/// used for covariance
void make_random_symmetric_matrix(EST_DMatrix &M, const double scale);

void make_poly_basis_function(EST_DMatrix &T, EST_DVector t);

/// elementwise add
EST_DVector add(const EST_DVector &a,const EST_DVector &b);
/// elementwise subtract
EST_DVector subtract(const EST_DVector &a,const EST_DVector &b);

/// enforce symmetry
void symmetrize(EST_DMatrix &a);
/// stack columns on top of each other to make a vector
void stack_matrix(const EST_DMatrix &M, EST_DVector &v);
/// inplace diagonalise
void inplace_diagonalise(EST_DMatrix &a);


double determinant(const EST_DMatrix &a);
/// not implemented ??
int singular(EST_DMatrix &a);
/// exchange rows and columns
void transpose(const EST_DMatrix &a,EST_DMatrix &b);
EST_DMatrix triangulate(const EST_DMatrix &a);

/// extract leading diagonal as a matrix
EST_DMatrix diagonalise(const EST_DMatrix &a);
/// extract leading diagonal as a vector
EST_DVector diagonal(const EST_DMatrix &a);
/// sum of elements
double sum(const EST_DMatrix &a);
void multiply(const EST_DMatrix &a, const EST_DMatrix &b, EST_DMatrix &c);
int  floor_matrix(EST_DMatrix &M, const double floor);

/// matrix product of two vectors (#rows = length of first vector, #cols = length of second vector)
EST_DMatrix cov_prod(const EST_DVector &v1,const EST_DVector &v2);

EST_DMatrix operator*(const EST_DMatrix &a, const EST_DMatrix &b);
EST_DMatrix operator-(const EST_DMatrix &a, const EST_DMatrix &b);
EST_DMatrix operator+(const EST_DMatrix &a, const EST_DMatrix &b);

EST_DVector operator-(const EST_DVector &a, const EST_DVector &b);
EST_DVector operator+(const EST_DVector &a, const EST_DVector &b);

EST_DMatrix sub(const EST_DMatrix &a, int row, int col);
EST_DMatrix DMatrix_abs(const EST_DMatrix &a);

EST_DMatrix row(const EST_DMatrix &a, int row);
EST_DMatrix column(const EST_DMatrix &a, int col);


/// least squares fit
bool
polynomial_fit(EST_DVector &x, EST_DVector &y, EST_DVector &co_effs, int order);

/// weighted least squares fit
bool
polynomial_fit(EST_DVector &x, EST_DVector &y, EST_DVector &co_effs, 
	       EST_DVector &weights, int order);

double
polynomial_value(const EST_DVector &coeffs, const double x);

/// vector dot product
double operator*(const EST_DVector &v1, const EST_DVector &v2);


#endif
