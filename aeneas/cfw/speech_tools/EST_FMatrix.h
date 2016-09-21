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
 /*                     Date   :  April 1996                              */
 /* --------------------------------------------------------------------- */
 /*                        Matrix class                                   */
 /*                                                                       */
 /*************************************************************************/

#ifndef __FMatrix_H__
#define __FMatrix_H__

#include "EST_TSimpleMatrix.h"
#include "EST_TSimpleVector.h"

#include "EST_Val.h"
#include "EST_Val_defs.h"

class EST_FVector;

/** A matrix class for floating point numbers. EST_FMatrix x should be
 used instead of float **x wherever possible.  
*/

class EST_FMatrix : public EST_TSimpleMatrix<float> {
private:
public:
    /// size constructor
    EST_FMatrix(int m, int n):EST_TSimpleMatrix<float>(m, n)  {}
    /// copy constructor
    EST_FMatrix(const EST_FMatrix &a):EST_TSimpleMatrix<float>(a)  {}

    static EST_String default_file_type;
    /// CHECK  - what does this do???
    EST_FMatrix(const EST_FMatrix &a, int b);
    /// default constructor
    EST_FMatrix():EST_TSimpleMatrix<float>()  {}

    /// Save in file (ascii or binary)
    EST_write_status save(const EST_String &filename,
			  const EST_String &type = 
			        EST_FMatrix::default_file_type );
    /// Load from file (ascii or binary as defined in file)
    EST_read_status load(const EST_String &filename);
    /// Save in file in est format
    EST_write_status est_save(const EST_String &filename,
			      const EST_String &type);
    /// Load from file in est format (binary/ascii defined in file itself)
    EST_read_status est_load(const EST_String &filename);

    /// Copy 2-d array {\tt x} of size {\tt rows x cols} into matrix.
    void copyin(float **x, int rows, int cols);

    /// Add elements of 2 same sized matrices.
    EST_FMatrix &operator+=(const EST_FMatrix &a);

    /// Subtract elements of 2 same sized matrices.
    EST_FMatrix &operator-=(const EST_FMatrix &a);

    /// elementwise multiply by scalar
    EST_FMatrix &operator*=(const float f); 

    /// elementwise divide by scalar
    EST_FMatrix &operator/=(const float f); 

    /// Multiply all elements of matrix by {\tt x}.
    friend EST_FMatrix operator*(const EST_FMatrix &a, const float x);

    /// Multiply matrix by vector.
    friend EST_FVector operator*(const EST_FMatrix &a, const EST_FVector &v);

    /// Multiply vector by matrix
    friend EST_FVector operator*(const EST_FVector &v,const EST_FMatrix &a);

    /// Multiply matrix by matrix.
    friend EST_FMatrix operator*(const EST_FMatrix &a, const EST_FMatrix &b);
};

/** A vector class for floating point numbers. 
    {\tt EST_FVector x} should be used instead of {\tt float *x}
    wherever possible.
*/
class EST_FVector: public EST_TSimpleVector<float> {
public:
    /// Size constructor.
    EST_FVector(int n): EST_TSimpleVector<float>(n) {}
    /// Copy constructor.
    EST_FVector(const EST_FVector &a): EST_TSimpleVector<float>(a) {}
    /// Default constructor.
    EST_FVector(): EST_TSimpleVector<float>() {}

    /// elementwise multiply
    EST_FVector &operator*=(const EST_FVector &s); 

    /// elementwise add
    EST_FVector &operator+=(const EST_FVector &s); 

    /// elementwise multiply by scalar
    EST_FVector &operator*=(const float f); 

    /// elementwise divide by scalar
    EST_FVector &operator/=(const float f); 

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

/// find largest element
float matrix_max(const EST_FMatrix &a);
/// find largest element
float vector_max(const EST_FVector &a);

int square(const EST_FMatrix &a);
/// inverse
int inverse(const EST_FMatrix &a, EST_FMatrix &inv);
int inverse(const EST_FMatrix &a, EST_FMatrix &inv, int &singularity);
/// pseudo inverse (for non-square matrices)
int pseudo_inverse(const EST_FMatrix &a, EST_FMatrix &inv); 
int pseudo_inverse(const EST_FMatrix &a, EST_FMatrix &inv,int &singularity); 

/// some useful matrix creators
/// make an identity matrix of dimension n
void eye(EST_FMatrix &a, const int n);
/// make already square matrix into I without resizing
void eye(EST_FMatrix &a);

/// the user should use est_seed to seed the random number generator
void est_seed();
void est_seed48();
/// all elements are randomised
void make_random_vector(EST_FVector &M, const float scale);
/// all elements are randomised
void make_random_matrix(EST_FMatrix &M, const float scale);
/// used for variance
void make_random_diagonal_matrix(EST_FMatrix &M, const float scale);
/// used for covariance
void make_random_symmetric_matrix(EST_FMatrix &M, const float scale);

void make_poly_basis_function(EST_FMatrix &T, EST_FVector t);

/// elementwise add
EST_FVector add(const EST_FVector &a,const EST_FVector &b);
/// elementwise subtract
EST_FVector subtract(const EST_FVector &a,const EST_FVector &b);

/// enforce symmetry
void symmetrize(EST_FMatrix &a);
/// stack columns on top of each other to make a vector
void stack_matrix(const EST_FMatrix &M, EST_FVector &v);
/// inplace diagonalise
void inplace_diagonalise(EST_FMatrix &a);


float determinant(const EST_FMatrix &a);
/// not implemented ??
int singular(EST_FMatrix &a);
/// exchange rows and columns
void transpose(const EST_FMatrix &a,EST_FMatrix &b);
EST_FMatrix triangulate(const EST_FMatrix &a);

/// extract leading diagonal as a matrix
EST_FMatrix diagonalise(const EST_FMatrix &a);
/// extract leading diagonal as a vector
EST_FVector diagonal(const EST_FMatrix &a);
/// sum of elements
float sum(const EST_FMatrix &a);
void multiply(const EST_FMatrix &a, const EST_FMatrix &b, EST_FMatrix &c);
int  floor_matrix(EST_FMatrix &M, const float floor);

/// matrix product of two vectors (#rows = length of first vector, #cols = length of second vector)
EST_FMatrix cov_prod(const EST_FVector &v1,const EST_FVector &v2);

EST_FMatrix operator*(const EST_FMatrix &a, const EST_FMatrix &b);
EST_FMatrix operator-(const EST_FMatrix &a, const EST_FMatrix &b);
EST_FMatrix operator+(const EST_FMatrix &a, const EST_FMatrix &b);

EST_FVector operator-(const EST_FVector &a, const EST_FVector &b);
EST_FVector operator+(const EST_FVector &a, const EST_FVector &b);

EST_FMatrix sub(const EST_FMatrix &a, int row, int col);
EST_FMatrix fmatrix_abs(const EST_FMatrix &a);

EST_FMatrix row(const EST_FMatrix &a, int row);
EST_FMatrix column(const EST_FMatrix &a, int col);


/// least squares fit
bool
polynomial_fit(EST_FVector &x, EST_FVector &y, EST_FVector &co_effs, int order);

/// weighted least squares fit
bool
polynomial_fit(EST_FVector &x, EST_FVector &y, EST_FVector &co_effs, 
	       EST_FVector &weights, int order);

float
polynomial_value(const EST_FVector &coeffs, const float x);

/// vector dot product
float operator*(const EST_FVector &v1, const EST_FVector &v2);

VAL_REGISTER_CLASS_DCLS(fmatrix,EST_FMatrix)
VAL_REGISTER_CLASS_DCLS(fvector,EST_FVector)

#endif
