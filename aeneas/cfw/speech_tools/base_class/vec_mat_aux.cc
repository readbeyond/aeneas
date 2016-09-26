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
/*                         Author :  Simon King                          */
/*                         Date   :  April 1995                          */
/*-----------------------------------------------------------------------*/
/*                  EST_FMatrix Class auxiliary functions                */
/*                                                                       */
/*=======================================================================*/
#include "EST_FMatrix.h"
#include "EST_system.h"
#include <cstdlib>
#include <climits>
#include "EST_unix.h"
#include "EST_math.h"
#include <ctime>

bool polynomial_fit(EST_FVector &x, EST_FVector &y, 
		    EST_FVector &co_effs, int order)
{
    EST_FVector weights;
    weights.resize(x.n());
    for(int i=0; i<x.n(); ++i)
	weights[i] = 1.0;
    
    return polynomial_fit(x,y,co_effs,weights,order);
}

bool polynomial_fit(EST_FVector &x, EST_FVector &y, EST_FVector &co_effs, 
	       EST_FVector &weights, int order)
{

    if(order <= 0){
	cerr << "polynomial_fit : order must be >= 1" << endl;
	return false;
    }

    if(x.n() != y.n()){
	cerr << "polynomial_fit : x and y must have same dimension" << endl;
	return false;
    }

    if(weights.n() != y.n()){
	cerr << "polynomial_fit : weights must have same dimension as x and y" << endl;
	return false;
    }

    if(x.n() <= order){
	cerr << "polynomial_fit : x and y must have at least order+1 elements"
	    << endl;
	return false;
    }
    

    // matrix of basis function values
    EST_FMatrix A;
    A.resize(x.n(),order+1);
    
    EST_FVector y1;
    y1.resize(y.n());
    
    for(int row=0;row<y.n();row++)
    {
	y1[row] = y[row] * weights[row];
	for(int col=0;col<=order;col++){
	    A(row,col) = pow(x[row],(float)col) * weights[row];
	    
	}
    }
    
    // could call pseudo_inverse, but save a bit by doing
    // it here since we need transpose(A) anyway
    
    EST_FMatrix At, At_A, At_A_inv;
    int singularity=-2;

    transpose(A,At);
    multiply(At,A,At_A);
    
    // error check
    if(!inverse(At_A,At_A_inv,singularity))
    {
	cerr << "polynomial_fit : inverse failed (";
	if(singularity == -2)
	    cerr << "unspecified reason)" << endl;
	else if(singularity == -1)
	    cerr << "non-square !!)" << endl;
	else
	{
	    cerr << "singularity at point : " << singularity;
	    cerr << " = " << x[singularity] << "," << y[singularity];
	    cerr << " )" << endl;
	}
	return false;
    }
    
    EST_FVector At_y1 = At * y1;
    co_effs = At_A_inv * At_y1;
    return true;
    
}

float matrix_max(const EST_FMatrix &a)
{
    int i, j;
    float v = INT_MIN;
    
    for (i = 0; i < a.num_rows(); ++i)
	for (j = 0; j < a.num_columns(); ++j)
	    if (a.a_no_check(i, j) > v)
		v = a.a_no_check(i, j);
    
    return v;
}

int square(const EST_FMatrix &a)
{
    return a.num_rows() == a.num_columns();
}
// add all elements in matrix.
float sum(const EST_FMatrix &a)
{
    int i, j;
    float t = 0.0;
    for (i = 0; i < a.num_rows(); ++i)
	for (j = 0; j < a.num_columns(); ++j)
	    t += a.a_no_check(i, j);
    return t;
}

// set all elements not on the diagonal to zero.
EST_FMatrix diagonalise(const EST_FMatrix &a)
{
    int i;
    EST_FMatrix b(a, 0);	// initialise and fill b with zeros

    if (a.num_rows() != a.num_columns())
    {
	cerr << "diagonalise: non-square matrix ";
	return b;
    }
    
    for (i = 0; i < a.num_rows(); ++i)
	b(i, i) = a.a_no_check(i, i);
    
    return b;
}

// set all elements not on the diagonal to zero.
void inplace_diagonalise(EST_FMatrix &a)
{
    // NB - will work on non-square matrices without warning
    int i,j;
    
    for (i = 0; i < a.num_rows(); ++i)
	for (j = 0; j < a.num_columns(); ++j)
	    if(i != j)
		a.a_no_check(i, j) = 0;
}

EST_FMatrix sub(const EST_FMatrix &a, int row, int col)
{
    int i, j, I, J;
    int n = a.num_rows() - 1;
    EST_FMatrix s(n, n);
    
    for (i = I = 0; i < n; ++i, ++I)
    {
	if (I == row)
	    ++I;
	for (j = J = 0; j < n; ++j, ++J)
	{
	    if (J == col)
		++J;
	    s(i, j) = a.a(I, J);
	}
    }
    
    //    cout << "sub: row " << row  << " col " << col << "\n" << s;
    
    return s;
}

EST_FMatrix row(const EST_FMatrix &a, int row)
{
    EST_FMatrix s(1, a.num_columns());
    int i;
    
    for (i = 0; i < a.num_columns(); ++i)
	s(0, i) = a.a(row, i);
    
    return s;
}

EST_FMatrix column(const EST_FMatrix &a, int col)
{
    EST_FMatrix s(a.num_rows(), 1);
    int i;
    
    for (i = 0; i < a.num_rows(); ++i)
	s(i, 0) = a.a(i, col);
    
    return s;
}

EST_FMatrix triangulate(const EST_FMatrix &a)
{
    EST_FMatrix b(a, 0);
    int i, j;
    
    for (i = 0; i < a.num_rows(); ++i)
	for (j = i; j < a.num_rows(); ++j)
	    b(j, i) = a.a(j, i);
    
    return b;
}

void transpose(const EST_FMatrix &a,EST_FMatrix &b)
{
    int i, j;
    b.resize(a.num_columns(), a.num_rows());
    
    for (i = 0; i < b.num_rows(); ++i)
	for (j = 0; j < b.num_columns(); ++j)
	    b.a_no_check(i, j) = a.a_no_check(j, i);
}

EST_FMatrix backwards(EST_FMatrix &a)
{
    int i, j, n;
    n = a.num_columns();
    EST_FMatrix t(n, n);
    
    for (i = 0; i < n; ++i)
	for (j = 0; j < n; ++j)
	    t(n - i - 1, n - j - 1) = a.a(i, j);
    
    return t;
}


// changed name from abs as it causes on at least on POSIX machine
// where int abs(int) is a macro
EST_FMatrix fmatrix_abs(const EST_FMatrix &a)
{
    int i, j;
    EST_FMatrix b(a, 0);
    
    for (i = 0; i < a.num_rows(); ++i)
	for (j = 0; j < a.num_columns(); ++j)
	    b.a_no_check(i, j) = fabs(a.a_no_check(i, j));
    
    return b;
}

static void row_swap(int from, int to, EST_FMatrix &a)
{
    int i;
    float f;

    if (from == to)
	return;

    for (i=0; i < a.num_columns(); i++)
    {
	f = a.a_no_check(to,i);
	a.a_no_check(to,i) = a.a_no_check(from,i);
	a.a_no_check(from,i) = f;
    }
}

int inverse(const EST_FMatrix &a,EST_FMatrix &inv)
{
    int singularity=0;
    return inverse(a,inv,singularity);
}

int inverse(const EST_FMatrix &a,EST_FMatrix &inv,int &singularity)
{

    // Used to use a function written by Richard Tobin (in C) but 
    // we no longer need C functionality any more.   This algorithm 
    // follows that in "Introduction to Algorithms", Cormen, Leiserson
    // and Rivest (p759) and the term Gauss-Jordon is used for some part,
    // As well as looking back at Richard's.
    // This also keeps a record of which rows are which from the original
    // so that it can return which column actually has the singularity
    // in it if it fails to find an inverse.
    int i, j, k;
    int n = a.num_rows();
    EST_FMatrix b = a;  // going to destructively manipulate b to get inv
    EST_FMatrix pos;    // the original position
    float biggest,s;
    int r=0,this_row,all_zeros;

    singularity = -1;
    if (a.num_rows() != a.num_columns())
	return FALSE;

    // Make the inverse the identity matrix.
    inv.resize(n,n);
    pos.resize(n,1);
    for (i=0; i<n; i++)
	for (j=0; j<n; j++)
	    inv.a_no_check(i,j) = 0.0;
    for (i=0; i<n; i++)
    {
	inv.a_no_check(i,i) = 1.0;
	pos.a_no_check(i,0) = (float)i;
    }

    // Manipulate b to make it into the identity matrix, while performing
    // the same manipulation on inv.  Once b becomes the identity inv will
    // be the inverse (unless theres a singularity)

    for (i=0; i<n; i++)
    {
	// Find the absolute largest val in this col as the next to
	// manipulate.
	biggest = 0.0;
	r = 0;
	for (j=i; j<n; j++)
	{
	    if (fabs(b.a_no_check(j,i)) > biggest)
	    {
		r = j;
		biggest = fabs(b.a_no_check(j,i));
	    }
	}

	if (biggest == 0.0)  // oops found a singularity
	{   
	    singularity = (int)pos.a_no_check(i,0);
	    return FALSE;
	}

	// Swap current with biggest
	this_row = (int)pos.a_no_check(i,0);  // in case we need this number
	row_swap(r,i,b);
	row_swap(r,i,inv);
	row_swap(r,i,pos);

	// Make b(i,i) = 1
	s = b(i,i);
	for (k=0; k<n; k++)
	{
	    b.a_no_check(i,k) /= s;
	    inv.a_no_check(i,k) /= s;
	}

	// make rest in col(i) 0
	for (j=0; j<n; j++)
	{
	    if (j==i) continue;
	    s = b.a_no_check(j,i);
	    all_zeros = TRUE;
	    for (k=0; k<n; k++)
	    {
		b.a_no_check(j,k) -= b.a_no_check(i,k) * s;
		if (b.a_no_check(j,k) != 0)
		    all_zeros = FALSE;
		inv.a_no_check(j,k) -= inv.a_no_check(i,k) * s;
	    }
	    if (all_zeros)
	    {
		// printf("singularity between (internal) columns %d %d\n",
		//       this_row,j);
		// always identify greater row so linear regression
		// can preserve intercept in column 0
		singularity = ((this_row > j) ? this_row : j);
		return FALSE;
	    }
	}
    }

    return TRUE;
}

int pseudo_inverse(const EST_FMatrix &a, EST_FMatrix &inv)
{
    int singularity=0;
    return pseudo_inverse(a,inv,singularity);
}

int pseudo_inverse(const EST_FMatrix &a, EST_FMatrix &inv,int &singularity)
{
    // for non-square matrices
    // useful for solving linear eqns
    // (e.g. polynomial fitting)
    
    // is it square ?
    if( a.num_rows() == a.num_columns() )
	return inverse(a,inv,singularity);
    
    if( a.num_rows() < a.num_columns() )
	return FALSE;
    
    EST_FMatrix a_trans,atrans_a,atrans_a_inverse;

    transpose(a,a_trans);
    multiply(a_trans,a,atrans_a);
    if (!inverse(atrans_a,atrans_a_inverse,singularity))
	return FALSE;
    multiply(atrans_a_inverse,a_trans,inv);
    
    return TRUE;
}


float determinant(const EST_FMatrix &a)
{
    int i, j;
    int n = a.num_rows();
    float det;
    if (!square(a))
    {
	cerr << "Tried to take determinant of non-square matrix\n";
	return 0.0;
    }
    
    EST_FVector A(n);
    
    if (n == 2)			// special case of 2x2 determinant
	return (a.a_no_check(0,0) * a.a_no_check(1,1)) - 
	    (a.a_no_check(0,1) * a.a_no_check(1,0));
    
    float p;
    
    // create cofactor matrix
    j = 1;
    for (i = 0; i < n; ++i)
    {
	p = (float)(i + j + 2);	// because i & j should start at 1
	//	cout << "power " <<p << endl;
	A[i] = pow((float)-1.0, p) * determinant(sub(a, i, j));
    }
    //    cout << "cofactor " << A;
    
    // sum confactor and original matrix 
    det = 0.0;
    for (i = 0; i < n; ++i)
	det += a.a_no_check(i, j) * A[i];
    
    return det;
}

void eye(EST_FMatrix &a, const int n)
{
    int i,j;
    a.resize(n,n);
    for (i=0; i<n; i++)
    {
	for (j=0; j<n; j++)
	    a.a_no_check(i,j) = 0.0;
	
	a.a_no_check(i,i) = 1.0;
    }
}

void eye(EST_FMatrix &a)
{
    int i,n;
    n=a.num_rows();
    if(n != a.num_columns())
    {
	cerr << "Can't make non-square identity matrix !" << endl;
	return;
    }

    a.fill(0.0);
    for (i=0; i<n; i++)
	a.a_no_check(i,i) = 1.0;
}

EST_FVector add(const EST_FVector &a,const EST_FVector &b)
{
  // a + b
  int a_len = a.length();
  EST_FVector ans( a_len );
  
  if(a_len != b.length()){
    cerr << "Can't add vectors of differing lengths !" << endl;
    ans.resize(0);
    return ans;
  };

  for( int i=0; i<a_len; i++ )
    ans.a_no_check(i) = a.a_no_check(i) + b.a_no_check(i);

  return ans;
}

EST_FVector subtract(const EST_FVector &a,const EST_FVector &b)
{
  // a - b
  int a_len = a.length();
  EST_FVector ans( a_len );
  
  if(a_len != b.length()){
    cerr << "Can't subtract vectors of differing lengths !" << endl;
    ans.resize(0);
    return ans;
  };

  for( int i=0; i<a_len; i++ )
    ans.a_no_check(i) = a.a_no_check(i) - b.a_no_check(i);

  return ans;
}

EST_FVector diagonal(const EST_FMatrix &a)
{

    EST_FVector ans;
    if(a.num_rows() != a.num_columns())
    {
	cerr << "Can't extract diagonal of non-square matrix !" << endl;
	return ans;
    }
    int i;
    ans.resize(a.num_rows());
    for(i=0;i<a.num_rows();i++)
	ans.a_no_check(i) = a.a_no_check(i,i);

    return ans;
}

float polynomial_value(const EST_FVector &coeffs, const float x)
{
    float y=0;

    for(int i=0;i<coeffs.length();i++)
	y += coeffs.a_no_check(i) * pow(x,(float)i);

    return y;
}

void symmetrize(EST_FMatrix &a)
{
    // uses include enforcing symmetry
    // of covariance matrices (to compensate
    // for rounding errors)

    float f;

    if(a.num_columns() != a.num_rows())
    {
	cerr << "Can't symmetrize non-square matrix !" << endl;
	return;
    }
	      
    // no need to look at entries on the diagonal !
    for(int i=0;i<a.num_rows();i++)
	for(int j=i+1;j<a.num_columns();j++)
	{
	    f = 0.5 * (a.a_no_check(i,j) + a.a_no_check(j,i));
	    a.a_no_check(i,j) = a.a_no_check(j,i) = f;
	    }
}

void 
stack_matrix(const EST_FMatrix &M, EST_FVector &v)
{
    v.resize(M.num_rows() * M.num_columns());
    int i,j,k=0;
    for(i=0;i<M.num_rows();i++)
	for(j=0;j<M.num_columns();j++)
	    v.a_no_check(k++) = M(i,j);
}


void
make_random_matrix(EST_FMatrix &M, const float scale)
{

    float r;
    for(int row=0;row<M.num_rows();row++)
	for(int col=0;col<M.num_columns();col++)
	{
	    r = scale * ((double)rand()/(double)RAND_MAX);
	    M.a_no_check(row,col) = r;
	}
}

void
make_random_vector(EST_FVector &V, const float scale)
{

    float r;
    for(int i=0;i<V.length();i++)
    {
	r = scale * ((double)rand()/(double)RAND_MAX);
	V.a_no_check(i) = r;
    }
}

void
make_random_symmetric_matrix(EST_FMatrix &M, const float scale)
{
    if(M.num_rows() != M.num_columns())
    {
	cerr << "Can't make non-square symmetric matrix !" << endl;
	return;
    }

    float r;

    for(int row=0;row<M.num_rows();row++)
	for(int col=0;col<=row;col++)
	{
	    r = scale * ((double)rand()/(double)RAND_MAX);
	    M.a_no_check(row,col) = r;
	    M.a_no_check(col,row) = r;
	}
}

void 
make_random_diagonal_matrix(EST_FMatrix &M, const float scale)
{
    if(M.num_rows() != M.num_columns())
    {
	cerr << "Can't make non-square symmetric matrix !" << endl;
	return;
    }

    M.fill(0.0);
    for(int row=0;row<M.num_rows();row++)
	M.a_no_check(row,row) = scale * ((double)rand()/(double)RAND_MAX);


}

void
make_poly_basis_function(EST_FMatrix &T, EST_FVector t)
{
    if(t.length() != T.num_rows())
    {
	cerr << "Can't make polynomial basis function : dimension mismatch !" << endl;
	cerr << "t.length()=" << t.length();
	cerr << "   T.num_rows()=" << T.num_rows() << endl;
	return;
    }
    for(int row=0;row<T.num_rows();row++)
	for(int col=0;col<T.num_columns();col++)
	    T.a_no_check(row,col) = pow(t[row],(float)col);
    
}

int
floor_matrix(EST_FMatrix &M, const float floor)
{
    int i,j,k=0;
    for(i=0;i<M.num_rows();i++)
	for(j=0;j<M.num_columns();j++)
	    if(M.a_no_check(i,j) < floor)
	    {
		M.a_no_check(i,j) = floor;
		k++;
	    }
    return k;
}

EST_FMatrix
cov_prod(const EST_FVector &v1,const EST_FVector &v2)
{
    // form matrix of vector product, e.g. for covariance
    // treat first arg as a column vector and second as a row vector

    EST_FMatrix m;
    m.resize(v1.length(),v2.length());
    
    for(int i=0;i<v1.length();i++)
	for(int j=0;j<v2.length();j++)
	    m.a_no_check(i,j) = v1.a_no_check(i) * v2.a_no_check(j);

    return m;
}

void est_seed()
{
#if !defined(SYSTEM_IS_WIN32)
    unsigned int seed;
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp,&tzp);
    seed = getpid() * (tp.tv_usec&0x7fff);
    cerr << "seed: " << seed << endl;
    srand(seed);
#else
    srand(time(NULL));
#endif
}

#if 0
void est_seed48()
{
    unsigned short seed;
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp,&tzp);
    seed = (getpid()&0x7f) * (tp.tv_usec&0xff);
    cerr << "seed48: " << seed << endl;
    seed48(&seed);
}
#endif
