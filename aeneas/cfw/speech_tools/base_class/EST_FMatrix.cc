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
/*                         Author :  Paul Taylor                         */
/*                         Date   :  April 1995                          */
/*-----------------------------------------------------------------------*/
/*                        Matrix Class for floats                        */
/*                                                                       */
/*=======================================================================*/

#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <cmath>
#include <climits>
using namespace std;

#include "EST_String.h"
#include "EST_types.h"
#include "EST_FileType.h"
#include "EST_Option.h"
#include "EST_FMatrix.h"
#include "EST_cutils.h"  // for swap functions 
#include "EST_Token.h"


/* EST_FVector may used as EST_Val */
VAL_REGISTER_CLASS(fvector,EST_FVector)

/* EST_FMatrix may used as EST_Val */
VAL_REGISTER_CLASS(fmatrix,EST_FMatrix)

EST_String EST_FMatrix::default_file_type = "est_ascii";

EST_FMatrix::EST_FMatrix(const EST_FMatrix &a, int b)
:EST_TSimpleMatrix<float>(a.num_rows(), a.num_columns())
{
    float vv = 0.0;
    if (b < 0)
	return;
    if (b == 0)
	fill(vv);
}

EST_FMatrix & EST_FMatrix::operator+=(const EST_FMatrix &a)
{
    int i, j;
    if (a.num_columns() != num_columns())
    {
	cerr <<"Matrix addition error: bad number of columns\n";
	return *this;
    }
    if (a.num_rows() != num_rows())
    {
	cerr <<"Matrix addition error: bad number of rows\n";
	return *this;
    }
    for (i = 0; i < num_rows(); ++i)
	for (j = 0; j < num_columns(); ++j)
	    a_no_check(i, j) += a.a_no_check(i,j);

    return *this;
}

EST_FMatrix & EST_FMatrix::operator-=(const EST_FMatrix &a)
{
    int i, j;
    if (a.num_columns() != num_columns())
    {
	cerr <<"Matrix subtraction error: bad number of columns\n";
	return *this;
    }
    if (a.num_rows() != num_rows())
    {
	cerr <<"Matrix subtraction error: bad number of rows\n";
	return *this;
    }
    for (i = 0; i < num_rows(); ++i)
	for (j = 0; j < num_columns(); ++j)
	    a_no_check(i, j) -= a.a_no_check(i,j);

    return *this;
}

EST_FMatrix & EST_FMatrix::operator*=(const float f)
{

    int i,j;
    for (i = 0; i < num_rows(); ++i)
	for (j = 0; j < num_columns(); ++j)
	    a_no_check(i, j) *= f;

    return *this;
}

EST_FMatrix & EST_FMatrix::operator/=(const float f)
{

    int i,j;
    for (i = 0; i < num_rows(); ++i)
	for (j = 0; j < num_columns(); ++j)
	    a_no_check(i, j) /= f;

    return *this;
}

EST_FMatrix operator+(const EST_FMatrix &a, const EST_FMatrix &b)
{
    EST_FMatrix ab;
    int i, j;
    if (a.num_columns() != b.num_columns())
    {
	cerr <<"Matrix addition error: bad number of columns\n";
	return ab;
    }
    if (a.num_rows() != b.num_rows())
    {
	cerr <<"Matrix addition error: bad number of rows\n";
	return ab;
    }
    ab.resize(a.num_rows(), a.num_columns());
    for (i = 0; i < a.num_rows(); ++i)
	for (j = 0; j < a.num_columns(); ++j)
	    ab.a_no_check(i, j) = a.a_no_check(i, j) + b.a_no_check(i, j);

    return ab;
}

EST_FMatrix operator-(const EST_FMatrix &a,const EST_FMatrix &b)
{
    EST_FMatrix ab;
    int i, j;

    if (a.num_columns() != b.num_columns())
    {
	cerr <<"Matrix subtraction error: bad number of columns:" <<
	    a.num_columns() << " and " << b.num_columns() << endl;
	return ab;
    }
    if (a.num_rows() != b.num_rows())
    {
	cerr <<"Matrix subtraction error: bad number of rows\n";
	return ab;
    }
    ab.resize(a.num_rows(), a.num_columns());
    for (i = 0; i < a.num_rows(); ++i)
	for (j = 0; j < a.num_columns(); ++j)
	    ab.a_no_check(i, j) = a.a_no_check(i, j) - b.a_no_check(i, j);

    return ab;
}

EST_FMatrix operator*(const EST_FMatrix &a, const float x)
{
    EST_FMatrix b(a, 0);
    int i, j;

    for (i = 0; i < a.num_rows(); ++i)
	for (j = 0; j < a.num_columns(); ++j)
	    b.a_no_check(i,j) = a.a_no_check(i,j) * x;

    return b;
}

int operator !=(const EST_FVector &fv1, 
		const EST_FVector &fv2)
{
    int i;
    if(fv1.length() != fv2.length())
	return FALSE;
    for(i=0;i<fv1.length();i++)
	if(fv1.a_no_check(i) != fv2.a_no_check(i))
	    return FALSE;

    return TRUE;
}

EST_FVector operator*(const EST_FMatrix &a, const EST_FVector &v)
{    
    // treat the vector as a column vector
    // multiply each row of the matrix in turn by the vector

    EST_FVector b;
    b.resize(a.num_rows());
    
    if(a.num_columns() != v.n())
    {
	cerr <<"Matrix-vector multiplication error: matrix rows != vector size"
	     << endl;
	return b;
    }

    int i, j;
    for (i = 0; i < a.num_rows(); ++i){
	b[i] = 0.0;
	for (j = 0; j < a.num_columns(); ++j)
	    b.a_no_check(i) += a.a_no_check(i,j) * v.a_no_check(j);
    }
    return b;
}

EST_FVector operator+(const EST_FVector &a, const EST_FVector &b)
{
    EST_FVector ab;
    int i;
    if (a.length() != b.length())
    {
	cerr <<"Vector addition error: mismatched lengths\n";
	return ab;
    }

    ab.resize(a.length());
    for (i = 0; i < a.length(); ++i)
	ab.a_no_check(i) = a.a_no_check(i) + b.a_no_check(i);

    return ab;
}

EST_FVector operator-(const EST_FVector &a, const EST_FVector &b)
{
    EST_FVector ab;
    int i;
    if (a.length() != b.length())
    {
	cerr <<"Vector subtraction error: mismatched lengths\n";
	return ab;
    }

    ab.resize(a.length());
    for (i = 0; i < a.length(); ++i)
	ab.a_no_check(i) = a.a_no_check(i) - b.a_no_check(i);

    return ab;
}


EST_FVector operator*(const EST_FVector &v,const EST_FMatrix &a)
{    
    // treat the vector as a row vector
    // multiply the vector by each column of the matrix in turn

    EST_FVector b;
    b.resize(a.num_columns());
    
    if(a.num_columns() != v.n())
    {
	cerr <<"Matrix-vector multiplication error: matrix rows != vector size"
	     << endl;
	return b;
    }

    int i, j;
    for (j = 0; j < a.num_columns(); ++j){
	b[j] = 0.0;
	for (i = 0; i < a.num_rows(); ++i)
	    b.a_no_check(i) += a.a_no_check(i,j) * v.a_no_check(j);
    }
    return b;
}


#if 0
EST_FMatrix operator/(const EST_FMatrix &a, float x)
{
    return (a * (1/x));
}
#endif

EST_FMatrix operator*(const EST_FMatrix &a, const EST_FMatrix &b)
{
    EST_FMatrix ab;
    multiply(a,b,ab);
    return ab;
}

void multiply(const EST_FMatrix &a, const EST_FMatrix &b, EST_FMatrix &ab)
{

    if (a.num_columns() != b.num_rows())
    {
	cerr <<"Matrix multiply error: a.num_columns() != b.num_rows()\n";
	return;
    }

    ab.resize(a.num_rows(), b.num_columns());
    int i, j, k, n;
    n = a.num_columns();	// could also be b.num_rows()
    
    for (i = 0; i < a.num_rows(); ++i)
	for (k = 0; k < b.num_columns(); ++k)
	{
	    ab.a_no_check(i, k) = 0.0;
	    for (j = 0; j < n; ++j)
		ab.a_no_check(i, k) += 
		    a.a_no_check(i, j) * b.a_no_check(j, k);
	}
}

void EST_FMatrix::copyin(float **inx, int rows, int cols)
{
    int i, j;

    resize(rows, cols);

    for (i = 0; i < rows; ++i)
	for (j = 0; j < cols; ++j)
	    a_no_check(i,j) = inx[i][j];
    
}

EST_write_status EST_FMatrix::save(const EST_String &filename,
				   const EST_String &type)
{

    if ((type == "est_ascii") || (type == "est_binary"))
	return est_save(filename,type);
    else
    {   // the old stuff raw unheadered
	int i, j;
	ostream *outf;
	if (filename == "-")
	    outf = &cout;
	else
	    outf = new ofstream(filename);
	
	if (!(*outf))
	{
	    cerr << "FMatrix: can't open file \"" << filename 
		<<"\" for writing" << endl;
	    return misc_write_error;
	}
    
	for (i = 0; i < num_rows(); ++i)
	{
	    for (j = 0; j < num_columns(); ++j)
		*outf << a_no_check(i,j) << " ";
	    *outf << endl;
	}
	
	if (outf != &cout)
	    delete outf;
	
	return write_ok;
    }
}

EST_write_status EST_FMatrix::est_save(const EST_String &filename,
				       const EST_String &type)
{
    // Binary save with short header for byte swap and sizes
    int i,j;
    FILE *fd;
    if (filename == "-")
	fd = stdout;
    else if ((fd = fopen(filename, "wb")) == NULL)
    {
	cerr << "EST_FMatrix: binsave: failed to open \"" << filename << 
	    "\" for writing" << endl;
	return misc_write_error;
    }

    fprintf(fd,"EST_File fmatrix\n");
    fprintf(fd,"version 1\n");
    if (type == "est_binary")
    {
	fprintf(fd,"DataType binary\n");
	if (EST_LITTLE_ENDIAN)
	    fprintf(fd,"ByteOrder LittleEndian\n");
	else
	    fprintf(fd,"ByteOrder BigEndian\n");
    }
    else
	fprintf(fd,"DataType ascii\n");

    fprintf(fd,"rows %d\n",num_rows());
    fprintf(fd,"columns %d\n",num_columns());

    fprintf(fd,"EST_Header_End\n");

    if (type == "est_binary")
    {
	for (i = 0; i < num_rows(); ++i)
	    for (j=0; j < num_columns(); j++)
		if (fwrite(&a_no_check(i,j),sizeof(float),1,fd) != 1)
		{
		    cerr << "EST_FMatrix: binsave: failed to write row " 
			<< i << " column " << j 
			    << " to \"" << filename << "\"" << endl;
		    return misc_write_error;
		}
    }
    else
    {   // est_ascii
	for (i = 0; i < num_rows(); ++i)
	{
	    for (j=0; j < num_columns(); j++)
		fprintf(fd,"%f ",a_no_check(i,j));
	    fprintf(fd,"\n");
	}
    }
    
    if (fd != stdout)
	fclose(fd);

    return write_ok;
}

EST_read_status EST_FMatrix::est_load(const EST_String &filename)
{
    // ascii/binary load with short header for byte swap and sizes
    int i,j,k;
    int rows, cols, swap;
    EST_TokenStream ts;
    EST_read_status r;
    EST_EstFileType t;
    EST_Option hinfo;
    bool ascii;
    
    if (((filename == "-") ? ts.open(cin) : ts.open(filename)) != 0)
    {
	cerr << "FMatrix: can't open fmatrix input file " 
	    << filename << endl;
	return misc_read_error;
    }
    if ((r = read_est_header(ts, hinfo, ascii, t)) != format_ok)
	return r;
    if (t != est_file_fmatrix)
	return misc_read_error;
    if (hinfo.ival("version") != 1)
    {
	cerr << "FMatrix load: " << ts.pos_description() <<
	    " wrong version of fmatrix format expected 1 but found " <<
		hinfo.ival("version") << endl;
	return misc_read_error;
    }
    rows = hinfo.ival("rows");
    cols = hinfo.ival("columns");
    resize(rows,cols);

    if (ascii)
    {   // an ascii file
	for (i = 0; i < num_rows(); ++i)
	{
	    for (j = 0; j < num_columns(); ++j)
		a_no_check(i,j) = atof(ts.get().string());
	    if (!ts.eoln())
	    {
		cerr << "FMatrix load: " << ts.pos_description() <<
		    " missing end of line at end of row " << i << endl;
		return misc_read_error;
	    }
	}
    }
    else
    {   // a binary file
	float *buff;
	if ((EST_BIG_ENDIAN && (hinfo.sval("ByteOrder")=="LittleEndian")) ||
	    (EST_LITTLE_ENDIAN && (hinfo.sval("ByteOrder") == "BigEndian")))
	    swap = TRUE;
	else
	    swap = FALSE;

	buff = walloc(float,rows*cols);
	// A single read is *much* faster than multiple reads
	if (ts.fread(buff,sizeof(float),rows*cols) != rows*cols)
	{
	    cerr << "EST_FMatrix: binload: short file in \""  
		<< filename << "\"" << endl;
	    return misc_read_error;
	}
	if (swap)
	    swap_bytes_float(buff,rows*cols);
	for (k = i = 0; i < num_rows(); ++i)
	    for (j = 0; j < num_columns(); ++j)
		a_no_check(i,j) = buff[k++];
	wfree(buff);
    }

    ts.close();
    
    return read_ok;
}

EST_read_status EST_FMatrix::load(const EST_String &filename)
{
    EST_read_status r;

    if ((r = est_load(filename)) == format_ok)
	return r;
    else if (r == wrong_format)
    {   // maybe its an ancient ascii file
	EST_TokenStream ts, tt;
	EST_StrList sl;
	int i, j, n_rows=0, n_cols=0;
	EST_String t;
	EST_Litem *p;
	if (((filename == "-") ? ts.open(cin) : ts.open(filename)) != 0)
	{
	    cerr << "Can't open fmatrix file " << filename << endl;
	    return misc_read_error;
	}
	// set up the character constant values for this stream
	ts.set_SingleCharSymbols(";");
    
	// first read in as list
	for (n_rows = 0; !ts.eof(); ++n_rows)
	    sl.append(ts.get_upto_eoln().string());

	if (n_rows > 0)
	{
	    tt.open_string(sl.first());
	    for (n_cols = 0; !tt.eof(); ++n_cols)
		tt.get().string();
	}

	// resize track and copy values in
	resize(n_rows, n_cols);
	
	for (p = sl.head(), i = 0; p != 0; ++i, p = p->next())
	{
	    tt.open_string(sl(p));
	    for (j = 0; !tt.eof(); ++j)
		a_no_check(i,j) = atof(tt.get().string());
	    if (j != n_cols)
	    {
		cerr << "Wrong number of points in row " << i << endl;
		cerr << "Expected " << n_cols << " got " << j << endl;
		return misc_read_error;
	    }
	}
	return format_ok;
    }
    else
	return r;
}


EST_FVector & EST_FVector::operator+=(const EST_FVector &s)
{
    int i;
    if(n() != s.n()){
	cerr << "Cannot elementwise add vectors of differing lengths" 
	    << endl;
	return *this;
    }
    
    for (i = 0; i < n(); ++i)
	(*this)[i] += s(i);

    
    return *this;
}


EST_FVector& EST_FVector::operator*=(const EST_FVector &s)
{
    if(n() != s.n()){
	cerr << "Cannot elementwise multiply vectors of differing lengths" 
	    << endl;
	return *this;
    }

    for (int i = 0; i < n(); ++i)
	(*this)[i] *= s(i);

    return *this;
}

EST_FVector& EST_FVector::operator*=(const float f)
{
    for (int i = 0; i < n(); ++i)
	(*this)[i] *= f;

    return *this;
}


EST_FVector& EST_FVector::operator/=(const float f)
{
    for (int i = 0; i < n(); ++i)
	(*this)[i] /= f;

    return *this;
}



EST_read_status EST_FVector::est_load(const EST_String &filename)
{    
  // ascii/binary load with short header for byte swap and sizes
  int i,k;
  int l, swap;
  EST_TokenStream ts;
  EST_read_status r;
  EST_EstFileType t;
  EST_Option hinfo;
  bool ascii;
    
  if (((filename == "-") ? ts.open(cin) : ts.open(filename)) != 0)
    {
      cerr << "FVector: can't open FVector input file " 
	   << filename << endl;
      return misc_read_error;
    }
  if ((r = read_est_header(ts, hinfo, ascii, t)) != format_ok)
    return r;
    if (t != est_file_fvector)
      return misc_read_error;
    if (hinfo.ival("version") != 1)
      {
	cerr << "FVector load: " << ts.pos_description() <<
	  " wrong version of FVector format expected 1 but found " <<
	  hinfo.ival("version") << endl;
	return misc_read_error;
      }
    l = hinfo.ival("length");
    resize(l);
    
    if (ascii)
      {   // an ascii file
	for (i = 0; i < length(); ++i)
	  a_no_check(i) = atof(ts.get().string());
      }
    else
      {   // a binary file
       float *buff;
	if ((EST_BIG_ENDIAN && (hinfo.sval("ByteOrder")=="LittleEndian")) ||
	    (EST_LITTLE_ENDIAN && (hinfo.sval("ByteOrder") == "BigEndian")))
	  swap = TRUE;
	else
	  swap = FALSE;
	
	buff = walloc(float,l);
	// A single read is *much* faster than multiple reads
	if (ts.fread(buff,sizeof(float),l) != l)
	  {
	    cerr << "EST_FVector: binload: short file in \""  
		 << filename << "\"" << endl;
	    return misc_read_error;
	  }
	if (swap)
	  swap_bytes_float(buff,l);
	for (k = i = 0; i < length(); ++i)
	  a_no_check(i) = buff[k++];
	wfree(buff);
      }
    
    ts.close();
    return read_ok;
}

EST_read_status EST_FVector::load(const EST_String &filename)
{    

    EST_read_status r;

    if ((r = est_load(filename)) == format_ok)
	return r;
    else if (r == wrong_format)
    {   // maybe its an ancient ascii file
      EST_TokenStream ts;
      EST_String s;
      int i;

      i = 0;
      
      if (((filename == "-") ? ts.open(cin) : ts.open(filename)) != 0)
	{
	  cerr << "can't open vector input file " << filename << endl;
	  return misc_read_error;
	}
      ts.set_SingleCharSymbols(";");
      
      while (!ts.eof())
	{
	  ts.get();
	  ++i;
	}
      resize(i);
      
      ts.close();
      if (((filename == "-") ? ts.open(cin) : ts.open(filename)) != 0)
	{
	  cerr << "can't open vector input file " << filename << endl;
	  return misc_read_error;
	}
      
      for (i = 0; !ts.eof(); ++i)
	{
	  s = ts.get().string();
	  (*this)[i] = (float)(atof(s));  // actually returns double
	}
      ts.close();
      return format_ok;
    }
    else
      return r;

    return format_ok;

}

/*

EST_read_status EST_FVector::load(EST_String &filename)
{    
    EST_TokenStream ts;
    EST_String s;
    int i;

    i = 0;
    
    if (((filename == "-") ? ts.open(cin) : ts.open(filename)) != 0)
    {
	cerr << "can't open vector input file " << filename << endl;
	return misc_read_error;
    }
    ts.set_SingleCharSymbols(";");

    while (!ts.eof())
    {
	ts.get();
	++i;
    }
    resize(i);

    ts.close();
    if (((filename == "-") ? ts.open(cin) : ts.open(filename)) != 0)
    {
	cerr << "can't open vector input file " << filename << endl;
	return misc_read_error;
    }

    for (i = 0; !ts.eof(); ++i)
    {
	s = ts.get().string();
	(*this)[i] = atof(s);
    }
    ts.close();
    return format_ok;
}
*/

//  EST_read_status EST_DVector::load(EST_String &filename)
//  {    
//      EST_TokenStream ts;
//      EST_String s;
//      int i;

//      i = 0;
    
//      if (((filename == "-") ? ts.open(cin) : ts.open(filename)) != 0)
//      {
//  	cerr << "can't open vector input file " << filename << endl;
//  	return misc_read_error;
//      }
//      ts.set_SingleCharSymbols(";");

//      while (!ts.eof())
//      {
//  	ts.get();
//  	++i;
//      }
//      resize(i);

//      ts.close();
//      if (((filename == "-") ? ts.open(cin) : ts.open(filename)) != 0)
//      {
//  	cerr << "can't open vector input file " << filename << endl;
//  	return misc_read_error;
//      }

//      for (i = 0; !ts.eof(); ++i)
//      {
//  	s = ts.get().string();
//  	(*this)[i] = atof(s);  // actually returns double
//      }
//      ts.close();
//      return format_ok;
//  }
    

float operator*(const EST_FVector &v1, const EST_FVector &v2)
{    
    // dot product

    float b=0;
    
    if(v1.length() != v2.length())
    {
	cerr <<"Vector dot product error: differing vector size"
	     << endl;
	return b;
    }

    int i;
    for (i = 0; i < v1.length(); ++i)
	b += v1.a_no_check(i) * v2.a_no_check(i);

    return b;
}


EST_write_status EST_FVector::save(const EST_String &filename,
				   const EST_String &type)
{

    if ((type == "est_ascii") || (type == "est_binary"))
	return est_save(filename,type);
    else
    {   // the old stuff raw unheadered
	int i;
	ostream *outf;
	if (filename == "-")
	    outf = &cout;
	else
	    outf = new ofstream(filename);

	outf->precision(25);
	if (!(*outf))
	{
	    cerr << "FVector: can't open file \"" << filename 
		<<"\" for writing" << endl;
	    return misc_write_error;
	}
    
	for (i = 0; i < length(); ++i)
	    *outf << a_no_check(i) << " ";
	*outf << endl;
	
	if (outf != &cout)
	    delete outf;
	
	return write_ok;
    }
}

EST_write_status EST_FVector::est_save(const EST_String &filename,
				      const EST_String &type)
{
    // Binary save with short header for byte swap and sizes
    int i;
    FILE *fd;
    if (filename == "-")
	fd = stdout;
    else if ((fd = fopen(filename, "wb")) == NULL)
    {
	cerr << "EST_FVector: binsave: failed to open \"" << filename << 
	    "\" for writing" << endl;
	return misc_write_error;
    }

    fprintf(fd,"EST_File fvector\n");
    fprintf(fd,"version 1\n");
    if (type == "est_binary")
    {
	fprintf(fd,"DataType binary\n");
	if (EST_LITTLE_ENDIAN)
	    fprintf(fd,"ByteOrder LittleEndian\n");
	else
	    fprintf(fd,"ByteOrder BigEndian\n");
    }
    else
	fprintf(fd,"DataType ascii\n");

    fprintf(fd,"length %d\n",length());
    fprintf(fd,"EST_Header_End\n");

    if (type == "est_binary")
    {
	for (i = 0; i < length(); ++i)
	    if (fwrite(&a_no_check(i),sizeof(float),1,fd) != 1)
	    {
		cerr << "EST_FVector: binsave: failed to write item " 
		     << i << " to \"" << filename << "\"" << endl;
		return misc_write_error;
	    }
    }
    else
    {   // est_ascii
	for (i = 0; i < length(); ++i)
	    fprintf(fd,"%.25f ",a_no_check(i));
	fprintf(fd,"\n");
    }
    
    if (fd != stdout)
	fclose(fd);

    return write_ok;
}

