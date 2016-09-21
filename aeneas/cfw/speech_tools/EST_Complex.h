/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                        Copyright (c) 1997                             */
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
/*                Author :  Paul Taylor (pault@cstr.ed.ac.uk)            */
/*                Date   :  December 1997                                */
/*-----------------------------------------------------------------------*/
/*                                                                       */
/*                                                                       */
/*************************************************************************/

#ifndef __EST_COMPLEX_H__
#define __EST_COMPLEX_H__

#include <iostream>
#include <cmath>
using namespace std;


#ifndef PI
#define PI 3.14159265358979323846
#endif


/** A class for complex numbers. The class stores the values as
cartesian real and imaginary parts, but these can be read as polar
coordinates using the {\tt mag()} and {\tt ang()} functions. Addition,
subtraction, multiplication and division are supported. */

class EST_Complex {
 private:
    double r;
    double i;
public:
    /**@name Constructor functions */
    //@{
    /// default constructor, initialises values to 0.0
    EST_Complex() {r = 0.0; i = 0.0;}
    /// Constructor initialising real and imaginary parts
    EST_Complex(double real, double imag)
    { r = real; i = imag;}
    //@}

    /// Polar magnitude, read only
    double mag() const 
    { return(sqrt(r*r+i*i)); }

    /// Polar angle, read only
    double ang(int degrees=0) const {
	double a;
	if ( r == 0. && i == 0. ) a = 0.0;
	else if ( r >= 0. ) a = atan(i/r);
	else if ( i >= 0. ) a = atan(i/r) + PI;
	else a = atan(i/r) - PI;
	return (degrees == 1) ? (a * 180 / PI) : a;
    }

    /// The real part - can be used for reading or writing
    double &real() {return r;}
    /// The imaginary part - can be used for reading or writing
    double &imag() {return i;}

friend EST_Complex operator + (const EST_Complex& z1, const EST_Complex &z2);
friend EST_Complex operator + (const EST_Complex &z, float x);
friend EST_Complex operator + (float x, const EST_Complex &z);
friend EST_Complex operator - (const EST_Complex &z1, const EST_Complex &z2);
friend EST_Complex operator - (const EST_Complex &z, float x);
friend EST_Complex operator - (float x, const EST_Complex &z);
friend EST_Complex operator * (const EST_Complex &z1, const EST_Complex &z2);
friend EST_Complex operator * (const EST_Complex &z, float x);
friend EST_Complex operator * (float x, const EST_Complex &z);
friend EST_Complex operator / (const EST_Complex &z1, const EST_Complex &z2);
friend EST_Complex operator / (const EST_Complex &z, float x);
friend EST_Complex operator / (float x, const EST_Complex &z);


friend ostream& operator<< (ostream& s,  const EST_Complex& a)
{ s << a.r << " " << a.i; return s;}
};  



#endif	
