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
/*                     Author :  Paul Taylor                             */
/*                     Date   :  December 1997                           */
/*-----------------------------------------------------------------------*/
/*                                                                       */
/*=======================================================================*/

#include "EST_Complex.h"


// Addition

EST_Complex operator + (const EST_Complex& z1, const EST_Complex &z2)
{
    return EST_Complex(z1.r + z2.r, z1.i + z2.i);
}

EST_Complex operator + (const EST_Complex& z, float x)
{
    return EST_Complex(z.r + x, z.i);
}

EST_Complex operator + (float x, const EST_Complex &z)
{
    return EST_Complex(z.r + x, z.i);
}

// Subtraction

EST_Complex operator - (const EST_Complex& z1, const EST_Complex &z2)
{
    return EST_Complex(z1.r - z2.r, z1.i - z2.i);
}

EST_Complex operator - (const EST_Complex& z, float x)
{
    return EST_Complex(z.r -x, z.i);
}

EST_Complex operator - (float x, const EST_Complex &z)
{
    return EST_Complex(x - z.r, - z.i);
}

// Multiplication

EST_Complex operator * (const EST_Complex& z1, const EST_Complex &z2)
{
    return EST_Complex((z1.r * z2.r) - (z1.i * z2.i), (z1.r * z2.i) + (z1.i * z2.r));
}

EST_Complex operator * (const EST_Complex& z, float x)
{
    return EST_Complex(z.r * x, z.i *x);
}

EST_Complex operator * (float x, const EST_Complex &z)
{
    return EST_Complex(z.r * x, z.i *x);
}

// Division

EST_Complex operator / (const EST_Complex& z1, const EST_Complex &z2)
{
  double m = z1.mag();
  
  EST_Complex inv(z1.r / m, z1.i /m);

  return inv * z2;
}

EST_Complex operator / (const EST_Complex& z, float x)
{
    return EST_Complex(z.r / x, z.i / x);
}

EST_Complex operator / (float x, const EST_Complex &z)
{
    EST_Complex a(x, 0.0);
    return (z / a);
}


