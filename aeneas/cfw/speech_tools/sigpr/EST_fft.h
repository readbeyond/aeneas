/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                     Copyright (c) 1995,1996                           */
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


#ifndef __EST_FFT_H__
#define __EST_FFT_H__

#include "EST_Wave.h"
#include "EST_Track.h"
#include "EST_FMatrix.h"

/**@name Fast Fourier Transform functions

<para>
These are the low level functions where the actual FFT is
performed. Both slow and fast implementations are available for
historical reasons. They have identical functionality. At this time,
vectors of complex numbers are handled as pairs of vectors of real and
imaginary numbers.
</para>

<formalpara> <title>What is a Fourier Transform ?</title>

<para>
The Fourier transform of a signal gives us a frequency-domain
representation of a time-domain signal. In discrete time, the Fourier
Transform is called a Discrete Fourier Transform (DFT) and is given
by:

\[y_k = \sum_{t=0}^{n-1} x_t \; \omega_{n}^{kt} \; ; \; k=0...n-1 \]

where \(y = (y_0,y_1,... y_{n-1})\) is the DFT (of order \(n\) ) of the
signal \(x = (x_0,x_1,... x_{n-1})\), where
\(\omega_{n}^{0},\omega_{n}^{1},... \omega_{n}^{n-1}\) are the n
complex nth roots of 1.
</para>

<para>
The Fast Fourier Transform (FFT) is a very efficient implementation of
a Discrete Fourier Transform. See, for example "Algorithms" by Thomas
H. Cormen, Charles E. Leiserson and Ronald L. Rivest (pub. MIT Press),
or any signal processing textbook.
</para>

</formalpara>

*/

//@{

/** Basic in-place FFT. 

<para>There's no point actually using this - use \Ref{fastFFT}
instead. However, the code in this function closely matches the
classic FORTRAN version given in many text books, so is at least easy
to follow for new users.</para>

<para>The length of <parameter>real</parameter> and
<parameter>imag</parameter> must be the same, and must be a power of 2
(e.g. 128).</para>

@see slowIFFT
@see FastFFT */
int slowFFT(EST_FVector &real, EST_FVector &imag);

/** Alternate name for slowFFT
*/
inline int FFT(EST_FVector &real, EST_FVector &imag){
    return slowFFT(real, imag);
}

/** Basic inverse in-place FFT
int slowFFT
*/
int slowIFFT(EST_FVector &real, EST_FVector &imag);

/** Alternate name for slowIFFT
*/
inline int IFFT(EST_FVector &real, EST_FVector &imag){
    return slowIFFT(real, imag);
}

/** Power spectrum using the fastFFT function.
The power spectrum is simply the squared magnitude of the
FFT. The result real and imaginary parts are both set equal
to the power spectrum (you only need one of them !)
*/
int power_spectrum(EST_FVector &real, EST_FVector &imag);

/** Power spectrum using the slowFFT function
*/
int power_spectrum_slow(EST_FVector &real, EST_FVector &imag);

/** Fast FFT 
An optimised implementation by Tony Robinson to be used
in preference to slowFFT
*/
int fastFFT(EST_FVector &invec);

// Auxiliary for fastFFT
int fastlog2(int);

//@}


#endif // __EST_FFT_H__

