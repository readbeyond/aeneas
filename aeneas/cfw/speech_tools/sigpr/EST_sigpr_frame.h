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

#ifndef __EST_SIGPR_FRAME_H__
#define __EST_SIGPR_FRAME_H__

#include "EST_FMatrix.h"



/**@name Linear Prediction functions
Including, generation of coefficients from the signal, reflection
coefficients, line spectral frequencies, areas.
*/
//@{

/** Produce the full set of linear prediction coefficients from a
 frame of speech waveform. 

@param sig: the frame of input waveform
@param acf: the autocorrelation coefficients
@param ref: the  reflection coefficients 
@param lpc: the LPC coefficients

The order of the lpc analysis is given as the size of the <parameter>
lpc <parameter> vector - 1. The coefficients are placed in the
locations 1 - size, and the energy is placed in location 0.  
*/

void sig2lpc(const EST_FVector &sig, EST_FVector &acf, 
		EST_FVector &ref, EST_FVector &lpc);


/** Calculate cepstral coefficients from lpc coefficients.

It is possible to calculate a set of cepstral coefficients from
lpc coefficients using the relationship: 

\[c_{k}= a_{k} + \frac{1}{k}\sum_{i=1}^{k-1} i c_{i} a_{k-1}\]

The order of the cepstral analysis can be different from the lpc
order. If the cepstral order is greater, interpolation is used (FINISH
add equation). Both orders are taken from the lengths of the
respective vectors.  Note that these cepstral coefficients take on the
assumptions (and errors) of the lpc model and hence will not be the
same as cepstral coefficients calculated using DFT functions.  

@param lpc: the LPC coefficients (input)
@param lpc: the cepstral coefficients (output)
*/

void lpc2cep(const EST_FVector &lpc, EST_FVector &cep);



/** Produce a set linear prediction coefficients from a
 frame of speech waveform. {\tt sig} is the frame of input waveform,
 and {\tt lpc} are the LPC coefficients.  The
 {\bf order} of the lpc analysis is given as the size of the {\tt lpc}
 vector -1. The coefficients are placed in the locations 1 - size, and
 the energy is placed in location 0.
*/
void sig2lpc(const EST_FVector &sig, EST_FVector &lpc);

/** Produce a set of reflection coefficients from a
 frame of speech waveform. {\tt sig} is the frame of input waveform,
 and {\tt ref} are the LPC coefficients.  The
 {\bf order} of the lpc analysis is given as the size of the {\tt lpc}
 vector -1. The coefficients are placed in the locations 1 - size, and
 the energy is placed in location 0.
*/
void sig2ref(const EST_FVector &sig, EST_FVector &ref);


/**@name Area Functions
Using the analogy of the lossless tube, the
cross-sectional areas of the sections of this tube are related to the reflection coefficients and can be calculated from the following relationship:

\[\frac{A_{i+1}}{A_{i}} = \frac{i - k_{i}}{1 + k_{i}} \]

*/
//@{
/** The area according to the formula. */
void ref2truearea(const EST_FVector &ref, EST_FVector &area);

/** An approximation of the area is calculate by skipping the denominator
in the formula. */
void ref2area(const EST_FVector &ref, EST_FVector &area);

/** The logs of the areas. */
void ref2logarea(const EST_FVector &ref, EST_FVector &logarea);
//@}

/** Calculate the reflection coefficients from the lpc
coefficients. Note that in the standard linear prediction analysis,
the reflection coefficients are generated as a by-product.  @see
sig2lpc */

void lpc2ref(const EST_FVector &lpc, EST_FVector &ref);

/** Calculate the linear prediction coefficients from the reflection
coefficients.
Use the equation:
\[power=\frac{1}{n}\sum_{i=1}^{n}a_{i}^2\]

@see lpc2ref*/

void ref2lpc(const EST_FVector &ref, EST_FVector &lpc);

/** Calculate line spectral frequencies from linear prediction coefficients.
Use the equation:
\[power=\frac{1}{n}\sum_{i=1}^{n}a_{i}^2\]

@see lsf2lpc
*/

void lpc2lsf(const EST_FVector &lpc,  EST_FVector &lsf);

/** Calculate line spectral frequencies from linear prediction coefficients.
Use the equation:
\[power=\frac{1}{n}\sum_{i=1}^{n}a_{i}^2\]

@see lpc2lsf
*/

void lsf2lpc(const EST_FVector &lsf, EST_FVector &lpc);
//@} 

void frame_convert(const EST_FVector &in_frame, const EST_String &in_type,
		   EST_FVector &out_frame, const EST_String &out_type);



// end of lpc functions

/**@name Energy and power frame functions
*/

//@{

/** Calculate the power for a frame of speech. This is defined as
\[power=\frac{1}{n}\sum_{i=1}^{n}a_{i}^2\]
*/


void sig2pow(EST_FVector &frame, float &power);

/** Calculate the root mean square energy for a frame of speech. This
is defined as \[energy=\sqrt{\frac{1}{n}\sum_{i=1}^{n}a_{i}^2}\] */

void sig2rms(EST_FVector &frame, float &rms_energy);

//@}
// end of power and  energy

/**@name Frame based filter bank and cepstral analysis

These functions are \Ref{Frame based signal processing functions}.
*/

//@{

/** Calculate the (log) energy (or power) in each channel of a Mel
scale filter bank for a frame of speech. The filters are triangular, are
evenly spaced and are all of equal width, on a Mel scale. The upper and lower
cutoffs of each filter are at the centre frequencies of the adjacent filters.
The Mel scale is described under {\tt Hz2Mel}.

@see Hz2Mel
@see sig2fft
@see fft2fbank
*/

void sig2fbank(const EST_FVector &sig,
	       EST_FVector &fbank_frame,
	       const float sample_rate,
	       const bool use_power_rather_than_energy,
	       const bool take_log);

/** Calculate the energy (or power) spectrum of a frame of speech. The FFT
order is determined by the number of samples in the frame of speech, and is
a power of 2. Note that the FFT vector returned corresponds to frequencies
from 0 to half the sample rate. Energy is the magnitude of the FFT; power is
the squared magnitude. 

@see fft2fbank
@see sig2fbank
*/

void sig2fft(const EST_FVector &sig,
	     EST_FVector &fft_vec,
	     const bool use_power_rather_than_energy);

/** Given a Mel filter bank description, bin the FFT coefficients
to compute the output of the filters. The first and last elements of 
{\tt mel_fbank_frequencies} define the lower and upper bound of
the first and last filters respectively and the intervening elements
give the filter centre  frequencies. That is, {\tt mel_fbank_frequencies} has
two more elements than {\tt fbank_vec}.

@see fastFFT
@see sig2fft
@see sig2fbank
@see fbank2melcep
*/
	     
void fft2fbank(const EST_FVector &fft_frame, 
	       EST_FVector &fbank_vec,
	       const float Hz_per_fft_coeff,
	       const EST_FVector &mel_fbank_frequencies);
	       
/** Compute the discrete cosine transform of log Mel-scale filter bank output
to get the Mel cepstral coefficients for a frame of speech.
Optional liftering (filtering in the cepstral domain) can be applied to
normalise the magnitudes of the coefficients. This is useful because,
typically, the higher order cepstral coefficients are significantly
smaller than the lower ones and it is often desirable to normalise
the means and variances across coefficients.

The lifter (cepstral filter) used is:
\[c_i' = \{ 1 + \frac{L}{2} sin \frac{\Pi i}{L} \} \; c_i\] 

A typical value of L used in speech recognition is 22. A value of L=0 is taken
to mean no liftering. This is equivalent to L=1.

@see sig2fft
@see fft2fbank
@see sig2fbank
*/

void fbank2melcep(const EST_FVector &fbank_vec,
		  EST_FVector &mfcc, 
		  const float liftering_parameter,
		  const bool include_c0 = false);

/** Make a triangular Mel scale filter. The filter is centred at
{\tt this_mel_centre} and
extends from {\tt this_mel_low} to {\tt this_mel_high}. {\tt half_fft_order}
is the length of a power/energy spectrum covering 0Hz to half the sampling
frequency with a resolution of {\tt Hz_per_fft_coeff}.

The routine returns a vector of weights to be applied to the energy/power
spectrum starting at element {\tt fft_index_start}.
The number of points (FFT coefficients) covered
by the filter is given by the length of the returned vector {\tt filter}.

@see fft2fbank
@see Hz2Mel
@see Mel2Hz
*/

void make_mel_triangular_filter(const float this_mel_centre,
				const float this_mel_low,
				const float this_mel_high,
				const float Hz_per_fft_coeff,
				const int half_fft_order,
				int &fft_index_start,
				EST_FVector &filter);

/**@name Frequency conversion functions

These are functions used in \Ref{Filter bank and cepstral analysis}.
*/

//@{

/** Convert Hertz to Mel. The Mel scale is defined by
\[f_{\mbox{Mel}} = 1127 \; log( 1 + \frac{f_{\mbox{Hertz}}}{700} )\]  

@see Mel2Hz
@see Frequency conversion functions
*/

float Hz2Mel(float frequency_in_Hertz);

/** 
Convert Mel to Hertz.

@see Hz2Mel 
*/

float Mel2Hz(float frequency_in_Mel);

//@}
// end of frequency conversion functions

//@}
// end of filter bank and cepstral analysis




#endif /* __EST_SIGPR_FRAME_H__ */
