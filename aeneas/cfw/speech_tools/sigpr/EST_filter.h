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




#ifndef __EST_FILTER_H__
#define __EST_FILTER_H__

#include "EST_Wave.h"
#include "EST_FMatrix.h"
#include "EST_Track.h"

#define DEFAULT_PRE_EMPH_FACTOR 0.95
#define DEFAULT_FILTER_ORDER 199

/**@name FIR filters

Finite impulse response (FIR) filters which are useful for band-pass,
low-pass and high-pass filtering.  

FIR filters perform the following operation:

\[y_t=\sum_{i=0}^{O-1} c_i \; x_{t-i}\]

where \(O\) is the filter order, \(c_i\) are the filter coefficients,
\(x_t\) is the input at time \(t\) and \(y_t\) is the output at time
\(t\). Functions are provided for designing the filter (i.e. finding
the coefficients).

*/

//@{

/** General purpose FIR filter.  This function will filter the
waveform {\tt sig} with a previously designed filter, given as {\tt
numerator}. The filter coefficients can be designed using one of the
designed functions, e.g. \Ref{design_FIR_filter}.

@see design_FIR_filter, design_lowpass_FIR_filter, design_highpass_FIR_filter,
@see FIRfilter, FIRlowpass_filter, FIRhighpass_filter

*/

void FIRfilter(EST_Wave &in_sig, const EST_FVector &numerator, 
	       int delay_correction=0);


/** General purpose FIR filter.  This function will filter the
waveform {\tt sig} with a previously designed filter, given as {\tt
numerator}. The filter coefficients can be designed using one of the
designed functions, e.g. \Ref{design_FIR_filter} .

@see design_FIR_filter, design_lowpass_FIR_filter, design_highpass_FIR_filter,
@see FIRfilter, FIRlowpass_filter, FIRhighpass_filter
*/

void FIRfilter(const EST_Wave &in_sig, EST_Wave &out_sig,
	       const EST_FVector &numerator, int delay_correction=0);

/** General purpose FIR double (zero-phase) filter.  This function
will double filter the waveform {\tt sig} with a previously designed
filter, given as {\tt numerator}. The filter coefficients can be
designed using one of the designed functions,
e.g. \Ref{design_FIR_filter}. Double filtering is performed by
filtering the signal normally, reversing the waveform, filtering
again and reversing the waveform again. Normal filtering will impose a
lag on the signal depending on the order of the filter. By filtering
the signal forwards and backwards, the lags cancel each other out and
the output signal is in phase with the input signal.

@see design_FIR_filter, design_lowpass_FIR_filter, design_highpass_FIR_filter,
@see FIRfilter, FIRlowpass_filter, FIRhighpass_filter 
*/
void FIR_double_filter(EST_Wave &in_sig, EST_Wave &out_sig, 
		       const EST_FVector &numerator);

/** Quick function for one-off low pass filtering. If repeated lowpass
filtering is needed, first design the required filter using
\Ref{design_lowpass_filter}, and then use \Ref{FIRfilter} to do the actual
filtering.

@see design_FIR_filter, design_lowpass_FIR_filter, design_highpass_FIR_filter, FIRfilter, FIRhighpass_filter, FIRlowpass_filter

@param in_sig input waveform, which will be overwritten
@param freq 
@param order number of filter coefficients, eg. 99
*/

void FIRlowpass_filter(EST_Wave &sigin, int freq, int order=DEFAULT_FILTER_ORDER);

/** Quick function for one-off low pass filtering. If repeated lowpass
filtering is needed, first design the required filter using
\Ref{design_lowpass_filter}, and then use \Ref{FIRfilter} to do the actual
filtering.

@param in_sig input waveform
@param out_sig output waveform
@param freq cutoff frequency in Hertz
@param order number of filter coefficients , e.g. 99

@see design_FIR_filter, design_lowpass_FIR_filter, design_highpass_FIR_filter, FIRfilter, FIRhighpass_filter
*/

void FIRlowpass_filter(const EST_Wave &in_sig, EST_Wave &out_sig,
		       int freq, int order=DEFAULT_FILTER_ORDER);

/** Quick function for one-off high pass filtering. If repeated lowpass
filtering is needed, first design the required filter using
design_lowpass_filter, and then use FIRfilter to do the actual
filtering.

@param in_sig input waveform, which will be overwritten
@param freq cutoff frequency in Hertz
@param order number of filter coefficients, eg. 99

@see design_FIR_filter, design_lowpass_FIR_filter, design_highpass_FIR_filter
@see FIRfilter, FIRlowpass_filter, FIRhighpass_filter
*/

void FIRhighpass_filter(EST_Wave &in_sig, int freq, int order);

/** Quick function for one-off high pass filtering. If repeated highpass
filtering is needed, first design the required filter using
design_highpass_filter, and then use FIRfilter to do the actual
filtering.

@param in_sig input waveform
@param out_sig output waveform
@param freq cutoff frequency in Hertz
@param order number of filter coefficients, eg. 99

@see design_FIR_filter, design_lowpass_FIR_filter, design_highpass_FIR_filter
@see FIRfilter, FIRlowpass_filter, FIRhighpass_filter
*/
void FIRhighpass_filter(const EST_Wave &sigin, EST_Wave &out_sig,
			int freq, int order=DEFAULT_FILTER_ORDER);


/** Quick function for one-off double low pass filtering. 

Normal low pass filtering (\Ref{FIRlowpass_filter}) introduces a time delay.
This function filters the signal twice, first forward and then backwards, 
which ensures a zero phase lag. Hence the order parameter need only be
half what it is for (\Ref{FIRlowpass_filter} to achieve the same effect.

@param in_sig input waveform, which will be overwritten
@param freq cutoff frequency in Hertz
@param order number of filter coefficients, eg. 99

@see FIRhighpass_filter
*/
void FIRhighpass_double_filter(EST_Wave &sigin, int freq, 
			      int order=DEFAULT_FILTER_ORDER);

/** Quick function for one-off double low pass filtering. 

Normal low pass filtering (\Ref{FIRlowpass_filter}) introduces a time delay.
This function filters the signal twice, first forward and then backwards, 
which ensures a zero phase lag. Hence the order parameter need only be
half what it is for (\Ref{FIRlowpass_filter} to achieve the same effect.

@param in_sig input waveform
@param out_sig output waveform
@param freq cutoff frequency in Hertz
@param order number of filter coefficients, eg. 99

@see FIRhighpass_filter

*/
void FIRhighpass_double_filter(const EST_Wave &int_sig, EST_Wave &out_sig,
			       int freq, int order=DEFAULT_FILTER_ORDER);

/** Quick function for one-off zero phase high pass filtering. 

Normal high pass filtering (\Ref{FIRhighpass_filter}) introduces a time delay.
This function filters the signal twice, first forward and then backwards, 
which ensures a zero phase lag. Hence the order parameter need only be
half what it is for (\Ref{FIRhighpass_filter} to achieve the same effect.

@param in_sig input waveform, which will be overwritten
@param freq cutoff frequency in Hertz
@param order number of filter coefficients, eg. 99

@see FIRlowpass_filter
*/
void FIRlowpass_double_filter(EST_Wave &sigin, int freq, 
			      int order=DEFAULT_FILTER_ORDER);

/** Quick function for one-off zero phase high pass filtering. 

Normal high pass filtering (\Ref{FIRhighpass_filter}) introduces a time delay.
This function filters the signal twice, first forward and then backwards, 
which ensures a zero phase lag. Hence the order parameter need only be
half what it is for (\Ref{FIRhighpass_filter} to achieve the same effect.

@param in_sig input waveform
@param out_sig output waveform
@param freq cutoff frequency in Hertz
@param order number of filter coefficients, eg. 99

@see FIRlowpass_filter
*/
void FIRlowpass_double_filter(const EST_Wave &in_sig, EST_Wave &out_sig,
			      int freq, int order=DEFAULT_FILTER_ORDER);

//@}

/**@name Linear Prediction filters

The linear prediction filters are used for the analysis and synthesis of
waveforms according the to linear prediction all-pole model.

The linear prediction states that the value of a signal at a given
point is equal to a weighted sum of the previous P values, plus a
correction value for that point:

\[s_{n} = \sum_{i=1}^{P} a_{i}.s_{n-i} + e_{n}\]

Given a set of coefficients and the original signal, we can use this
equation to work out e, the {\it residual}. Conversely given the
coefficients and the residual signal, an estimation of the original
signal can be calculated.

If a single set of coefficients were used for the entire waveform, the
filtering process would be simple. It is usual however to have a
different set of coefficients for every frame, and there are many
possible ways to switch from one coefficient set to another so as not
to cause discontinuities at the frame boundaries.
*/ 

//@{

/** Synthesize a signal from a single set of linear prediction
coefficients and the residual values.

@param  sig the waveform to be synthesized
@param  a a single set of LP coefficients
@param  res the input residual waveform
*/
void lpc_filter(EST_Wave &sig, EST_FVector &a, EST_Wave &res);

/** Filter the waveform using a single set of coefficients so as to
produce a residual signal.

@param  sig the speech waveform to be filtered
@param  a a single set of LP coefficients
@param  res the output residual waveform
*/

void inv_lpc_filter(EST_Wave &sig, EST_FVector &a, EST_Wave &res);

/** Synthesize a signal from a track of linear prediction coefficients.
This function takes a set of LP frames and a residual and produces a
synthesized signal. 

For each frame, the function picks an end point, which is half-way
between the current frame's time position and the next frame's. A
start point is defined as being the previous frame's end. Using these
two values, a portion of residual is extracted and passed to
\Ref{lpc_filter} along with the LP coefficients for that frame.  This
function writes directly into the signal for the values between start
and end;

@param  sig the waveform to be synthesized
@param  lpc a track of time positioned LP coefficients
@param  res the input residual waveform
*/

void lpc_filter_1(EST_Track &lpc, EST_Wave & res, EST_Wave &sig);

/** Synthesize a signal from a track of linear prediction coefficients.
This function takes a set of LP frames and a residual and produces a
synthesized signal. 

This is functionally equivalent to \Ref{lpc_filter_1} except it
reduces the residual by 0.5 before filtering.  Importantly it is
about three times faster than \Ref{lpc_filter_1} but in doing so uses
direct C buffers rather than the neat C++ access function.  This
function should be regarded as temporary and will be deleted after
we restructure the low level classes to give better access.

@param  sig the waveform to be synthesized
@param  lpc a track of time positioned LP coefficients
@param  res the input residual waveform
*/

void lpc_filter_fast(EST_Track &lpc, EST_Wave & res, EST_Wave &sig);

/** Produce a residual from a track of linear prediction coefficients
and a signal using an overlap add technique.

For each frame, the function estimates the local pitch period and
picks a start point one period before the current time position and an
end point one period after it.

A portion of residual corresponding to these times is then produced
using \Ref{inv_lpc_filter}. The resultant section of residual is then
overlap-added into the main residual wave object.

@param  sig the speech waveform to be filtered
@param  lpc a track of time positioned LP coefficients
@param  res the output residual waveform
*/

void inv_lpc_filter_ola(EST_Wave &sig, EST_Track &lpc, EST_Wave &res);

//@}

/**@name Pre/Post Emphasis filters.

These functions adjust the spectral tilt of the input waveform.

*/

//@{

/** Pre-emphasis filtering.  This performs simple high pass
filtering with a one tap filter of value {\tt a}. Normal values of a
range between 0.95 and 0.99.  */

void pre_emphasis(EST_Wave &sig, float a=DEFAULT_PRE_EMPH_FACTOR);

/** Pre-emphasis filtering.  This performs simple high pass
filtering with a one tap filter of value {\tt a}. Normal values of a
range between 0.95 and 0.99.  */


void pre_emphasis(EST_Wave &sig, EST_Wave &out, 
		  float a=DEFAULT_PRE_EMPH_FACTOR);

/** Post-emphasis filtering.  This performs simple low pass
filtering with a one tap filter of value a. Normal values of a range
between 0.95 and 0.99. The same values of {\tt a} should be used when
pre- and post-emphasizing the same signal.  */

void post_emphasis(EST_Wave &sig, float a=DEFAULT_PRE_EMPH_FACTOR);

/** Post-emphasis filtering.  This performs simple low pass
filtering with a one tap filter of value a. Normal values of a range
between 0.95 and 0.99. The same values of {\tt a} should be used when
pre- and post-emphasizing the same signal.  */

void post_emphasis(EST_Wave &sig, EST_Wave &out, 
		   float a=DEFAULT_PRE_EMPH_FACTOR);

//@}

/**@name Miscellaneous filters.

Some of these filters are non-linear and therefore don't fit the
normal paradigm.

*/ //@{

/** Filters the waveform by means of median smoothing. 

This is a sort of low pass filter which aims to remove extreme values.
Median smoothing works examining each sample in the wave, taking all
the values in a window of size {\tt n} around that sample, sorting
them and replacing that sample with the middle ranking sample in the
sorted samples.

@param sig waveform to be filtered
@param n size of smoothing window

*/

void simple_mean_smooth(EST_Wave &c, int n);

//@}

#endif /* __EST_FILTER_H__ */

