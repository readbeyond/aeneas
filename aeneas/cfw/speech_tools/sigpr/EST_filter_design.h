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

#ifndef __EST_FILTER_DESIGN_H__
#define __EST_FILTER_DESIGN_H__

#include "EST_Wave.h"
#include "EST_FMatrix.h"
#include "EST_Track.h"

/**@name Filter Design

FIR Filtering is a 2 stage process, first involving design and then
the filtering itself. As the design is somewhat costly, it is usually
desirable to design a filter outside the main loop.

For one off filtering operations, functions are
provided which design and filter the waveform in a single go.  

It is impossible to design an ideal filter, i.e. one which exactly
obeys the desired frequency response. The "quality" of a filter is
given by the order parameter, with high values indicating good
approximations to desired responses. High orders are slower. The
default is 199 which gives a pretty good filter, but a value as low as
19 is still usable if speech is important.

*/ 
//@{

/** Create an arbitrary filter or order {\tt order} that attempts to
give the frequency response given by {\tt freq_response}. The vector
{\tt freq_response} should be any size 2**N and contain a plot of the
desired frequency response with values ranging between 0.0 and
1.0. The actual filtering is done by \Ref{FIRfilter}.

@see design_lowpass_FIR_filter, design_highpass_FIR_filter
@see FIRfilter, FIRlowpass_filter, FIRhighpass_filter

*/
EST_FVector design_FIR_filter(const EST_FVector &freq_response, int
			      filter_order);

/** Design a FIR lowpass filter of order {\tt order} and cut-off
frequency {\tt freq}. The filter coefficients are returned in the
FVector and should be used in conjunction with \Ref{FIRfilter}.

@see design_FIR_filter, design_highpass_FIR_filter, FIRfilter,
FIRlowpass_filter, FIRhighpass_filter 
*/

EST_FVector design_lowpass_FIR_filter(int sample_rate, int freq, int
				      order);

/** Design a FIR highpass filter of order {\tt order} and cut-off frequency
{\tt freq}. The filter coefficients are returned in the FVector and should be used in conjunction with \Ref{FIRfilter}
@see design_FIR_filter, design_lowpass_FIR_filter, design_highpass_FIR_filter
@see FIRfilter, FIRlowpass_filter, FIRhighpass_filter

*/
EST_FVector design_highpass_FIR_filter(int sample_rate, int
				       freq, int order);

//@}


#endif /* __EST_FILTER_DESIGN_H__ */

