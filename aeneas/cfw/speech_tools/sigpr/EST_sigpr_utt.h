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

#ifndef __EST_SIGPR_UTT_H__
#define __EST_SIGPR_UTT_H__

#include "sigpr/EST_sigpr_frame.h"
#include "sigpr/EST_Window.h"
#include "EST_Track.h"
#include "EST_Wave.h"

#define DEFAULT_WINDOW_NAME "hamming"
#define DEFAULT_FRAME_FACTOR 2.0

/* Note: some of these functions deliberately don't have
   doc++ style comments, mainly because they are, or will be
   superseded soon.
*/

/**@name Functions for use with frame based processing

In the following functions, the input is a \Ref{EST_Wave} waveform,
and the output is a (usually multi-channel) \Ref{EST_Track}.  The
track must be set up appropriately before hand. This means the track
must be resized accordingly with the correct numbers of frame and
channels.

The positions of the frames are found by examination of the {\bf time}
array in the EST_Track, which must be filled prior to the function
call. The usual requirement is for fixed frame analysis, where each
analysis frame is, say, 10ms after the previous one. 

A common alternative is to perform pitch-synchronous
analysis where the time shift is related to the local pitch period.

*/ 

//@{

/** Produce a single set of coefficients from a waveform. The type of 
  coefficient required is given in the argument <parameter>type</parameter>. 
  Possible types are:

<variablelist>

<varlistentry><term>lpc</term><listitem>linear predictive coding</listitem></varlistentry>

<varlistentry><term>cep</term><listitem>cepstrum coding from lpc coefficients</listitem></varlistentry>

<varlistentry><term>melcep</term><listitem>Mel scale cepstrum coding via fbank</listitem></varlistentry>

<varlistentry><term>fbank</term><listitem>Mel scale log filterbank analysis</listitem></varlistentry>

<varlistentry><term>lsf</term><listitem>line spectral frequencies</listitem></varlistentry>

<varlistentry><term>ref</term><listitem>Linear prediction reflection coefficients</listitem></varlistentry>

<varlistentry><term>power</term><listitem></listitem></varlistentry>

<varlistentry><term>f0</term><listitem>srpd algorithm</listitem></varlistentry>

<varlistentry><term>energy</term><listitem>root mean square energy</listitem></varlistentry>

</variablelist>

The order of the analysis is calculated from the number of
channels in <parameter>fv</parameter>.  The positions of the analysis
windows must be given by filling in the track's time array.

This function windows the waveform at the intervals given by the track
time array. The length of each window is <parameter>factor<parameter>
* the local time shift. The windowing function is giveb by
<parameter>wf</parameter>.

@param sig: input waveform
@param fv: output coefficients. These have been pre-allocated and the
       number of channels in a indicates the order of the analysis.
@param type: the types of coefficients to be produced. "lpc", "cep" etc
@param factor: the frame length factor, i.e. the analysis frame length
       will be this times the local pitch period.

@param wf: function for windowing. See \Ref{Windowing mechanisms}
*/

void sig2coef(EST_Wave &sig, EST_Track &a, EST_String type, 
	      float factor = 2.0, 
	      EST_WindowFunc *wf = EST_Window::creator(DEFAULT_WINDOW_NAME));

/** Produce multiple coefficients from a waveform by repeated calls to
  sig2coef.

@param sig: input waveform
@param fv: output coefficients. These have been pre-allocated and the
       number of channels in a indicates the order of the analysis.
@param op: Features structure containing options for analysis order,
        frame shift etc.
@param slist: list of types of coefficients required, from the set of
possible types that sig2coef can take.
*/

void sigpr_base(EST_Wave &sig, EST_Track &fv, EST_Features &op, 
		const EST_StrList &slist);

/** Calculate the power for each frame of the waveform.

@param sig: input waveform
@param a: output power track
@param factor: the frame length factor, i.e. the analysis frame length
       will be this times the local pitch period.
*/

void power(EST_Wave &sig, EST_Track &a, float factor);

/** Calculate the rms energy for each frame of the waveform.

This function calls
\Ref{sig2energy}


@param sig input waveform
@param a output coefficients
@param factor optional: the frame length factor, i.e. the analysis frame length
       will be this times the local pitch period.

*/

void energy(EST_Wave &sig, EST_Track &a, float factor);


/** Mel scale filter bank analysis. The Mel scale triangular filters
are computed via an FFT (see \Ref{fastFFT}). This routine is required
for Mel cepstral analysis (see \Ref{melcep}). The analysis of each
frame is done by \Ref{sig2fbank}.

A typical filter bank analysis for speech recognition might use log
energy outputs from 20 filters.

@param sig: input waveform
@param fbank: the output. The number of filters is determined from the number
       size of this track.
@param factor: the frame length factor, i.e. the analysis frame length
       will be this times the local pitch period
@param wf: function for windowing. See \Ref{Windowing mechanisms}
@param up: whether the filterbank analysis should use
       power rather than energy.
@param take_log: whether to take logs of the filter outputs

@see sig2fbank
@see melcep
*/

void fbank(EST_Wave &sig,
	   EST_Track &fbank,
	   const float factor,
	   EST_WindowFunc *wf = EST_Window::creator(DEFAULT_WINDOW_NAME),
	   const bool up = false,
	   const bool take_log = true);

/** Mel scale cepstral analysis via filter bank analysis. Cepstral
parameters are computed for each frame of speech. The analysis
requires \Ref{fbank}. The cepstral analysis of the filterbank outputs
is performed by \Ref{fbank2melcep}.

A typical Mel cepstral coefficient (MFCC) analysis for speech recognition
might use 12 cepstral coefficients computed from a 20 channel filterbank.


@param sig input: waveform
@param mfcc_track: the output
@param factor: the frame length factor, i.e. the analysis frame length
       will be this times the local pitch period
@param fbank_order: the number of Mel scale filters used for the analysis
@param liftering_parameter:  for filtering in the cepstral domain
       See \Ref{fbank2melcep}
@param wf: function for windowing. See \Ref{Windowing mechanisms}
@param include_c0: whether the zero'th cepstral coefficient is to be included
@param up: whether the filterbank analysis should use
       power rather than energy.

@see fbank
@see fbank2melcep
*/

void melcep(EST_Wave &sig, 
	    EST_Track &mfcc_track, 
	    float factor,
	    int fbank_order,
	    float liftering_parameter,
	    EST_WindowFunc *wf = EST_Window::creator(DEFAULT_WINDOW_NAME),
	    const bool include_c0 = false,
	    const bool up = false);

//@}


/**@name Pitch/F0 Detection Algorithm functions

These functions are used to produce a track of fundamental frequency
(F0) against time of a waveform.
*/

//@{   


/** Top level pitch (F0) detection algorithm. Returns a track
containing evenly spaced frames of speech, each containing a F0 value
for that point.

At present, only the \Rref{srpd} pitch tracker is implemented, so
this is always called regardless of what <parameter>method</parameter>
is set to.

@param sig: input waveform
@param fz: output f0 contour
@param op: parameters for pitch tracker
@param method: pda method to be used.
*/


void pda(EST_Wave &sig, EST_Track &fz, EST_Features &op, EST_String method="");


/** Top level intonation contour detection algorithm. Returns a track
containing evenly spaced frames of speech, each containing a F0 for that point. {\tt icda} differs from \Ref{pda} in that the contour is
smoothed, and unvoiced portions have interpolated F0
values.

@param sig: input waveform
@param fz: output f0 contour
@param speech: Interpolation is controlled by the <tt>speech</tt> track. When
a point has a positive value in the speech track, it is a candidate
for interpolation.  
@param op: parameters for pitch tracker
@param method: pda method to be used.
*/

void icda(EST_Wave &sig, EST_Track &fz, EST_Track &speech, 
	  EST_Option &op, EST_String method = "");

/** Create a set sensible defaults for use in pda and icda.

*/
void default_pda_options(EST_Features &al);


/** Super resolution pitch tracker.

srpd is a pitch detection algorithm that produces a fundamental
frequency contour from a speech waveform. At present only the super
resolution pitch determination algorithm is implemented.  See (Medan,
Yair, and Chazan, 1991) and (Bagshaw et al., 1993) for a detailed
description of the algorithm.  </para><para>

Frames of data are read in from <parameter>sig</parameter> in
chronological order such that each frame is shifted in time from its
predecessor by <parameter>pda_frame_shift</parameter>. Each frame is
analysed in turn.

</para><para> 

The maximum and minimum signal amplitudes are initially found over the
duration of two segments, each of length N_min samples. If the sum of
their absolute values is below two times
<parameter>noise_floor</parameter>, the frame is classified as
representing silence and no coefficients are calculated. Otherwise, a
cross correlation coefficient is calculated for all n from a period in
samples corresponding to <parameter>min_pitch
</parameter> to a period in samples corresponding to
<parameter>max_pitch</parameter>, in steps
of <parameter>decimation_factor</parameter>. In calculating the
coefficient only one in <parameter>decimation_factor</parameter>
samples of the two segments are used. Such down-sampling permits rapid
estimates of the coefficients to be calculated over the range 
N_min <= n <= N_max. This results in a cross-correlation track for the
frame being analysed.  </para><para>

Local maxima of the track with a coefficient value above a specified
threshold form candidates for the fundamental period. The threshold is
adaptive and dependent upon the values <parameter>v2uv_coeff_thresh
</parameter>, <parameter>min_v2uv_coef_thresh </parameter>, and
<parameter> v2uv_coef_thresh_rati_ratio</parameter>. If the previously
analysed frame was classified as unvoiced or silent (which is the
initial state) then the threshold is set to
<parameter>v2uv_coef_thresh</parameter>. Otherwise, the previous
frame was classified as being voiced, and the threshold is set equal
to [\-r] <parameter>v2uv_coef_thresh_rati_ratio
</parameter> times the cross-correlation coefficient
value at the point of the previous fundamental period in the former
coefficients track. This product is not permitted to drop below
<parameter>v2uv_coef_thresh</parameter>.

</para><para>

If no candidates for the fundamental period are found, the frame is classified
as being unvoiced. Otherwise, the candidates are further processed to identify
the most likely true pitch period. During this additional processing, a
threshold given by <parameter>anti_doubling_thres</parameter> is used.

</para><para>

If the <parameter>peak_tracking</parameter> flag is set to true,
biasing is applied to the cross-correlation track as described in
(Bagshaw et al., 1993).  </para><para> </para><para>


@param sig: input waveform
@param op:  options regarding pitch tracking parameters
@param op.min_pitch: minimum permitted F0 value
@param op.max_pitch: maximum permitted F0 value
@param op.pda_frame_shift: analysis frame shift
@param op.pda_frame_length: analysis frame length
@param op.lpf_cutoff: cut off frequency for low pass filtering
@param op.lpf_order: order of low pass filtering (must be odd)
@param op.decimation
@param op.noise_floor
@param op.min_v2uv_coef_thresh
@param op.v2uv_coef_thresh_ratio
@param op.v2uv_coef_thresh
@param op.anti_doubling_thresh
@param op.peak_tracking

*/
void srpd(EST_Wave &sig, EST_Track &fz, EST_Features &options);

/** Smooth selected parts of an f0 contour.  Interpolation is
controlled by the <tt>speech</tt> track. When a point has a positive
value in the speech track, it is a candidate for interpolation.  
*/
void smooth_phrase(EST_Track &c, EST_Track &speech, EST_Features &options, 
		   EST_Track &sm);

/** Smooth all the points in an F0 contour*/
void smooth_portion(EST_Track &c, EST_Option &op);

//@}


/**@name  Delta and Acceleration coefficients

Produce delta and acceleration coefficients from a set of coefficients
or the waveform.
*/

//@{

/** Produce a set of delta coefficients for a track

The delta function is used to produce a set of coefficients which
estimate the rate of change of a set of parameters. The output track
<parameter>d<parameter> must be setup before hand, i.e. it must have
the same number of frames and channels as <parameter>tr</parameter>.

@param tr: input track of base coefficients
@param d: output track of delta coefficients. 
@param regression_length: number of previous frames on which delta
       estimation is calculated on.
*/

void delta(EST_Track &tr, EST_Track &d, int regression_length = 3);

/** Produce multiple sets of delta coefficients from a waveform.

  Calculate specified types of delta coefficients. This function is
  used when the base types of coefficients haven't been calculated.
  This function calls sig2coef to calculate the base types from which
  the deltas are calculated, and hence the requirements governing the
  setup of <parameter>fv</parameter> for sig2coef also hold here.

@param sig: input waveform
@param fv: output coefficients. These have been pre-allocated and the
       number of channels in a indicates the order of the analysis.
@param op: Features structure containing options for analysis order,
        frame shift etc.
@param slist: list of types of delta coefficients required.
*/

void sigpr_delta(EST_Wave &sig, EST_Track &fv, EST_Features &op, 
		const EST_StrList &slist);

/** Produce multiple sets of acceleration coefficients from a waveform

  Calculate specified types of acceleration coefficients. This function
  is used when the base types of coefficient haven't been calculated.
  This function calls sig2coef to calculate the base types from which
  the deltas are calculated, and hence the requirements governing the
  setup of <parameter>fv</parameter> for sig2coef also hold here.

@param sig: input waveform
@param fv: output coefficients. These have been pre-allocated and the
       number of channels in a indicates the order of the analysis.
@param op: Features structure containing options for analysis order,
        frame shift etc.
@param slist: list of types of acceleration coefficients required.


The delta function is used to produce a set of coefficients which
estimate the rate of change of a set of parameters. 
*/

void sigpr_acc(EST_Wave &sig, EST_Track &fv, EST_Features &op, 
		const EST_StrList &slist);

//@}

/* Convert a track containing coefficients of one type to a track
containing coefficients of another.

@param in_track input set of coefficients
@param out_track input set of coefficients
@param out_name name of desired output coefficients.
@param in_name optional: often it is possible to determine the type of 
the input coefficients from the channel names. If this is not possible or
these names should be ignored, the {\tt in_type} parameter can be used.

*/

void convert_track(EST_Track &in_track, EST_Track &out_track,
		   const EST_String &out_type, 
		   const EST_String &in_type = "");



#endif /* __EST_SIGPR_UTT_H__ */

