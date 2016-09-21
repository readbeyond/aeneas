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

#ifndef __EST_PITCHMARK_H__
#define __EST_PITCHMARK_H__

#include "EST_Wave.h"
#include "EST_Option.h"
#include "EST_Track.h"
#include "EST_TBuffer.h"

#define LX_LOW_FREQUENCY  400
#define LX_LOW_ORDER      19
#define LX_HIGH_FREQUENCY 40
#define LX_HIGH_ORDER     19
#define DF_LOW_FREQUENCY  1000
#define DF_LOW_ORDER      19
#define MIN_PERIOD        0.003
#define MAX_PERIOD        0.02
#define DEF_PERIOD        0.01
#define PM_END            -1.0

/** @name Pitchmarking Functions

Pitchmarking involves finding some pre-defined pitch related instant
for every pitch period in the speech. At present, only functions for
analysing laryngograph waveforms are available - the much harder
problem of doing this on actual speech has not been attempted.

 */
//@{

/** Find pitchmarks in Laryngograph (lx) signal. 

This high level function places a pitchmark on each positive peak in
the voiced portions of the lx signal. Pitchmarks are stored in the
time component of a EST_Track object and returned. The function works
by high and low pass filtering the signal using forward and backward
filtering to remove phase shift. The negative going points in the
smoothed differentiated signal, corresponding to peaks in the original
are then chosen.

@param lx laryngograph waveform
@param op options, mainly for filter control:
\begin{itemize}
\item {\bf lx_low_frequency} low pass cut off for lx filtering : typical value {\tt 400}
\item {\bf lx_low_order} order of low pass lx filter: typical value 19
\item {\bf lx_high_frequency} high pass cut off for lx filtering: typical value 40
\item {\bf lx_high_order} order of high pass lx filter: typical value 19
\item {\bf median_order} order of high pass lx filter: typical value 19
\end{itemize}
*/

EST_Track pitchmark(EST_Wave &lx, EST_Features &op);

/** Find pitchmarks in Laryngograph (lx) signal. The function is the
same as \Ref{pitchmark} but with more explicit control over
the parameters.

@param lx laryngograph waveform
@param lx_lf low pass cut off for lx filtering : typical value 400
@param lx_fo order of low pass lx filter : typical value 19
@param lx_hf high pass cut off for lx filtering : typical value 40
@param lx_ho  : typical value 19
@param mo order of median smoother used to smoother differentiated lx  : typical value 19

*/

EST_Track pitchmark(EST_Wave &lx, int lx_lf, int lx_lo, int lx_hf, 
		    int lx_ho, int df_lf, int df_lo, int mo, int debug=0);


/** Find times where waveform cross zero axis in negative direction.

@param sig waveform
@param pm pitchmark track which stores time positions of negative crossings
*/

void neg_zero_cross_pick(EST_Wave &lx, EST_Track &pm);

/** Produce a set of sensible pitchmarks. 

Given a set of raw pitchmarks, this function makes sure no pitch
period is shorter that {\tt min} seconds and no longer than {\tt max}
seconds. Periods that are too short are eliminated. If a period is too
long, extra pitchmarks are inserted whose period is {\it
approximately} {\tt def} seconds in duration. The approximation is to
ensure that the pitch period in the interval, D, is constant, and so
the actual pitch period is given by \[T = D / floor(D/def)\] */

void pm_fill(EST_Track &pm, float new_end, float max, float min, float def);

/** Remove pitchmarks which are too close together. 

This doesn't work in a particularly sophisticated way, in that it
removes a sequence of too close pitchmarks left to right, and doesn't
attempt to find which ones in the sequence are actually spurious.  */

void pm_min_check(EST_Track &pm, float min);


void pm_to_f0(EST_Track &pm, EST_Track &f0);

// for constant shift pitchmarks
void pm_to_f0(EST_Track &pm, EST_Track &fz, float shift);


//@}

#endif /* __EST_PITCHMARK_H__ */


//@}
