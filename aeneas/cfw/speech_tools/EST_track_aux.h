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


/**@name EST_track_aux.h
  * EST_Track Auxiliary functions
  * @author Paul Taylor <pault@cstr.ed.ac.uk>
  * @version $Id: EST_track_aux.h,v 1.4 2004/05/24 11:15:51 korin Exp $
  */

//@{

#ifndef __EST_TRACK_AUX_H__
#define __EST_TRACK_AUX_H__

#include "EST_FMatrix.h"
#include "EST_TList.h"
#include "ling_class/EST_Relation.h"
#include "EST_Option.h"
#include "EST_Track.h"
#include "EST_TBuffer.h"

void track_smooth(EST_Track &c, float x, EST_String stype = "");
void time_med_smooth(EST_Track &c, float x);
void time_mean_smooth(EST_Track &c, float x);
void simple_med_smooth(EST_Track &c, int n, int channel=0);
void simple_mean_smooth(EST_Track &c, int n, int channel=0);

/** Calculate the mean absolute error between the same channel in
  * two tracks. This is given by \[\frac{1}{n}\sum_{i=1}^{n}|a_{i} - b_{i}|\]
  * @see abs_error, rms_error(EST_Track &a, EST_Track &b)
  */

float abs_error(EST_Track &a, EST_Track &b, int channel);

void absolute(EST_Track &tr);
void normalise(EST_Track &tr);
void normalise(EST_Track &tr, float mean, float sd, int channel, 
	       float upper, float lower);
void normalise(EST_Track &tr, EST_FVector &mean, EST_FVector &sd,
	       float upper, float lower);
void normalise(EST_TrackList &trlist, EST_FVector &mean, 
	       EST_FVector &sd, float upper, float lower);

/** Calculate the simple derivative of a track. This is given by
  * \[a_{i+1} - a_{i}\] The values in the resultant track are spaced
  * midway between the values in the input track, resulting in 1 fewer
  * frames in the track. This is a very local estimation of the derivative
  * of the track at a point in time. A smoother value can be obtained
  * using the delta function. 
  * @see delta
  */

EST_Track differentiate(EST_Track &c, float samp_int=0.0);
EST_Track difference(EST_Track &a, EST_Track &b);

float mean( const EST_Track &a, int channel );
void mean( const EST_Track &a, EST_FVector &m );

void meansd(EST_Track &a, float &m, float &sd, int channel);

/** Calculate the root mean square error between the same channel in
  * two tracks. The channel is identified by its index.
  * @see abs_error, float rms_error(EST_Track &a, EST_Track &b)
  */
float rms_error(EST_Track &a, EST_Track &b, int channel);

float correlation(EST_Track &a, EST_Track &b, int channel);

void meansd(EST_Track &a, EST_FVector &m, EST_FVector &sd);

/** Calculate the root mean square error between each channels in two
  * tracks. For two tracks of M channels, the result is returned as an
  * EST_FVector of size M, with element {\it i} representing the
  * rms error for channel {\it i}. 
  * @see abs_error, rms_error 
  */ 
EST_FVector rms_error(EST_Track &a, EST_Track &b); 

EST_FVector abs_error(EST_Track &a, EST_Track &b); EST_FVector
correlation(EST_Track &a, EST_Track &b);

/// Move the start and end variables to the nearest frame.
void align_to_track(EST_Track &tr, float &start, float &end);
/// Move the start and end variables to the nearest frame.
void align_to_track(EST_Track &tr, int &start, int &end, int sample_rate);
/// Move the start and end variables to the start and end of the nearest frame.
void move_to_frame_ends(EST_Track &tr, 
			int &start, int &end, 
			int sample_rate, float offset=0.0);
/// Index of the frame whose start boundary
int nearest_boundary(EST_Track &tr, float time, int sample_rate, float offset=0);

/// Move the track so that it starts at the indicated time.
void set_start(EST_Track &tr, float start);
/// Move the track by {\it shift} seconds
void move_start(EST_Track &tr, float shift);

EST_Track error(EST_Track &ref, EST_Track &test, int relax= 0);
void extract(EST_Track &orig, float start, float end, EST_Track &res);

int track_divide(EST_TrackList &mtfr, EST_Track &fv, EST_Relation &key);
void ParallelTracks(EST_Track &a, EST_TrackList &list,const EST_String &style);
void track_info(EST_Track &track);

EST_String options_track_filetypes(void);
EST_String options_track_filetypes_long(void);
EST_String options_subtrack(void);

int read_track(EST_Track &tr, const EST_String &in_file, EST_Option &al);

/** Return the frame size in {\bf seconds} based on analysis of
current time points.  This function basically determines the local
frame size (shift) by subtracting the current time point from the next
time point. If the {\tt prefer_prev} flag is set to {\tt true}, or the
index is the last in the track, the size is determined by subtracting
the previous time point from the current one.

This is most commonly used in pitch synchronous analysis to determine
the local pitch period.

@see get_frame_size
*/

float get_time_frame_size(EST_Track &pms, int i, int prefer_prev = 0);

/** Return the frame size in {\bf samples} based on analysis of
current time points.  This function basically determines the local
frame size (shift) by subtracting the current time point from the next
time point. If the {\tt prefer_prev} flag is set to {\tt true}, or the
index is the last in the track, the size is determined by subtracting
the previous time point from the current one.

This is most commonly used in pitch synchronous analysis to determine
the local pitch period.

@see get_time_frame_size
*/
int get_frame_size(EST_Track &pms, int current_pos, int sample_rate, 
			 int prefer_prev=0);


/// How many coefficients in track (looks for Coef0 and coefN channels)
int get_order(const EST_Track &t, EST_CoefficientType type, int d=0);
int get_order(const EST_Track &t);

/// Total the length channel values.
int sum_lengths(const EST_Track &t, 
		int sample_rate,
		int start_frame=0, int end_frame=-1);

/// Find the start point in the signal of the sections of speech related to each frame.
void get_start_positions(const EST_Track &t, 
			 int sample_rate,
			 EST_TBuffer<int> &pos);

/**@name Analysis frame position
  * Functions which define which part of a single is associated with a
  * given frame in a track. 
  * <p>
  * This is defined here in one place for consistency. They are inline since 
  * they tend to be used in inner loops. There are two versions,
  * the second for when there are offsets in the track.
  */
//@{

/// Get the start and end of a given frame (in samples)
static inline void get_frame(const EST_Track &tr, int sample_rate,
			     int f,
			     int &start, int &center, int &end)
{
  center = (int)(tr.t(f)*sample_rate + 0.5);
  start  = center - (int)(tr.a(f, channel_length)/2.0);
  end    = start + (int)(tr.a(f, channel_length));
}

/// Get the start and end of a given frame (in seconds)
static inline void get_frame(const EST_Track &tr, int sample_rate,
			     int f,
			     float &start, float &center, float &end)
{
  center = tr.t(f);
  start  = center - tr.a(f, channel_length)/(float)sample_rate/2.0;
  end    = start + tr.a(f, channel_length)/(float)sample_rate;
}

/// Get the start and end of a given frame (in samples)
static inline void get_frame_o(const EST_Track &tr, int sample_rate,
			       int f,
			       int &start, int &center, int &end)
{
  center = (int)(tr.t(f)*sample_rate + tr.a(f,channel_offset) + 0.5);
  start  = center - (int)(tr.a(f, channel_length)/2.0);
  end    = start + (int)(tr.a(f, channel_length));
}

/// Get the start and end of a given frame (in seconds)
static inline void get_frame_o(const EST_Track &tr, int sample_rate,
			       int f,
			       float &start, float &center, float &end)
{
  center = tr.t(f) + tr.a(f,channel_offset)/(float)sample_rate;
  start  = center - tr.a(f, channel_length)/(float)sample_rate/2.0;
  end    = start + tr.a(f, channel_length)/(float)sample_rate;
}

//@}

// take one of the channels as the timeline
void channel_to_time(EST_Track &tr, int channel, float scale=1.0);
void channel_to_time(EST_Track &tr, EST_ChannelType c,float  scale=1.0);
void channel_to_time(EST_Track &tr, const EST_String c_name, float scale=1.0);

void channel_to_time_lengths(EST_Track &tr, int channel, float scale=1.0);
void channel_to_time_lengths(EST_Track &tr, EST_ChannelType c,float  scale=1.0);
void channel_to_time_lengths(EST_Track &tr, const EST_String c_name, float scale=1.0);

/* Allow EST_Track to be used in an EST_Val */
VAL_REGISTER_CLASS_DCLS(track,EST_Track)

#endif /* __EST_TRACK_AUX_H__ */
//@}
