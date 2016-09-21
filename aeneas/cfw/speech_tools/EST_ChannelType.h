 /************************************************************************/
 /*                                                                      */
 /*                Centre for Speech Technology Research                 */
 /*                     University of Edinburgh, UK                      */
 /*                       Copyright (c) 1996,1997                        */
 /*                        All Rights Reserved.                          */
 /*                                                                      */
 /*  Permission is hereby granted, free of charge, to use and distribute */
 /*  this software and its documentation without restriction, including  */
 /*  without limitation the rights to use, copy, modify, merge, publish, */
 /*  distribute, sublicense, and/or sell copies of this work, and to     */
 /*  permit persons to whom this work is furnished to do so, subject to  */
 /*  the following conditions:                                           */
 /*   1. The code must retain the above copyright notice, this list of   */
 /*      conditions and the following disclaimer.                        */
 /*   2. Any modifications must be clearly marked as such.               */
 /*   3. Original authors' names are not deleted.                        */
 /*   4. The authors' names are not used to endorse or promote products  */
 /*      derived from this software without specific prior written       */
 /*      permission.                                                     */
 /*                                                                      */
 /*  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK       */
 /*  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING     */
 /*  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT  */
 /*  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE    */
 /*  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES   */
 /*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN  */
 /*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,         */
 /*  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF      */
 /*  THIS SOFTWARE.                                                      */
 /*                                                                      */
 /************************************************************************/

#ifndef __EST_CHANNELTYPE_H__
#define __EST_CHANNELTYPE_H__

/**@name Channel Types
  */
//@{

/** Symbolic names for coefficient types.
  *
  * Used to record what kinds of information are in a track and
  * anywhere else we need to refer to coefficient types.
  *
  * @see EST_ChannelType
  */

enum EST_CoefficientType
{
  /// Linear prediction filter
  cot_lpc=0,
  /// guaranteed to be the first known type
  cot_first=cot_lpc,
  /// reflection coefficients.
  cot_reflection,
  /// Cepstral coefficients
  cot_cepstrum,
  /// Mel Scale Cepstrum
  cot_melcepstrum,
  /// Mel Scale filter bank
  cot_fbank,
  /// Line spectral pairs.
  cot_lsf,
  /// Tube areas for filter.
  cot_tubearea,
  /// Unknown filter type.
  cot_filter,
  /// Free for experimentation
  cot_user1,
  /// Free for experimentation
  cot_user2,
  /// Guaranteed to be one more than last legal coefficient type
  cot_free
};

/**@name Channel Type Numbering Scheme
  *
  * Channel types are given numbers containing the following information:
  * \begin{itemize}
  *	\item A numeric index.
  *	\item A Number of differentiations 0-2
  *	\item 0 for start, 1 for end
  * \end{itemize}
  * Things which do not require all these features are packed in according
  * to the following rules:
  * \begin{itemize}
  *	\item Single values which can be differentiated are paired as
  *		if they were start and end positions of an unknown type
  *		of coefficient.
  *	\item Single values which can't be differentiated are put in the
  *		positions where the 3rd derivatives would logically be
  *		found.
  * \end{itemize}
  */
//@{

/// extract the coefficient type
#define EST_ChannelTypeCT(T) ( (T) >> 3 )
/// extract the number of differentiations
#define EST_ChannelTypeD(T) ( (T) >> 1 & 3 )
/// extract the start/end flag.
#define EST_ChannelTypeSE(T) ( (T) & 1 )

/// get start from end
#define EST_ChannelTypeStart(T) EST_CoefChannelId(\
						 EST_ChannelTypeCT(T),  \
						 EST_ChannelTypeD(T), \
						 0)
/// get end from start
#define EST_ChannelTypeEnd(T) EST_CoefChannelId(\
						 EST_ChannelTypeCT(T),  \
						 EST_ChannelTypeD(T), \
						 1)
/// differentiate once
#define EST_ChannelTypeIncD(T) EST_CoefChannelId(\
						 EST_ChannelTypeCT(T),  \
						 EST_ChannelTypeD(T)+1, \
						 EST_ChannelTypeSE(T))
/// differentiate N times
#define EST_ChannelTypeDelta(T, N) EST_CoefChannelId(\
						 EST_ChannelTypeCT(T),  \
						 EST_ChannelTypeD(T)+(N), \
						 EST_ChannelTypeSE(T))
/// integrate once
#define EST_ChannelTypeDecD(T) EST_CoefChannelId(\
						 EST_ChannelTypeCT(T),  \
						 EST_ChannelTypeD(T)-1, \
						 EST_ChannelTypeSE(T))


/** Build a number representing a channel type for a coefficient type.
  * 
  * CT = coefficient type
  * D  = Number of levels of differentiation.
  * SE = Start=0 end=1
  */
#define EST_CoefChannelId(CT,D,SE) ( (CT)<<3 | ((D)<<1 & 6) | ((SE)&1) )

/** Build a number representing a channel type for a single value which can
  * N = count starting from 0
  * D  = Number of levels of differentiation.
  * be differentiated.
  */

#define EST_DiffChannelId(N,D) ( EST_CoefChannelId(((N)>>1)+(int)cot_free, D, (N)&1) )

/** Build a number representing a channel type for a simple value
  * such as length or voicing probability.
  */

#define EST_ChannelId(N) EST_CoefChannelId((N)>>1, 3, (N)&1)
//@}


/** Symbolic names for track channels.
  * Used in track maps to label channels so they can be accessed without
  * knowing exactly where in the track they are.
  *
  * @see EST_CoefficientType
  * @see EST_TrackMap
  * @see EST_Track
  * @see EST_TrackMap:example
  * @author Richard Caley <rjc@cstr.ed.ac.uk>
  * @version $Id: EST_ChannelType.h,v 1.4 2009/07/03 17:13:56 awb Exp $
  */

enum EST_ChannelType {
  /// Value to return for errors, never occurs in TrackMaps
  channel_unknown	=  EST_ChannelId(0),
  /// order of analysis.
  channel_order		= EST_ChannelId(1),
  /// So we know how many there are
  first_channel_type=channel_order,
  /// Peak amplitude.
  channel_peak		= EST_ChannelId(2),		
  /// Duration of section of signal.
  channel_duration	= EST_ChannelId(3),	
  /// Length of section in samples.
  channel_length	= EST_ChannelId(4),	
  /// Offset from frame center to center of window
  channel_offset	= EST_ChannelId(5),	
  /// Voicing decision.
  channel_voiced	= EST_ChannelId(6),	
  /// Number of related frame in another track.
  channel_frame		= EST_ChannelId(7),	
  /// Time in seconds this frame refers to.
  channel_time		= EST_ChannelId(8),		

  /// RMS power of section of signal.
  channel_power		= EST_DiffChannelId(0,0),	
  channel_power_d	= EST_DiffChannelId(0,1),	
  channel_power_a	= EST_DiffChannelId(0,2),	
  /// RMS energy of section of signal.
  channel_energy	= EST_DiffChannelId(1,0),	
  channel_energy_d	= EST_DiffChannelId(1,1),	
  channel_energy_a	= EST_DiffChannelId(1,2),	
  /// F0 in Hz.
  channel_f0		= EST_DiffChannelId(2,0),	
  channel_f0_d		= EST_DiffChannelId(2,1),	
  channel_f0_a		= EST_DiffChannelId(2,2),	

  channel_lpc_0			= EST_CoefChannelId(cot_lpc,0,0),
  channel_lpc_N			= EST_CoefChannelId(cot_lpc,0,1),
  channel_lpc_d_0		= EST_CoefChannelId(cot_lpc,1,0),
  channel_lpc_d_N		= EST_CoefChannelId(cot_lpc,1,1),
  channel_lpc_a_0		= EST_CoefChannelId(cot_lpc,2,0),
  channel_lpc_a_N		= EST_CoefChannelId(cot_lpc,2,1),

  channel_reflection_0		= EST_CoefChannelId(cot_reflection,0,0),
  channel_reflection_N		= EST_CoefChannelId(cot_reflection,0,1),
  channel_reflection_d_0	= EST_CoefChannelId(cot_reflection,1,0),
  channel_reflection_d_N	= EST_CoefChannelId(cot_reflection,1,1),
  channel_reflection_a_0	= EST_CoefChannelId(cot_reflection,2,0),
  channel_reflection_a_N	= EST_CoefChannelId(cot_reflection,2,1),


  channel_cepstrum_0		= EST_CoefChannelId(cot_cepstrum,0,0),
  channel_cepstrum_N		= EST_CoefChannelId(cot_cepstrum,0,1),
  channel_cepstrum_d_0		= EST_CoefChannelId(cot_cepstrum,1,0),
  channel_cepstrum_d_N		= EST_CoefChannelId(cot_cepstrum,1,1),
  channel_cepstrum_a_0		= EST_CoefChannelId(cot_cepstrum,2,0),
  channel_cepstrum_a_N		= EST_CoefChannelId(cot_cepstrum,2,1),


  channel_melcepstrum_0		= EST_CoefChannelId(cot_melcepstrum,0,0),
  channel_melcepstrum_N		= EST_CoefChannelId(cot_melcepstrum,0,1),
  channel_melcepstrum_d_0	= EST_CoefChannelId(cot_melcepstrum,1,0),
  channel_melcepstrum_d_N	= EST_CoefChannelId(cot_melcepstrum,1,1),
  channel_melcepstrum_a_0	= EST_CoefChannelId(cot_melcepstrum,2,0),
  channel_melcepstrum_a_N	= EST_CoefChannelId(cot_melcepstrum,2,1),

  channel_fbank_0		= EST_CoefChannelId(cot_fbank,0,0),
  channel_fbank_N		= EST_CoefChannelId(cot_fbank,0,1),
  channel_fbank_d_0		= EST_CoefChannelId(cot_fbank,1,0),
  channel_fbank_d_N		= EST_CoefChannelId(cot_fbank,1,1),
  channel_fbank_a_0		= EST_CoefChannelId(cot_fbank,2,0),
  channel_fbank_a_N		= EST_CoefChannelId(cot_fbank,2,1),


  channel_lsf_0			= EST_CoefChannelId(cot_lsf,0,0),
  channel_lsf_N			= EST_CoefChannelId(cot_lsf,0,1),
  channel_lsf_d_0		= EST_CoefChannelId(cot_lsf,1,0),
  channel_lsf_d_N		= EST_CoefChannelId(cot_lsf,1,1),
  channel_lsf_a_0		= EST_CoefChannelId(cot_lsf,2,0),
  channel_lsf_a_N		= EST_CoefChannelId(cot_lsf,2,1),


  channel_tubearea_0		= EST_CoefChannelId(cot_tubearea,0,0),
  channel_tubearea_N		= EST_CoefChannelId(cot_tubearea,0,1),
  channel_tubearea_d_0		= EST_CoefChannelId(cot_tubearea,1,0),
  channel_tubearea_d_N		= EST_CoefChannelId(cot_tubearea,1,1),
  channel_tubearea_a_0		= EST_CoefChannelId(cot_tubearea,2,0),
  channel_tubearea_a_N		= EST_CoefChannelId(cot_tubearea,2,1),


  channel_filter_0		= EST_CoefChannelId(cot_filter,0,0),
  channel_filter_N		= EST_CoefChannelId(cot_filter,0,1),
  channel_filter_d_0		= EST_CoefChannelId(cot_filter,1,0),
  channel_filter_d_N		= EST_CoefChannelId(cot_filter,1,1),
  channel_filter_a_0		= EST_CoefChannelId(cot_filter,2,0),
  channel_filter_a_N		= EST_CoefChannelId(cot_filter,2,1),


  channel_user1_0		= EST_CoefChannelId(cot_user1,0,0),
  channel_user1_N		= EST_CoefChannelId(cot_user1,0,1),
  channel_user1_d_0		= EST_CoefChannelId(cot_user1,1,0),
  channel_user1_d_N		= EST_CoefChannelId(cot_user1,1,1),
  channel_user1_a_0		= EST_CoefChannelId(cot_user1,2,0),
  channel_user1_a_N		= EST_CoefChannelId(cot_user1,2,1),


  channel_user2_0		= EST_CoefChannelId(cot_user2,0,0),
  channel_user2_N		= EST_CoefChannelId(cot_user2,0,1),
  channel_user2_d_0		= EST_CoefChannelId(cot_user2,1,0),
  channel_user2_d_N		= EST_CoefChannelId(cot_user2,1,1),
  channel_user2_a_0		= EST_CoefChannelId(cot_user2,2,0),
  channel_user2_a_N		= EST_CoefChannelId(cot_user2,2,1),



  last_channel_type = channel_f0_a,
  /// Can be used to size arrays etc.
  num_channel_types 
};
//@}
typedef enum EST_ChannelType EST_ChannelType;

#endif
