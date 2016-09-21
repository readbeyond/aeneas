 /*************************************************************************/
 /*                                                                       */
 /*                Centre for Speech Technology Research                  */
 /*                     University of Edinburgh, UK                       */
 /*                       Copyright (c) 1995,1996                         */
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
 /*                                                                       */
 /*                   Author :  Paul Taylor                               */
 /*                Rewritten :  Richard Caley                             */
 /* -------------------------------------------------------------------   */
 /*                  EST_Track Class header file                          */
 /*                                                                       */
 /*************************************************************************/

class EST_Track;

#ifndef __Track_H__
#define __Track_H__

#include "EST_FMatrix.h"
#include "EST_types.h"
#include "EST_TrackMap.h"
#include "EST_ChannelType.h"
#include "EST_Featured.h"

typedef EST_TMatrix<EST_Val> EST_ValMatrix;

class EST_TokenStream;
class EST_String;

typedef enum EST_TrackFileType {
  tff_none=0,
  tff_ascii,
  tff_esps,
  tff_htk,
  tff_htk_fbank,
  tff_htk_mfcc,
  tff_htk_mfcc_e,
  tff_htk_user,
  tff_htk_discrete,
  tff_xmg,
  tff_xgraph,
  tff_ema,
  tff_ema_swapped,
  tff_NIST,
  tff_est_ascii,
  tff_est_binary,
  tff_snns,
  tff_ssff
} EST_TrackFileType;

typedef enum EST_InterpType {
  it_nearest,			// nearest time point
  it_linear,			// linerar interpolation
  it_linear_nz			// .. unless one end near zero
} EST_InterpType;

/** A class for storing time aligned coefficients.

some stuff.
*/

class EST_Track : public EST_Featured {

protected:
    EST_FMatrix p_values;		// float x array 
    EST_FVector p_times;	        // float y array 
    EST_CVector p_is_val;		// for breaks and non-breaks

    EST_ValMatrix p_aux;		// Auxiliary channels
    EST_StrVector p_aux_names;		// Names of auxiliary channels

    float p_t_offset;			// time shift.
    
    EST_TrackMap::P p_map;
    EST_StrVector p_channel_names;	// name of each track

    bool p_equal_space;			// fixed or variable frame rate
    bool p_single_break;		// single break lots between data 

    void default_vals();
    void default_channel_names();
    void clear_arrays();
    void pad_breaks();		     // put in extra breaks 
    
    int interp_value(float x, float f);
    float interp_amp(float x, int c, float f);
    float estimate_shift(float x);
    void copy(const EST_Track& a);

public:
    static const float default_frame_shift;
    static const int default_sample_rate;

    /**@name Constructor and Destructor functions
     */

    //@{

    /// Default constructor
    EST_Track();

    /// Copy constructor
    EST_Track(const EST_Track &a);

    /// resizing constructor
    EST_Track(int num_frames, int num_channels);

    /// resizing constructor
    EST_Track(int num_frames, EST_StrList &map);

    /// default destructor
    ~EST_Track();
    //@}

    /** @name Configuring Tracks
     */
    //@{
    
    /** resize the track to have {\tt num_frames} and {\tt num_channels}.
	if {\tt preserve} is set to 1, any existing values in the track
	are kept, up to the limits imposed by the new number of frames
	and channels. If the new track size is bigger, new positions
	are filled with 0 */
    void resize(int num_frames, int num_channels, bool preserve = 1);

    /** resize the track to have {\tt num_frames} and {\tt num_channels}.
	if {\tt preserve} is set to 1, any existing values in the track
	are kept, up to the limits imposed by the new number of frames
	and channels. If the new track size is bigger, new positions
	are filled with 0 */
    void resize(int num_frames, EST_StrList &map, bool preserve = 1);

    /** resize the track's auxiliary channels.
     */
    void resize_aux(EST_StrList &map, bool preserve = 1);

    /** Change the number of channels while keeping the number of
	frames the same.  if {\tt preserve} is set to 1, any existing
	values in the track are kept, up to the limits imposed by the
	new number of frames and channels. If the new track size is
	bigger, new positions are filled with 0 */
    void set_num_channels(int n, bool preserve = 1) 
	{ resize(EST_CURRENT, n, preserve); }

    /** Change the number of frames while keeping the number of
	channels the same.  if {\tt preserve} is set to 1, any
	existing values in the track are kept, up to the limits
	imposed by the new number of frames and channels. If the new
	track size is bigger, new positions are filled with 0 */
    void set_num_frames(int n, bool preserve = 1)
	{ resize(n, EST_CURRENT, preserve); }

    /// set the name of the channel.
    void set_channel_name(const EST_String &name, int channel);

    /// set the name of the auxiliary channel.
    void set_aux_channel_name(const EST_String &name, int channel);

    /// copy everything but data

    void copy_setup(const EST_Track& a); 
    //@}

    /**@name Global track information
     */
    //@{
    /// name of track - redundant use access to features
    EST_String name() const 
       { return f_String("name");}
    /// set name of track - redundant use access to features
    void set_name(const EST_String &n) 
	{f_set("name",n);}

    //@}

    /**@name Functions for sub tracks, channels and frames.
     */
    //@{

    /** make {\tt fv} a window to frame {\tt n} in the track. 
     */
    void frame(EST_FVector &fv, int n, int startf=0, int nf=EST_ALL)
	{ p_values.row(fv, n, startf, nf); }

    /** make {\tt fv} a window to channel {\tt n} in the track. 
     */
    void channel(EST_FVector &cv, int n, int startf=0, int nf=EST_ALL)
	{ p_values.column(cv, n, startf, nf); }

    /** make {\tt fv} a window to the named channel in the track. 
     */
    void channel(EST_FVector &cv, const char * name, int startf=0, 
		 int nf=EST_ALL);

    /** make {\tt st} refer to a portion of the track. No values
	are copied - an internal pointer in st is set to the specified
	portion of the the track. After this, st behaves like a normal
	track. Its first channel is the same as start_channel and its
	first frame is the same as start_frame. Any values written into
	st will changes values in the main track. st cannot be resized.

	@param start_frame first frame at which sub-track starts
	@param nframes number of frames to be included in total
	@param start_channel first channel at which sub-track starts
	@param nframes number of channels to be included in total
    */
    void sub_track(EST_Track &st,
	         int start_frame=0, int nframes=EST_ALL,
		   int start_chan=0, int nchans=EST_ALL);

    /** make {\tt st} refer to a portion of the track. No values
	are copied - an internal pointer in st is set to the specified
	portion of the the track. After this, st behaves like a normal
	track. Its first channel is the same as start_channel and its
	first frame is the same as start_frame. Any values written into
	st will changes values in the main track. st cannot be resized.

	@param start_frame first frame at which sub-track starts
	@param nframes number of frames to be included in total
	@param start_channel_name name of channel at which sub-track starts
	@param end_channel_name name of channel at which sub-track stops
    */
    void sub_track(EST_Track &st,
	         int start_frame, int nframes,
		   const EST_String &start_chan_name,
		   int nchans=EST_ALL);

    /** make {\tt st} refer to a portion of the track. No values
	are copied - an internal pointer in st is set to the specified
	portion of the the track. After this, st behaves like a normal
	track. Its first channel is the same as start_channel and its
	first frame is the same as start_frame. Any values written into
	st will changes values in the main track. st cannot be resized.

	@param start_frame first frame at which sub-track starts
	@param nframes number of frames to be included in total
	@param start_channel_name name of channel at which sub-track starts
	@param end_channel_name name of channel at which sub-track stops
    */
    void sub_track(EST_Track &st,
	         int start_frame, int nframes,
		   const EST_String &start_chan_name,
		   const EST_String &end_chan_name);

    /** make {\tt st} refer to a portion of the track. (const version)
	No values
	are copied - an internal pointer in st is set to the specified
	portion of the the track. After this, st behaves like a normal
	track. Its first channel is the same as start_channel and its
	first frame is the same as start_frame. Any values written into
	st will changes values in the main track. st cannot be resized.

	@param start_frame first frame at which sub-track starts
	@param nframes number of frames to be included in total
	@param start_channel first channel at which sub-track starts
	@param nframes number of channels to be included in total
    */
    void sub_track(EST_Track &st,
		   int start_frame=0, int nframes=EST_ALL,
		   int start_chan=0, int nchans=EST_ALL) const
	{ ((EST_Track *)this)->sub_track(st, start_frame, nframes, 
					 start_chan, nchans); } 

    /** Copy contiguous portion of track into {\tt st}. Unlike the
	normal sub_track functions, this makes a completely new track.
	values written into this will not affect the main track and it
	can be resized.

	@param start_frame first frame at which sub-track starts
	@param nframes number of frames to be included in total
	@param start_channel first channel at which sub-track starts
	@param nframes number of channels to be included in total
    */

    void copy_sub_track( EST_Track &st,
			 int start_frame=0, int nframes=EST_ALL,
			 int start_chan=0, int nchans=EST_ALL) const;

    void copy_sub_track_out( EST_Track &st, const EST_FVector& frame_times ) const;
    void copy_sub_track_out( EST_Track &st, const EST_IVector& frame_indices ) const;

    /** copy channel {\tt n} into pre-allocated buffer buf */
    void copy_channel_out(int n, float *buf, int offset=0, int num=EST_ALL)
	const
	{ p_values.copy_column(n, buf, offset, num); } 

    /** copy channel {\tt n} into EST_FVector */
    void copy_channel_out(int n, EST_FVector &f, int offset=0, int num=EST_ALL)
	const
	{ p_values.copy_column(n, f, offset, num); } 

    /** copy frame {\tt n} into pre-allocated buffer buf */
    void copy_frame_out(int n, float *buf, int offset=0, int num=EST_ALL) 
	const {p_values.copy_row(n, buf, offset, num); } 

    /** copy frame {\tt n} into EST_FVector */
    void copy_frame_out(int n, EST_FVector &f, int offset=0, int num=EST_ALL)
        const {p_values.copy_row(n, f, offset, num); } 

    /** copy buf into pre-allocated channel n of track */
    void copy_channel_in(int n, const float *buf, int offset=0, 
			  int num=EST_ALL)
	{ p_values.set_column(n, buf, offset, num); }

    /** copy f into pre-allocated channel n of track */
    void copy_channel_in(int n, const EST_FVector &f, int offset=0, 
			 int num=EST_ALL)
      { p_values.set_column(n, f, offset, num); }

    /** copy channel buf into pre-allocated channel n of track */
    void copy_channel_in(int c, 
		     const EST_Track &from, int from_c, int from_offset=0,
		     int offset=0, int num=EST_ALL)
	{ p_values.set_column(c, from.p_values, from_c, 
			      from_offset, offset, num); }

    /** copy buf into frame n of track */
    void copy_frame_in(int n, const float *buf, int offset=0, 
		       int num=EST_ALL)
	{ p_values.set_row(n, buf, offset, num); }

    /** copy t into frame n of track */
    void copy_frame_in(int n, const EST_FVector &t, int offset=0, 
		   int num=EST_ALL)
	{ p_values.set_row(n, t, offset, num); }

    /** copy from into frame n of track */
    void copy_frame_in(int i, 
		   const EST_Track &from, int from_f, int from_offset=0, 
		   int offset=0, int num=EST_ALL)
	{ p_values.set_row(i, from.p_values, from_f, from_offset, offset, 
			   num); }

    //@}

    /**@name Channel information
     */

    //@{

    /** Return the position of channel {\tt name} if it exists,
	otherwise return -1.
    */
    int channel_position(const char *name, int offset=0) const;

    /** Return the position of channel {\tt name} if it exists,
	otherwise return -1.
    */
    int channel_position(EST_String name, int offset=0) const
	{ return  channel_position((const char *)name, offset); }


    /** Returns true if the track has a channel named {\tt name}, 
	otherwise  false.
    */

    bool has_channel(const char *name) const
	{ return channel_position(name) >=0; }

    /** Returns true if the track has a channel named {\tt name}, 
	otherwise  false.
    */
    bool has_channel(EST_String name) const 
	{ return has_channel((const char *)name); }

    //@}

    /** @name Accessing amplitudes The following functions can be used
	to access to amplitude of the track at certain points. Most of
	these functions can be used for reading or writing to this
	point, thus

	tr.a(10, 5) = 10.3;

	can be used to set the 10th frame of the 5th channel and

	cout << tr.a(10, 5);

	can be used to print the same information. Most of these functions
	have a const equivalent for helping the compiler in
	read only operations.
    */

    //@{

    /** return amplitude of frame i, channel c.*/
    float &a(int i, int c=0);
    float a(int i, int c=0) const;

    /** return amplitude of frame i, channel c with no bounds
	checking. */
    float &a_no_check(int i, int c=0) { return p_values.a_no_check(i,c); }
    float a_no_check(int i, int c=0) const {return p_values.a_no_check(i,c);}

    /** return amplitude of point i, in the channel named name plus
	offset. If you have a track with say channels called F0 and
	voicing, you can access the 45th frame's F0 as t.a(45, "F0");
	If there are 20 cepstral coefficients for each frame, the 5th can
	be accessed as t.a(45, "cepstrum", 5);
     */

    float &a(int i, const char *name, int offset=0);

    float  a(int i, const char *name, int offset=0) const
	{ return ((EST_Track *)this)->a(i, name, offset); }
    float &a(int i, EST_String name, int offset=0) 
	{ return a(i, (const char *)name, offset); }
    float  a(int i, EST_String name, int offset=0) const
	{ return ((EST_Track *)this)->a(i, (const char *)name, offset); }

    /** return amplitude of time t, channel c. This can be used for
	reading or writing to this point. By default the nearest frame
	to this time is used. If {\tt interp} is set to {\tt
	it_linear}, linear interpolation is performed between the two
	amplitudes of the two frames either side of the time point to
	give an estimation of what the amplitude would have been at
	time {\tt t}.  If {\tt interp} is set to {\tt it_linear_nz},
	interpolation is as above, unless the time requested is off
	the end of a portion of track in which case the nearest
	amplitude is returned.
    */
    float &a(float t, int c=0, EST_InterpType interp=it_nearest);	
    float  a(float t, int c=0, EST_InterpType interp=it_nearest) const
	{ return ((EST_Track *)this)->a(t, c, interp); }	


    /** return amplitude of frame i, channel c. */
    float &operator() (int i, int c)       { return a(i,c); }	
    /** return amplitude of frame i, channel 0. */
    float &operator() (int i)              { return a(i,0); }
    float  operator() (int i, int c) const { return a(i,c); }	
    float  operator() (int i) const        { return a(i,0); }	
  
    /** return amplitude of frame nearest time t, channel c. */
    float &operator() (float t, int c)       {return a(t,c); }
    /** return amplitude of frame nearest time t, channel 0. */
    float &operator() (float t)              {return a(t,0); }
    float  operator() (float t, int c) const {return a(t,c); }
    float  operator() (float t) const        {return a(t,0); }

    //@}

    /** @name Timing

     */

    //@{

    /// return time position of frame i
    float &t(int i=0)			   { return p_times[i]; }
    float  t(int i=0) const                    { return p_times(i); } 

    /// return time of frame i in milli-seconds.
    float ms_t(int i) const		   { return p_times(i) * 1000.0; }

    /** set frame times to regular intervals of time {\tt t}.
	The {\tt start} parameter specifies the integer multiple of {\tt t} at
	which to start.  For example, setting this to 0 will start at time
	0.0 (The default means the first time value = {\tt t}
     */
    void fill_time( float t, int start=1 );

    /** set frame times to regular intervals of time {\tt t}.
	The {\tt start} parameter specifies the first time value.
     */
    void fill_time( float t, float start );

    /** fill time channel with times from another track
     */
    void fill_time( const EST_Track &t );

    /** fill all amplitudes with value {\tt v} */
    void fill(float v) { p_values.fill(v); }

    /** resample track at this frame shift, specified in seconds. 
	This can be used to change a variable frame spaced track into
	a fixed frame track, or to change the spacing of an existing
	evenly spaced track.
     */
    void sample(float shift);

    /// REDO
    void change_type(float nshift, bool single_break);

    /** return an estimation of the frame spacing in seconds. 
	This returns -1 if the track is not a fixed shift track */
    float shift() const;
    /// return time of first value in track
    float start() const;
    /// return time of last value in track
    float end() const;
    //@}

    /** @name Auxiliary channels

	Auxiliary information is used to store information that goes
	along with frames, but which are not amplitudes and hence
	not appropriate for operations such as interpolation, 
	smoothing etc. The aux() array is an array of EST_Vals which 
	allows points to be int, float or a string. 

	The following functions can be used to access to auxiliary
	track information. Most of these functions can be used for
	reading or writing to this point, thus

	tr.aux(10, "voicing") = 1;

	can be used to set the 10th frame of the "voicing" channel and

	cout << tr.a(10, "voicing");

	can be used to print the same information. Most of these functions
	have a const equivalent for helping the compiler in
	read only operations.  

	Auxiliary channels are usually accessed by name rather than
	numerical index. The names are set using the set_aux_channel_names()
	function.
    */

    //@{

    EST_Val &aux(int i, int c);
    EST_Val &aux(int i, int c) const;

    EST_Val &aux(int i, const char *name);
    EST_Val aux(int i, const char *name) const
	{ return ((EST_Track *)this)->aux(i, name); }

    EST_Val &aux(int i, EST_String name) 
	{ return aux(i, (const char *)name); }

    EST_Val aux(int i, EST_String name) const
	{ return ((EST_Track *)this)->aux(i, (const char *)name); }

    //@}

    /** @name File i/o functions
     */

    //@{

    /** Load a file called {\tt name} into the track. 
	The load function attempts to
	automatically determine which file type is being loaded from the
	file's header. If no header is found, the function assumes the
	file is ascii data, with a fixed frame shift, arranged with rows
	representing frames and columns channels. In this case, the
	frame shift must be specified as an argument to this function.
        For those file formats which don't contain provision for specifying
	an initial time (e.g. HTK, or ascii...), the {\tt startt} parameter
	may be specified.
    */
    EST_read_status load(const EST_String name, float ishift = 0.0, float startt = 0.0);

    /** Load character data from an already opened tokenstream {\tt ts} 
	into the track. 
	The load function attempts to
	automatically determine which file type is being loaded from the
	file's header. If no header is found, the function assumes the
	file is ascii data, with a fixed frame shift, arranged with rows
	representing frames and columns channels. In this case, the
	frame shift must be specified as an argument to this function
        For those file formats which don't contain provision for specifying
	an initial time (e.g. HTK, or ascii...), the {\tt startt} parameter
	may be specified.
    */
    EST_read_status load(EST_TokenStream &ts, float ishift = 0.0, float startt = 0.0);

    /** Load a file called {\tt name} of format {\tt type} 
	into the track. If no header is found, the function assumes the
	file is ascii data, with a fixed frame shift, arranged with rows
	representing frames and columns channels. In this case, the
	frame shift must be specified as an argument to this function
        For those file formats which don't contain provision for specifying
	an initial time (e.g. HTK, or ascii...), the {\tt startt} parameter
	may be specified.
    */
    EST_read_status load(const EST_String name, const EST_String type, 
			 float ishift = 0.0, float startt = 0.0 );

    /** Save the track to a file {\tt name} of format {\tt type}. */
    EST_write_status save(const EST_String name, 
			  const EST_String EST_filetype = "");

    /** Save the track to a already opened file pointer{\tt FP} 
	and write a file of format {\tt type}. */
    EST_write_status save(FILE *fp,
			  const EST_String EST_filetype = "");

    //@}

    /** @name Utility functions */

    //@{
    /// returns true if no values are set in the frame 
    int empty() const;
    
    /// set frame i to be a break
    void set_break(int i);
    /// set frame i to be a value
    void set_value(int i);
    /// return true if frame i is a value
    int val(int i) const;
    /// return true if frame i is a break
    int track_break(int i) const { return (p_is_val(i)); }

    /** starting at frame i, return the frame index of the first
	value frame before i. If frame i is a value, return i */
    int prev_non_break(int i) const;

    /** starting at frame i, return the frame index of the first
	value frame after i. If frame i is a value, return i */
    int next_non_break(int i) const;

    /// return the frame index nearest time t
    int index(float t) const;		

    /// return the frame index before time t
    int index_below(float x) const;

    /// return number of frames in track
    int num_frames() const {return p_values.num_rows();}

    /// return number of frames in track
    int length() const { return num_frames(); }

    /// return number of channels in track
    int num_channels() const {return p_values.num_columns();}

    /// return number of auxiliary channels in track
    int num_aux_channels() const {return p_aux.num_columns();}

    void add_trailing_breaks();
    void rm_trailing_breaks();
    /** If the contour has multiple break values between sections
	containing values, reduce the break sections so that each has
	a single break only. */
    void rm_excess_breaks();	     

    /// return true if track has equal (i.e. fixed) frame spacing */
    bool equal_space() const {return p_equal_space;}

    /**return true if track has only single breaks between value sections */
    bool single_break() const {return p_single_break;}

    void set_equal_space(bool t) {p_equal_space = t;}
    void set_single_break(bool t) {p_single_break = t;}


    //@}


    EST_Track& operator = (const EST_Track& a);
    EST_Track& operator+=(const EST_Track &a); // add to existing track
    EST_Track& operator|=(const EST_Track &a); // add to existing track in parallel
    friend ostream& operator << (ostream& s, const EST_Track &tr);

    // Default constructor
    EST_Track(int num_frames, EST_TrackMap &map);

    // assign a known description to a track
    void assign_map(EST_TrackMap::P map);
    void assign_map(EST_TrackMap &map) { assign_map(&map); }

    // create a description for an unknown track
    void create_map(EST_ChannelNameMap &names);
    void create_map(void) { create_map(EST_default_channel_names); }

    EST_TrackMap::P map() const { return p_map; }

    int channel_position(EST_ChannelType type, int offset=0) const;



    // return amplitude of point i, channel type c (plus offset)
    float &a(int i, EST_ChannelType c, int offset=0);
    float  a(int i, EST_ChannelType c, int offset=0) const
	{ return ((EST_Track *)this)->a(i,c, offset); }

    // return amplitude at time t, channel type c
    float &a(float t, EST_ChannelType c, EST_InterpType interp=it_nearest);
    float  a(float t, EST_ChannelType c, EST_InterpType interp=it_nearest) const
	{ return ((EST_Track *)this)->a(t, c, interp); }

    float &operator() (int i, EST_ChannelType c)       { return a(i,c); }
    float  operator() (int i, EST_ChannelType c) const { return a(i,c); }

    float &t_offset()			   { return p_t_offset; }
    float t_offset() const		   { return p_t_offset; }


    EST_read_status load_channel_names(const EST_String name);
    EST_write_status save_channel_names(const EST_String name);

    const EST_String channel_name(int channel, const EST_ChannelNameMap &map, int strings_override=1) const;
    const EST_String channel_name(int channel, int strings_override=1) const 
	{ return channel_name(channel, EST_default_channel_names, strings_override); }

    const EST_String aux_channel_name(int channel) const 
	{ return p_aux_names(channel);}

    void resize(int num_frames, EST_TrackMap &map);

    EST_TrackFileType file_type() const {return (EST_TrackFileType)f_Int("file_type",0);}


    void set_file_type(EST_TrackFileType t) {f_set("file_type", (int)t);}


    bool has_channel(EST_ChannelType type) const 
	{ int cp = channel_position(type);
	return cp>=0; }

    // Frame iteration support

protected:
  class IPointer_f { 
  public:
    EST_Track *frame; int i; 
    IPointer_f();
    IPointer_f(const IPointer_f &p);
    ~IPointer_f();
  };

  void point_to_first(IPointer_f &ip) const { ip.i = 0; }
  void move_pointer_forwards(IPointer_f &ip) const { ip.i++; }
  bool points_to_something(const IPointer_f &ip) const { return ip.i <num_frames(); }
  EST_Track &points_at(const IPointer_f &ip) { sub_track(*(ip.frame), ip.i, 1);
					 return *(ip.frame); }

  friend class EST_TIterator< EST_Track, IPointer_f, EST_Track >;
  friend class EST_TRwIterator< EST_Track, IPointer_f, EST_Track >;

public:
  typedef EST_Track Entry;
  typedef EST_TIterator< EST_Track, IPointer_f, EST_Track > Entries;
  typedef EST_TRwIterator< EST_Track, IPointer_f, EST_Track > RwEntries;

};

// list of tracks in serial
typedef EST_TList<EST_Track> EST_TrackList;


#endif /* __Track_H__ */
