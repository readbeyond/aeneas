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
 /*                                                                       */
 /*             Author :  Paul Taylor and Alan W Black                    */
 /*          Rewritten :  Richard Caley                                   */
 /* -------------------------------------------------------------------   */
 /*             EST_Wave Class header file                                */
 /*                                                                       */
 /*************************************************************************/

#ifndef __Wave_H__
#define __Wave_H__

#include <cstdio>
#include "EST_Featured.h"
#include "EST_rw_status.h"
#include "EST_types.h"

class EST_Track;
class EST_String;
class EST_TokenStream;


/** A class for storing digital waveforms. The waveform is stored as
an array of 16 bit shorts. Multiple channels are supported, but if no
channel information is given the 0th channel is accessed.
<p>

The waveforms can be of any sample rate, and can be changed to another
sampling rate using the <tt>resample</tt> function.

*/

class EST_Wave : public EST_Featured
{
protected:
  EST_SMatrix p_values;

  int p_sample_rate;

  void default_vals(int n=0, int c=1);
  void free_wave();
  void copy_data(const EST_Wave &w);
  void copy_setup(const EST_Wave &w);

public:

  static const int default_sample_rate;
  static const int default_num_channels;

  /// default constructor
  EST_Wave();
  /// copy constructor
  EST_Wave(const EST_Wave &a);

  EST_Wave(int n, int c, int sr);

  /// Construct from memory supplied by caller
  EST_Wave(int samps, int chans,
	   short *memory, int offset=0, int sample_rate=default_sample_rate, 
	   int free_when_destroyed=0);
  
  ~EST_Wave();
    

  /**@name Access functions for finding amplitudes of samples */
  //@{

  /** return amplitude of sample <tt>i</tt> from channel <tt>
      channel</tt>.  By default the 0th channel is selected. This
      function can be used for assignment.
  */
  short &a(int i, int channel = 0);
  short a(int i, int channel = 0) const;
  INLINE short &a_no_check(int i, int channel = 0)
        { return p_values.a_no_check(i,channel); }
  INLINE short a_no_check(int i, int channel = 0) const
        { return p_values.a_no_check(i,channel); }
  INLINE short &a_no_check_1(int i, int channel = 0)
        { return p_values.a_no_check_1(i,channel); }
  INLINE short a_no_check_1(int i, int channel = 0) const
        { return p_values.a_no_check_1(i,channel); }

  
  /** explicit set_a, easier to wrap than assignment
   */
  INLINE short set_a(int i, int channel = 0, short val = 0)
  { return a(i,channel) = val; }

  /** return amplitude of sample <tt>i</tt> from channel <tt>
      channel</tt>.  By default the 0th channel is selected.
  */
  short operator()(int i, int channel) const
    { return a(i,channel); }

  /** return amplitude of sample <tt>i</tt> from channel 0. 
    */
  short operator()(int i) const
    { return a(i,0); }
       
  /** Version of a() that returns zero if index is out of array
      bounds.  This is particularly useful in signal processing when
      you want to have windows going off the end of the waveform.  */
  short &a_safe(int i, int channel = 0);

  /// return the time position in seconds of the ith sample
  float t(int i) const { return (float)i/(float)p_sample_rate; }
  //@}

  /**@name Information functions */
  //@{
  /// return the number of samples in the waveform
  int num_samples() const { return p_values.num_rows();}
  /// return the number of channels in the waveform
  int num_channels() const { return p_values.num_columns(); }
  /// return the sampling rate (frequency)
  int sample_rate() const { return p_sample_rate; }
  /// Set sampling rate to <tt>n</tt>
  void set_sample_rate(const int n){p_sample_rate = n;}
  /// return the size of the waveform, i.e. the number of samples.
  int length() const { return num_samples();}
  /// return the time position of the last sample.
  float end(){ return t(num_samples()-1); }

  /// Can we look N samples to the left?
  bool have_left_context(unsigned int n) const
    { return p_values.have_rows_before(n); }

  /** returns the file format of the file from which the waveform
      was read. If the waveform has not been read from a file, this is set
      to the default type */

  EST_String sample_type() const { return f_String("sample_type","short"); }
  void set_sample_type(const EST_String t) { f_set("sample_type", t); }

  EST_String file_type() const { return f_String("file_type","riff"); }
  void set_file_type(const EST_String t) { f_set("file_type", t); }

  /// A string identifying the waveform, commonly used to store the filename
  EST_String name() const { return f_String("name"); }

  /// Sets name.
  void set_name(const EST_String n){ f_set("name", n); }

  //@}

  const EST_SMatrix &values() const { return p_values; }
  EST_SMatrix &values() { return p_values; }

  /**@name Waveform manipulation functions */
  //@{

  /// resize the waveform 
  void resize(int num_samples, int num_channels = EST_ALL, int set=1) 
    { p_values.resize(num_samples, num_channels, set); }

  /// Resample waveform to <tt>rate</tt>
  void resample(int rate);

  /** multiply all samples by a factor <tt>gain</tt>. This checks for
      overflows and puts them to the maximum positive or negative value
      as appropriate.
  */
  void rescale(float gain,int normalize=0);

  // multiply samples by a factor contour.  The factor_contour track
  // should contains factor targets at time points throughout the wave,
  // between which linear interpolation is used to calculate the factor
  // for each sample.
  void rescale( const EST_Track &factor_contour );

  /// clear waveform and set size to 0.
  void clear() {resize(0,EST_ALL);}

  void copy(const EST_Wave &from);

  void fill(short v=0, int channel=EST_ALL);

  void empty(int channel=EST_ALL) { fill(0,channel); }

  void sample(EST_TVector<short> &sv, int n)
    { p_values.row(sv, n); }
  void channel(EST_TVector<short> &cv, int n)
    { p_values.column(cv, n); }

  void copy_channel(int n, short *buf, int offset=0, int num=EST_ALL) const
    { p_values.copy_column(n, buf, offset, num); } 
  void copy_sample(int n, short *buf, int offset=0, int num=EST_ALL) const
    {  p_values.copy_row(n, buf, offset, num); } 

  void set_channel(int n, const short *buf, int offset=0, int num=EST_ALL)
    { p_values.set_column(n, buf, offset, num); }
  void set_sample(int n, const short *buf, int offset=0, int num=EST_ALL)
    { p_values.set_row(n, buf, offset, num); }


  void sub_wave(EST_Wave &sw, 
		int offset=0, int num=EST_ALL,
		int start_c=0, int nchan=EST_ALL);

  void sub_wave(EST_Wave &sw, 
		int offset=0, int num=EST_ALL,
		int start_c=0, int nchan=EST_ALL) const
    { ((EST_Wave *)this)->sub_wave(sw, offset, num, start_c, nchan); }

  //@}

  /**@name File i/o functions */
  //@{

  /** Load a file into the waveform. The load routine attempts to
      automatically determine which file type is being loaded.  A
      portion of the waveform can be loaded by setting <tt>
      offset</tt> to the sample position from the beginning and
      <length> to the number of required samples after this.  */

  EST_read_status load(const EST_String filename, 
		       int offset=0, 
		       int length = 0,
		       int rate = default_sample_rate);

  EST_read_status load(EST_TokenStream &ts,
		       int offset=0, 
		       int length = 0,
		       int rate = default_sample_rate);

  EST_read_status load(const EST_String filename, 
		       const EST_String filetype,
		       int offset=0, 
		       int length = 0,
		       int rate = default_sample_rate);

  EST_read_status load(EST_TokenStream &ts,
		       const EST_String filetype,
		       int offset=0, 
		       int length = 0,
		       int rate = default_sample_rate);

  /** Load a file of type <tt>filetype</tt> into the waveform. This
      can be used to load unheadered files, in which case the fields
      <tt>sample_rate, sample_type, bo</tt> and <tt>nc</tt> are used
      to specify the sample rate, type, byte order and number of
      channels.  A portion of the waveform can be loaded by setting
      <tt> offset</tt> to the sample position from the beginning and
      <length> to the number of required samples after this.
  */

  EST_read_status load_file(const EST_String filename, 
			    const EST_String filetype, int sample_rate, 
			    const EST_String sample_type, int bo, int nc,
			    int offset = 0, int length = 0);
  EST_read_status load_file(EST_TokenStream &ts,
			    const EST_String filetype, int sample_rate, 
			    const EST_String sample_type, int bo, int nc,
			    int offset = 0, int length = 0);

  /* Save waveform to a file called <tt>filename</tt> of file
     format <tt>EST_filetype</tt>.
  */
  EST_write_status save(const  EST_String filename, 
			const EST_String EST_filetype = "");

  EST_write_status save(FILE *fp,
			const EST_String EST_filetype = "");

  EST_write_status save_file(const EST_String filename, 
			     EST_String filetype,
			     EST_String sample_type, int bo, const char *mode = "wb");

  EST_write_status save_file(FILE *fp,
			     EST_String filetype,
			     EST_String sample_type, int bo);

  EST_write_status save_file_header(FILE *fp,
				     EST_String ftype,
				     EST_String stype, int obo);
  EST_write_status save_file_data(FILE *fp,
				     EST_String ftype,
				     EST_String stype, int obo);
  //@}

  /// Assignment operator
  EST_Wave& operator = (const EST_Wave& w);
  /** Add to existing wave in serial. Waveforms must have the same
      number of channels.
  */
  EST_Wave& operator +=(const EST_Wave &a);
  /** Add wave in parallel, i.e. make wave <tt>a</tt> become new
      channels in existing waveform.
  */
  EST_Wave& operator |=(const EST_Wave &a);

  /// print waveform
  friend ostream& operator << (ostream& p_values, const EST_Wave &sig);

  // integrity check *** debug
  void integrity() const { p_values.integrity() ; }

};

typedef EST_TList<EST_Wave> EST_WaveList;

int operator != (EST_Wave a, EST_Wave b);
int operator == (EST_Wave a, EST_Wave b);

#endif /* __Wave_H__ */
